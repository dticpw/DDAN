from math import pi, log
from functools import wraps

import torch
from torch import dropout, nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from agg_block.pos_encoding import build_position_encoding
from agg_block.attention import *
    

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

class AggregationBlock(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        pos_enc_type = 'none',
        pre_norm = True,
        post_norm = True, 
        activation = 'geglu',
        last_ln = False,
        ff_mult = 4,
        more_dropout = False,
        xavier_init = False,
        query_fixed = False,
        query_xavier_init = False,
        query_type = 'learned',
        encoder_isab = False,
        first_order=False
    ):
        """
        Args:
            depth: Depth of net.
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of element slots
            latent_dim: slot dimension.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.num_latents = num_latents
        self.query_type = query_type
        self.latent_dim = latent_dim
        self.encoder_isab = encoder_isab
        self.first_order = first_order
        
        if self.query_type == 'learned':
            self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim))
            if query_fixed: # ？
                self.latents.requires_grad = False
            if query_xavier_init:
                nn.init.xavier_normal_(self.latents)
        elif self.query_type == 'slot':
            self.slots_mu = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
            self.slots_log_sigma = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
        else:
            raise NotImplementedError
        
        assert (pre_norm or post_norm)
        self.prenorm = PreNorm if pre_norm else lambda dim, fn, context_dim=None: fn
        self.postnorm = PostNorm if post_norm else nn.Identity # I()
        ff = FeedForward
        
        # * decoder cross attention layers
        get_cross_attn = \
            lambda: self.prenorm( # 返回一个前馈归一化层self.prenorm，包含了一个 CA模块
                latent_dim, 
                Attention(
                    latent_dim, input_dim, # 特征维度，输入维度
                    heads = 4, dim_head = 512, dropout = attn_dropout, more_dropout = more_dropout, xavier_init = xavier_init
                ), 
                context_dim = input_dim)
        # ff 前馈归一化层，     # 网络宽度乘数 ff_mult
        get_cross_ff = lambda: self.prenorm(latent_dim, ff(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_cross_postnorm = lambda: self.postnorm(latent_dim)
        
        get_cross_attn, get_cross_ff = map(cache_fn, (get_cross_attn, get_cross_ff)) # cache_fn来缓存函数的结果 # 还是这两个函数，但是更快

        self.layers = nn.ModuleList([])
        
        for i in range(depth): # 4 # append四次，每次都有四个小层
            should_cache = i >= 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args), # CA
                get_cross_postnorm(), # I()
                get_cross_ff(**cache_args), # 前馈呀，这么经典的结构
                get_cross_postnorm() # I()
            ]))

        # Last FC layer
        assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not post_norm else nn.Identity() # nn.LayerNorm(latent_dim)
        )
        
        self.encoder_output_holder = nn.Identity()
        self.decoder_output_holder = nn.Identity()

        #### 0926
        self.embed_size = 1024

        ####
        
    def get_queries(self, b):
        if self.query_type == 'learned': # 1
            ret = repeat(self.latents, 'n d -> b n d', b = b) # query便是slot # (4,dim) -> (bs,4,dim)
        elif self.query_type == 'slot':
            slots_init = torch.randn((b, self.num_latents, self.latent_dim)).cuda()
            ret = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        return ret

    # def forward(self, data, mask = None): # (bs,34,1,dim) # mask:对pad部分取了true
    #     b, *axis, _, device = *data.shape, data.device # 这里为什么有 * 
    #     assert len(axis) == self.input_axis, 'input data must have the right number of axis'

    #     # concat to channels of data and flatten axis
    #     pos = self.pos_enc(data) # None
        
    #     data = rearrange(data, 'b ... d -> b (...) d') # (bs,34,1,dim) -> (bs,34,dim) # 还能这样
        
    #     x = self.get_queries(b).type_as(data) # 这便是slot (bs,34,dim)

    #     for i, (cross_attn, pn1, cross_ff, pn2) in enumerate(self.layers):
    #         x = cross_attn(x, context = data, mask = mask, k_pos = pos, q_pos = None) + x
    #         x = pn1(x) # I()
    #         x = cross_ff(x) + x
    #         x = pn2(x) # I()
            
    #     x = self.decoder_output_holder(x) # I()
    #     return self.last_layer(x)

    ### 你的问题出在了！两个模态公用一个agg模块，这是有参数的
    def forward(self, data, mask = None): # (bs,34,1,dim) # mask:对pad部分取了true的padmask
        b, *axis, _, device = *data.shape, data.device # 这里为什么有 * 
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data) # None
        
        data = rearrange(data, 'b ... d -> b (...) d') # (bs,34,1,dim) -> (bs,34,dim) # 还能这样
        
        # 
        # data = l2norm(data)
        #####1006
        softA,img_pat = rank_A_mul_ESA(data)
        softA = self.linear_softA(softA)
        img_rank_mul = data * softA
        #####


        cap_glo1 = torch.mean(self.linear_ESA1(img_rank_mul),dim=1)
        cap_glo2 = torch.mean(self.linear_ESA2(img_rank_mul),dim=1)
        cap_glo3 = torch.mean(self.linear_ESA3(img_rank_mul),dim=1)
        cap_glo4 = torch.mean(self.linear_ESA4(img_rank_mul),dim=1)

        # features_in1 = self.linear_ESA1(img_rank_mul)
        # features_in1 = features_in1.masked_fill(mask == 0,-10000)
        # features_k_softmax1 = nn.Softmax(dim=1)(features_in1-torch.max(features_in1,dim=1)[0].unsqueeze(1)) # 这为什么全是同样的数？这还有用吗
        # cap_glo1 = torch.sum(features_k_softmax1 * data, dim=1) # (5bs,dim) # 存在着非常多的0值#为什么呢？
        cap_glo1 = l2norm(cap_glo1)

        # features_in2 = self.linear_ESA2(img_rank_mul)
        # features_in2 = features_in2.masked_fill(mask == 0,-10000)
        # features_k_softmax2 = nn.Softmax(dim=1)(features_in2-torch.max(features_in2,dim=1)[0].unsqueeze(1)) 
        # cap_glo2 = torch.sum(features_k_softmax2 * data, dim=1) 
        cap_glo2 = l2norm(cap_glo2)

        # features_in3 = self.linear_ESA3(img_rank_mul)
        # features_in3 = features_in3.masked_fill(mask == 0,-10000)
        # features_k_softmax3 = nn.Softmax(dim=1)(features_in3-torch.max(features_in3,dim=1)[0].unsqueeze(1)) 
        # cap_glo3 = torch.sum(features_k_softmax3 * data, dim=1) 
        cap_glo3 = l2norm(cap_glo3)

        # features_in4 = self.linear_ESA4(img_rank_mul)
        # features_in4 = features_in4.masked_fill(mask == 0,-10000)
        # features_k_softmax4 = nn.Softmax(dim=1)(features_in4-torch.max(features_in4,dim=1)[0].unsqueeze(1)) 
        # cap_glo4 = torch.sum(features_k_softmax4 * data, dim=1) 
        cap_glo4 = l2norm(cap_glo4)


        x = torch.stack((cap_glo1,cap_glo2,cap_glo3,cap_glo4), dim=1)

        x = nn.Softmax(dim=1)(x)

        x = self.decoder_output_holder(x) # I()
        return self.last_layer(x) # norm



def rank_A_mul_ESA(X, lengths=None, mask=None): 
    """
    input:
        X: 原始区域特征(bs,k,d)
    output:
        A: (bs,k,d) 记录着各区域特征值排名，并经过取倒数，之后会有MLP转化
        B: (d,) 1024维最大值的区域号，0到35
    """
    bs, k, dim = X.shape
    if lengths==None:
        lengths = torch.zeros(bs, dtype=int, device=X.device)+k
    A = torch.zeros((bs, k, dim), device=X.device)
    B = torch.zeros((bs, dim), device=X.device)
    for i in range(bs):
        feature_values = X[i, 0:lengths[i], :] # (k, dim)
        _, indices = torch.sort(feature_values, dim=-2, descending=True)
        _, indices2 = torch.sort(indices,dim=-2)
        A[i, 0:lengths[i], :] = indices2 
        B[i] = indices[0]
    # 到这里应该是0到35这样
    A=A+1
    if mask != None:
        A = A.masked_fill(mask==0,1e6)
    rankA = 1/A
    return rankA, B