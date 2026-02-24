import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision

from agg_block.agg_block import AggregationBlock

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange, reduce
from transformers import BertModel
from agg_block.attention import default
from agg_block.agg_block import rank_A_mul_ESA

def get_cnn(arch, pretrained):
    if arch == 'resnext_wsl':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    else:
        model = torchvision.models.__dict__[arch](pretrained=pretrained) 
    return model

# Problematic: could induce NaN loss
def l2norm_old(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0)
    if torch.cuda.is_available():
        ind = ind.cuda()
    mask = torch.tensor((ind >= lengths.unsqueeze(1))) if set_pad_to_one \
        else torch.tensor((ind < lengths.unsqueeze(1)))
    return mask.cuda() if torch.cuda.is_available() else mask

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
    # rankA = (torch.exp(-A))
    rankA = 1 / (A)
    return rankA, B


def variable_len_pooling(data, input_lens, reduction):
    if input_lens is None:
        if reduction =='avg':
            ret = reduce(data, 'h i k ->  h k', 'mean')
        elif reduction =='max':
            ret = reduce(data, 'h i k ->  h k', 'max')
        elif reduction =='avgmax':
            ret = reduce(data, 'h i k ->  h k', 'max')/2 + reduce(data, 'h i k ->  h k', 'mean')/2
        elif reduction == 'minmax':
            ret = reduce(data, 'h i k ->  h k', 'max')/2 + reduce(data, 'h i k ->  h k', 'min')/2
        else:
            raise NotImplementedError
    else:
        B, N, D = data.shape
        idx = torch.arange(N).unsqueeze(0).expand(B, -1).cuda()
        idx = idx < input_lens.unsqueeze(1)
        idx = idx.unsqueeze(2).expand(-1, -1, D)
        if reduction == 'avg':
            ret = (data * idx.float()).sum(1) / input_lens.unsqueeze(1).float()
        elif reduction == 'max': # 1 # 找到每个图像的有效区域特征的最大值，并将这些最大值存储在 ret 变量中。
            ret = data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0] # ~逻辑非 # 将 data 中 idx 为 False 的位置用一个非常小的值填充，之后进行max
        elif reduction == 'avgmax':
            ret = (data * idx.float()).sum(1) / input_lens.unsqueeze(1).float() +\
                data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0]
            ret /= 2
        elif reduction == 'minmax':
            ret = data.masked_fill(~idx, torch.finfo(data.dtype).max).min(1)[0] +\
                data.masked_fill(~idx, -torch.finfo(data.dtype).max).max(1)[0]
            ret /= 2
        else:
            raise NotImplementedError
    return ret # (bs,dim)

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        if len(x.shape) == 4:
            x = rearrange(x, 'h i j k -> h i (j k)')
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1,2,0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)
        
        output = torch.bmm(attn.transpose(1,2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, local_feat, global_feat, pad_mask=None, lengths=None):
        residual, attn = self.attention(local_feat, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            global_feat = global_feat.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(global_feat + residual)
        return out, attn, residual
        
        
class SetPredictionModule(nn.Module):
    def __init__(
        self, 
        num_embeds, 
        d_in, 
        d_out, 
        axis, 
        pos_enc, 
        query_dim,
        args
    ):
        super(SetPredictionModule, self).__init__()
        self.num_embeds = num_embeds
        self.residual_norm = nn.LayerNorm(d_out) if args.spm_residual_norm else nn.Identity() # nn.LayerNorm(1024)
        self.spm_residual = args.spm_residual # 1
        self.res_act_local_dropout = nn.Dropout(args.res_act_local_dropout) # 0
        self.res_act_global_dropout = nn.Dropout(args.res_act_global_dropout) # 0
        self.fc = nn.Linear(d_out, 1024) if args.spm_residual_fc else nn.Identity() # I()
        self.res_only_norm = args.res_only_norm # 1
        
        self.agg_block = AggregationBlock(
            depth = args.spm_depth, # 4
            input_channels = d_in, # 图像1024，文本300
            input_axis = axis, # 图像2，文本1
            num_latents = num_embeds, # 4
            latent_dim = query_dim, # 1024
            num_classes = d_out, # 1024
            attn_dropout = args.dropout, # 0.1
            ff_dropout = args.dropout,
            weight_tie_layers = args.spm_weight_sharing, # 1
            pos_enc_type = pos_enc, # 图像'none'，文本'sine'
            pre_norm = args.spm_pre_norm, # 1
            post_norm=args.spm_post_norm, # 0
            activation = args.spm_activation, # gelu
            last_ln = args.spm_last_ln, # 1
            ff_mult = args.spm_ff_mult, # 4.0
            more_dropout = args.spm_more_dropout, # 0
            xavier_init = args.spm_xavier_init, # 0
            query_fixed = args.query_fixed, # 0
            query_xavier_init = args.query_xavier_init, # 0
            query_type = 'slot' if args.query_slot else 'learned', # 'learned'
            first_order=args.first_order # 0
        )
        
    def forward(self, local_feat, global_feat=None, pad_mask=None, lengths=None): # (bs,34,1,dim),(bs,dim),(bs,34),(200)
        # set_prediction = self.agg_block(local_feat, mask=pad_mask)  # (bs,4,dim)
        set_prediction = local_feat  # (bs,4,dim)
        set_prediction = self.res_act_local_dropout(set_prediction) # Dropout(0.0)
        global_feat = global_feat.unsqueeze(1).repeat(1, self.num_embeds, 1)    # (bs,4,dim)
        # out = self.residual_norm(self.res_act_global_dropout(global_feat)) + set_prediction # 图中的 res
        out = set_prediction # 图中的 res
        out = self.fc(out)  # I()
        
        return out, None, set_prediction # set_prediction 是lastlayer的


class VSE(nn.Module):
    def __init__(self, word2idx, opt):
        super(VSE, self).__init__()

        self.mil = opt.img_num_embeds >= 1 or opt.txt_num_embeds >= 1
        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(word2idx, opt) if not opt.use_bert else EncoderTextBERT(opt)
        self.amp = opt.amp

    def forward(self, images, sentences, img_len, txt_len): # (200,34,2048),((vocab编号)1000,68), ((还真有img_len啊)200),(1000)
        with torch.cuda.amp.autocast(enabled=self.amp):
            img_emb, img_attn, img_ori = self.img_enc(images, img_len)     # (bs, 4, dim), none, (bs, 4, 1024)
            txt_emb, txt_attn, txt_ori = self.txt_enc(sentences, txt_len)  # (5bs, 4, dim), none, (5bs, 4, 1024)
            return img_emb, txt_emb, img_attn, txt_attn, img_ori, txt_ori
        
        
class SequenceBN(nn.Module):
    def __init__(self, dim, affine=True):
        super(SequenceBN, self).__init__()
        self.bn = nn.BatchNorm1d(dim, affine=affine)
    
    def forward(self, x):
        shape = x.shape
        x = self.bn(rearrange(x, '... d -> (...) d')).reshape(shape)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = SequenceBN(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc2(self.relu(self.bn(self.fc1(x))))
        return x

class EncoderImage(nn.Module):

    def __init__(self, opt, shared_memory=None, shared_query=None):
        super(EncoderImage, self).__init__()
        self.butd = 'butd' in opt.data_name # True
        embed_size, num_embeds = opt.embed_size, opt.img_num_embeds # dim(1024), 4
        self.grid_drop_prob = opt.grid_drop_prob # 0
        self.global_feat_holder = nn.Identity()

        if not self.butd:
            # Backbone CNN
            self.cnn = get_cnn(opt.cnn_type, True)
            local_feat_dim = self.local_feat_dim = self.cnn.fc.in_features
            self.cnn.avgpool = nn.Sequential()
            self.cnn.fc = nn.Sequential()
        else: # 1
            self.cnn = nn.Identity()
            local_feat_dim = self.local_feat_dim = 2048

        self.img_1x1_dropout = nn.Dropout(opt.img_1x1_dropout) # nn.Dropout(0.1)
        self.fc = nn.Linear(local_feat_dim, embed_size) # 2048 -> 1024

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) ## 将最后两维缩减到1x1
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        ###
        # self.linear_ESA = nn.Sequential(
        #     nn.Linear(embed_size, embed_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(embed_size, embed_size)
        # )
        self.linear_softA = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )

        # self.linear_ESA1 = nn.Sequential(
        #     nn.ReLU()
        # )
        # self.linear_ESA2 = nn.Sequential(
        #     nn.Linear(embed_size, embed_size)
        # )
        # self.linear_ESA3 = nn.Sequential(
        #     nn.Linear(embed_size, embed_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(embed_size, embed_size)
        # )
        # self.linear_ESA4 = nn.Sequential(
        #     nn.Linear(embed_size, embed_size * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(embed_size * 2, embed_size)
        # )
        self.linear_ESA1 = nn.Sequential(
            nn.Linear(embed_size, embed_size)
        )
        self.linear_ESA2 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.linear_ESA3 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.linear_ESA4 = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_size * 2, embed_size),
        )
        # Temperature parameters for different aggregation strategies
        self.tau_selective = opt.tau_selective      # Channel 2
        self.tau_comprehensive = opt.tau_comprehensive  # Channel 3
        
        self.fc = nn.Linear(local_feat_dim, opt.embed_size)
        if opt.gpo_1x1: # 1
            self.mlp = MLP(opt.embed_size, opt.embed_size // 2, opt.embed_size)
            self.spm_fc = lambda x: self.img_1x1_dropout(self.mlp(self.fc(x)) + self.fc(x)) # dim: 2048 -> 1024
        else:
            self.mlp = None
            self.spm_fc = self.fc if opt.spm_1x1 else nn.Identity()
        
        if 'slot' == opt.arch: # 1
            self.spm = SetPredictionModule(
                num_embeds=num_embeds, 
                d_in=opt.spm_input_dim if opt.spm_1x1 else 2048, 
                d_out=embed_size, 
                axis=2, 
                pos_enc=opt.spm_img_pos_enc_type, 
                query_dim=opt.spm_query_dim,
                args=opt
            )
        elif 'pvse' == opt.arch: # 0
            raise Exception('pvse == opt.arch') ##
        #     self.spm = PIENet(num_embeds, opt.spm_input_dim, embed_size, opt.spm_input_dim // 2)
            
        self.residual = opt.spm_residual # 1
        assert opt.img_res_first_fc or opt.img_res_last_fc # first_fc
        assert opt.img_res_pool in ['avg', 'max'] # max
        
        self.img_res_pool = opt.img_res_pool
        self.inter_dim = opt.spm_input_dim if opt.img_res_first_fc else local_feat_dim
        
        self.residual_first_fc = self.spm_fc if opt.img_res_first_fc else nn.Identity() # spm_fc: Droupout(0.1)(mlp(fc(x))+fc(x))
        self.residual_last_fc = nn.Linear(self.inter_dim, embed_size) if opt.img_res_last_fc else nn.Identity()
        self.residual_first_pool = variable_len_pooling if opt.img_res_first_pool else lambda x, y, z: x # lambda x, y, z: x
        self.residual_after_pool = variable_len_pooling if not opt.img_res_first_pool else lambda x, y, z: x # variable_len_pooling
        if opt.img_res_last_pool: # 0
            self.residual_first_pool = lambda x, y, z: x
            self.residual_after_pool = lambda x, y, z: x
        self.residual_last_pool = variable_len_pooling if opt.img_res_last_pool else lambda x, y, z: x # x
        
        assert opt.spm_img_pos_enc_type == 'none' if self.butd else True

        if opt.spm_xavier_init: # 0
            self.init_weights()
        for idx, param in enumerate(self.cnn.parameters()): # 0 ?
            param.requires_grad = opt.img_finetune

    #### 0921
    # def residual_connection(self, x, l):
    #     x = rearrange(x, 'h i j k -> h (i j) k') # (bs,34,1,2048) -> (bs,34,2048)
    #     x = self.residual_first_fc(self.residual_first_pool(x, l, self.img_res_pool)) # Droupout(0.1)(mlp(fc(x))+fc(x)) # (bs,34,1024)
    #     x = self.residual_last_fc(self.residual_after_pool(x, l, self.img_res_pool)) # 就只是一个max_pooling，对不同region
    #     x = self.residual_last_pool(x, l, self.img_res_pool) # 直接返回了 x 
    #     return x # 也就是 Droupout(0.1)(mlp(fc(x))+fc(x)) 接个 maxpool # (bs,dim)
    def residual_connection(self, x, l): 
        x = rearrange(x, 'h i j k -> h (i j) k') # (bs,34,1,2048) -> (bs,34,2048)
        x = self.residual_last_fc(self.residual_after_pool(x, l, self.img_res_pool))
        x = self.residual_last_pool(x, l, self.img_res_pool)
        return x # 
    ####
    
    def init_weights(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.mlp.apply(fn)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        

    
    def forward(self, images, lengths=None): # (bs,34,2048)

        #### loc
        img_emb = self.spm_fc(images) # (bs,34,2048) # 修改后是1024
        pad_mask = get_pad_mask(images.shape[1], lengths, True) # (200, 34)
        mask = pad_mask.unsqueeze(2).expand(-1,-1,img_emb.shape[-1])
        #### 241005


        img_emb_detach = img_emb.detach()
        # softA, img_pat, count_pat = rank_A_mul_ESA(img_emb_detach) ## 0.到最高的1（或2？已经忘了 ## (ba,k,d), (bs,d)
        softA, img_pat  = rank_A_mul_ESA(img_emb_detach)
        softA = self.linear_softA(softA)
        img_rank_mul = img_emb * softA
        ####

        posmask = ~mask
        features_in1 = self.linear_ESA1(img_rank_mul)
        features_in1 = features_in1.masked_fill(posmask == 0,-10000)
        if self.training:
            rand_list_1 = torch.rand(img_emb.size(0), img_emb.size(1)).to(img_emb.device)
            mask1 =(rand_list_1 >= 0.2).unsqueeze(-1)
            features_in1 = features_in1.masked_fill(mask1 == 0,-10000)
        features_k_softmax1 = nn.Softmax(dim=1)(features_in1-torch.max(features_in1,dim=1)[0].unsqueeze(1)) # 这为什么全是同样的数？这还有用吗
        img_glo1 = torch.sum(features_k_softmax1 * img_emb, dim=1)

        features_in2 = self.linear_ESA2(img_rank_mul)
        features_in2 = features_in2 / self.tau_selective ## scale 20260218
        features_in2 = features_in2.masked_fill(posmask == 0,-10000)
        if self.training:
            rand_list_2 = torch.rand(img_emb.size(0), img_emb.size(1)).to(img_emb.device)
            mask2 =(rand_list_2 >= 0.2).unsqueeze(-1)
            features_in2 = features_in2.masked_fill(mask2 == 0,-10000)
        features_k_softmax2 = nn.Softmax(dim=1)(features_in2-torch.max(features_in2,dim=1)[0].unsqueeze(1)) 
        img_glo2 = torch.sum(features_k_softmax2 * img_emb, dim=1) 

        features_in3 = self.linear_ESA3(img_rank_mul)
        features_in3 = features_in3 / self.tau_comprehensive
        features_in3 = features_in3.masked_fill(posmask == 0,-10000)
        if self.training:
            rand_list_3 = torch.rand(img_emb.size(0), img_emb.size(1)).to(img_emb.device)
            mask3 =(rand_list_3 >= 0.2).unsqueeze(-1)
            features_in3 = features_in3.masked_fill(mask3 == 0,-10000)
        features_k_softmax3 = nn.Softmax(dim=1)(features_in3-torch.max(features_in3,dim=1)[0].unsqueeze(1)) 
        img_glo3 = torch.sum(features_k_softmax3 * img_emb, dim=1) 

        features_in4 = self.linear_ESA4(img_rank_mul)
        # features_in4 = features_in4 + img_rank_mul  # Residual
        features_in4 = features_in4.masked_fill(posmask == 0,-10000)
        if self.training:
            rand_list_4 = torch.rand(img_emb.size(0), img_emb.size(1)).to(img_emb.device)
            mask4 =(rand_list_4 >= 0.1).unsqueeze(-1)
            features_in4 = features_in4.masked_fill(mask4 == 0,-10000)
        features_k_softmax4 = nn.Softmax(dim=1)(features_in4-torch.max(features_in4,dim=1)[0].unsqueeze(1)) 
        img_glo4 = torch.sum(features_k_softmax4 * img_emb, dim=1) 

        img_glo_cro = x = torch.stack((img_glo1,img_glo2,img_glo3,img_glo4), dim=1)
        # img_glo_r = img_glo_cro
        img_glo_soft = nn.Softmax(dim=1)(img_glo_cro)
        img_glo_cro = nn.Softmax(dim=1)(img_glo_cro) + img_glo_cro
        img_glo_cro = l2norm(img_glo_cro) # (bs,4,dim)

        img_glo = torch.sum(img_glo_cro,dim=1)
            
        # out_nxn = rearrange(out_nxn, 'h i j k -> h j k i')  # (bs,34,1,2048)
        if self.grid_drop_prob > 0: # 0
            out_nxn, lengths = self.grid_feature_drop(out_nxn)
        
        out = img_glo_cro
        attn = None

        out = l2norm(out) # (bs,4,1024)

        img_ori = l2norm(img_emb)

        return out, attn, img_glo_soft



class EncoderText(nn.Module):

    def __init__(self, word2idx, opt, shared_memory=None, shared_query=None):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_size, num_embeds = \
            opt.wemb_type, opt.word_dim, opt.embed_size, opt.txt_num_embeds # glove,300,1024,4

        self.embed_size = embed_size # 1024
        self.use_attention = opt.txt_attention # 1

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim) # (8482,300)形状的映射表
        self.embed.weight.requires_grad = opt.txt_finetune # 1 # 这为什么需要grad

        # Sentence embedding
        self.gpo_rnn = opt.gpo_rnn # 1
        self.rnn_hidden_size = embed_size if opt.gpo_rnn else embed_size // 2
        self.rnn = nn.GRU(word_dim, self.rnn_hidden_size, bidirectional=True, batch_first=True)
        
        self.txt_attention_input = opt.txt_attention_input # wemb # wemb到底是什么？
        self.txt_pooling = opt.txt_pooling # 'rnn'
        self.txt_pooling_fc = nn.Linear(embed_size, word_dim) if opt.txt_pooling_fc else nn.Identity() # I()
        assert self.txt_attention_input in ['wemb', 'rnn']
        assert self.txt_pooling in ['rnn', 'max']
        
        self.txt_attention_head = opt.arch # 'slot'
        self.txt_attention_input_dim = word_dim if self.txt_attention_input == 'wemb' \
            else embed_size
        self.residual = opt.spm_residual # 1


        # self.linear_ESA = nn.Sequential(
        #     nn.Linear(embed_size, embed_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(embed_size, embed_size)
        # )
        self.linear_softA = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )

        # self.linear_ESA1 = nn.Sequential(
        #     nn.ReLU()
        # )
        # self.linear_ESA2 = nn.Sequential(
        #     nn.Linear(embed_size, embed_size)
        # )
        # self.linear_ESA3 = nn.Sequential(
        #     nn.Linear(embed_size, embed_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(embed_size, embed_size)
        # )
        # self.linear_ESA4 = nn.Sequential(
        #     nn.Linear(embed_size, embed_size * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(embed_size * 2, embed_size)
        # )
        self.linear_ESA1 = nn.Sequential(
            nn.Linear(embed_size, embed_size)
        )
        self.linear_ESA2 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.linear_ESA3 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.linear_ESA4 = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_size * 2, embed_size),
        )
        self.tau_selective = opt.tau_selective      # Channel 2
        self.tau_comprehensive = opt.tau_comprehensive  # Channel 3

        
        if opt.arch == 'pvse':
            self.spm = PIENet(num_embeds, self.txt_attention_input_dim, embed_size, word_dim//2, opt.dropout)
        elif opt.arch == 'slot': # 1
            self.spm = SetPredictionModule(
                num_embeds=num_embeds, 
                d_in=self.txt_attention_input_dim, 
                d_out=embed_size, 
                axis=1, 
                pos_enc=opt.spm_txt_pos_enc_type, 
                query_dim=opt.spm_query_dim,
                args=opt
            )
        else:
            raise NotImplementedError("Invalid attention head for text modality.")
        
        self.dropout = nn.Dropout(opt.dropout if not opt.rnn_no_dropout else 0) # 0.1

        self.init_weights(wemb_type, word2idx, word_dim)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if 'none' == wemb_type: # 为none则表示不使用预训练的词嵌入
            # nn.init.xavier_uniform_(self.embed.weight)
            self.embed.weight.data.uniform_(-0.1, 0.1)
            print('No wemb init. with pre-trained weight')
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                import torchtext
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():  # 1 # 导入 torchtext 并初始化 GloVe 词嵌入对象 wemb
                import torchtext
                wemb = torchtext.vocab.GloVe()
                
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim    # wemb.vectors.shape() : [2196017,300]

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-','').replace('.','').replace("'",'')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx)-len(missing_words), len(word2idx), len(missing_words)))
            
    def residual_connection(self, rnn_out, rnn_out_last, lengths):
        if self.txt_pooling == 'rnn':
            ret = rnn_out_last
        elif self.txt_pooling == 'max':
            ret = variable_len_pooling(rnn_out, lengths, reduction='max')
        ret = self.txt_pooling_fc(ret) # I()
        return ret



    def forward(self, x, lengths):
        # Embed word ids to vectors
        wemb_out = self.embed(x) # ((词表)1000,69) -> ((特征值)1000,69,300) # 是直接将特征值提取出来了吗
        wemb_out = self.dropout(wemb_out) # Dropout(0.1)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths.cpu(), batch_first=True) # .cpu()? # 是这lengths要求长度必须是从大到小排列的，所以映射来映射去
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        
        rnn_out, rnn_out_last = self.rnn(packed) # rnn_out_last(2,1000,1024)
        # Reshape *final* output to (batch_size, hidden_size)
        rnn_out_last = rnn_out_last.permute(1, 0, 2).contiguous()
        rnn_out = pad_packed_sequence(rnn_out, batch_first=True)[0]
        if self.gpo_rnn: # 1
            rnn_out_last = (rnn_out_last[:, 0, :] + rnn_out_last[:, 1, :]) / 2 # 这难道就是传说中的首尾相加除以二
            rnn_out = (rnn_out[:, :, :rnn_out.shape[-1] // 2] + rnn_out[:, :, rnn_out.shape[-1] // 2:]) / 2
        else:
            rnn_out_last = rnn_out_last.view(-1, self.embed_size)

        rnn_out, rnn_out_last = self.dropout(rnn_out), self.dropout(rnn_out_last) # (1000,69,1024), (1000,1024)

        #### loc
        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        mask = pad_mask.unsqueeze(2).expand(-1,-1,self.embed_size) # ((5bs,69)*1024)
        cap_emb = rnn_out
        #### 1005

        ####
        cap_emb_detach = cap_emb.detach()
        # softA, cap_pat, count_pat = rank_A_mul_ESA(cap_emb_detach, lengths.int(), mask) ## 0.到最高的2
        softA, cap_pat = rank_A_mul_ESA(cap_emb_detach, lengths.int(), mask) ## 0.到最高的2
        softA = self.linear_softA(softA)
        cap_rank_mul = cap_emb * softA # 这里好像 cap_emb后面是0吧，不是的话前面l2norm加上看看
        ####

        posmask = ~mask
        features_in1 = self.linear_ESA1(cap_rank_mul)
        features_in1 = features_in1.masked_fill(posmask == 0,-10000)
        features_k_softmax1 = nn.Softmax(dim=1)(features_in1-torch.max(features_in1,dim=1)[0].unsqueeze(1)) # 这为什么全是同样的数？这还有用吗
        cap_glo1 = torch.sum(features_k_softmax1 * cap_emb, dim=1) # (5bs,dim) ### 这里不太对得上啊，文本模态从5bd到bs

        features_in2 = self.linear_ESA2(cap_rank_mul)
        features_in2 = features_in2 / self.tau_selective
        features_in2 = features_in2.masked_fill(posmask == 0,-10000)
        features_k_softmax2 = nn.Softmax(dim=1)(features_in2-torch.max(features_in2,dim=1)[0].unsqueeze(1)) 
        cap_glo2 = torch.sum(features_k_softmax2 * cap_emb, dim=1) 

        features_in3 = self.linear_ESA3(cap_rank_mul)
        features_in3 = features_in3 / self.tau_comprehensive
        features_in3 = features_in3.masked_fill(posmask == 0,-10000)
        features_k_softmax3 = nn.Softmax(dim=1)(features_in3-torch.max(features_in3,dim=1)[0].unsqueeze(1)) 
        cap_glo3 = torch.sum(features_k_softmax3 * cap_emb, dim=1) 

        features_in4 = self.linear_ESA4(cap_rank_mul)
        # features_in4 = features_in4 + cap_rank_mul  # 残差
        features_in4 = features_in4.masked_fill(posmask == 0,-10000)
        features_k_softmax4 = nn.Softmax(dim=1)(features_in4-torch.max(features_in4,dim=1)[0].unsqueeze(1)) 
        cap_glo4 = torch.sum(features_k_softmax4 * cap_emb, dim=1) 

        cap_glo_cro = x = torch.stack((cap_glo1,cap_glo2,cap_glo3,cap_glo4), dim=1)
        cap_glo_r = cap_glo_cro
        cap_glo_cro = nn.Softmax(dim=1)(cap_glo_cro) + cap_glo_cro 
        cap_glo_cro = l2norm(cap_glo_cro) # (bs,4,dim)

        cap_glo = torch.sum(cap_glo_cro,dim=1)


        out = cap_glo_cro
        attn = None
        
        out = l2norm(out)
        
        cap_ori = l2norm(cap_emb)

        return out, attn, cap_ori
        

    
class EncoderTextBERT(nn.Module):

    def __init__(self, opt, shared_memory=None, shared_query=None):
        super(EncoderTextBERT, self).__init__()

        wemb_type, word_dim, embed_size, num_embeds = \
            opt.wemb_type, opt.word_dim, opt.embed_size, opt.txt_num_embeds

        self.embed_size = embed_size
        self.use_attention = opt.txt_attention

        self.bert = BertModel.from_pretrained('/root/autodl-tmp/dataset/bert-base-uncased')
        self.linear = nn.Linear(768, self.embed_size)
        # self.use_checkpoint = opt.use_checkpoint

        # Sentence embedding
        self.txt_attention_input = opt.txt_attention_input
        # self.txt_pooling = opt.txt_pooling
        # assert self.txt_pooling in ['cls', 'max']
        
        self.linear_ESA = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )
        self.linear_softA = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size)
        )

        self.txt_attention_head = opt.arch
        self.txt_attention_input_dim = embed_size
        self.residual = opt.spm_residual
        self.sep_bert_fc = opt.sep_bert_fc
        if self.sep_bert_fc:
            self.linear2 = nn.Linear(768, self.embed_size)
        
        if opt.arch == 'pvse':
            self.spm = PIENet(
                num_embeds, 
                self.txt_attention_input_dim, 
                embed_size, 
                word_dim//2, 
                opt.dropout
            )
        elif opt.arch == 'slot':
            self.spm = SetPredictionModule(
                num_embeds=num_embeds, 
                d_in=self.txt_attention_input_dim, 
                d_out=embed_size, 
                axis=1, 
                pos_enc=opt.spm_txt_pos_enc_type, 
                query_dim=opt.spm_query_dim,
                args=opt
            )
        else:
            raise NotImplementedError("Invalid attention head for text modality.")
        
        self.dropout = nn.Dropout(opt.dropout if not opt.rnn_no_dropout else 0)

    
    # def residual_connection(self, bert_out, bert_out_cls, lengths):
    #     if self.txt_pooling == 'cls':
    #         ret = bert_out_cls
    #     elif self.txt_pooling == 'max':
    #         ret = variable_len_pooling(bert_out, lengths, reduction='max')
    #     return ret

    # def forward(self, x, lengths):
    #     bert_attention_mask = (x != 0).float()
    #     pie_attention_mask = (x == 0)
    #     bert_emb = self.bert(x, bert_attention_mask)
    #     bert_emb = bert_emb[0]
    #     cap_len = lengths
        
    #     local_cap_emb = self.linear(bert_emb)
    #     global_cap_emb = self.residual_connection(
    #         local_cap_emb if not self.sep_bert_fc else self.linear2(bert_emb), 
    #         local_cap_emb[:, 0] if not self.sep_bert_fc else self.linear2(bert_emb)[:, 0], 
    #         cap_len
    #     )

    #     if self.txt_attention_head == 'pvse':
    #         out, attn, residual = self.spm(local_feat=local_cap_emb, global_feat=global_cap_emb, pad_mask=pie_attention_mask)
    #     elif self.txt_attention_head == 'slot':
    #         out, attn, residual = self.spm(
    #             local_feat=local_cap_emb,
    #             global_feat=global_cap_emb, 
    #             pad_mask=pie_attention_mask,
    #             lengths=lengths
    #         )
        
    #     out = l2norm(out)
        
    #     return out, attn, residual
    def forward(self, x, lengths):
        bert_attention_mask = (x != 0).float()
        pie_attention_mask = (x == 0)
        bert_emb = self.bert(x, bert_attention_mask)
        bert_emb = bert_emb[0]
        cap_len = lengths
        
        local_cap_emb = self.linear(bert_emb) # (5bs,44,dim)
        # global_cap_emb = self.residual_connection(
        #     local_cap_emb if not self.sep_bert_fc else self.linear2(bert_emb), 
        #     local_cap_emb[:, 0] if not self.sep_bert_fc else self.linear2(bert_emb)[:, 0], 
        #     cap_len
        # )

        #### loc
        pad_mask = get_pad_mask(local_cap_emb.shape[1], lengths, True) # (5bs,44)
        mask = pad_mask.unsqueeze(2).expand(-1,-1,self.embed_size) # ((1000,69)*1024)
        cap_emb = local_cap_emb
        softA,cap_pat = rank_A_mul_ESA(local_cap_emb)
        softA = self.linear_softA(softA)
        cap_rank_mul = local_cap_emb * softA # (5bs,44,dim)

        #### glo
        features_in = self.linear_ESA(cap_rank_mul)
        features_in = features_in.masked_fill(mask == 1,-10000) # 真的是塞进了
        features_k_softmax = nn.Softmax(dim=1)(features_in-torch.max(features_in,dim=1)[0].unsqueeze(1)) # 这为什么全是同样的数？这还有用吗
        cap_glo = torch.sum(features_k_softmax * cap_emb, dim=1) # (5bs,dim)

        if self.txt_attention_head == 'pvse':
            out, attn, residual = self.spm(local_feat=local_cap_emb, global_feat=cap_glo, pad_mask=pie_attention_mask)
        elif self.txt_attention_head == 'slot':
            out, attn, residual = self.spm(
                local_feat=cap_rank_mul,
                global_feat=cap_glo, 
                pad_mask=pie_attention_mask,
                lengths=lengths
            )
        
        out = l2norm(out) # (5bs,4,1024)
        
        return out, attn, residual

    ## 换EMB的BERT看看
    # def forward(self, x, lengths):
    #     bert_attention_mask = (x != 0).float()
    #     pie_attention_mask = (x == 0)
    #     bert_emb = self.bert(x, bert_attention_mask)
    #     bert_emb = bert_emb[0]
    #     cap_len = lengths
        
    #     local_cap_emb = self.linear(bert_emb) # (5bs,44,dim)
    #     # global_cap_emb = self.residual_connection(
    #     #     local_cap_emb if not self.sep_bert_fc else self.linear2(bert_emb), 
    #     #     local_cap_emb[:, 0] if not self.sep_bert_fc else self.linear2(bert_emb)[:, 0], 
    #     #     cap_len
    #     # )

    #     #### loc
    #     pad_mask = get_pad_mask(local_cap_emb.shape[1], lengths, True) # (5bs,44)
    #     mask = pad_mask.unsqueeze(2).expand(-1,-1,self.embed_size) # ((1000,69)*1024)
    #     cap_emb = local_cap_emb
    #     softA,cap_pat = rank_A_mul_ESA(local_cap_emb)
    #     softA = self.linear_softA(softA)
    #     cap_rank_mul = local_cap_emb * softA # (5bs,44,dim)

    #     #### glo
    #     features_in = self.linear_ESA(cap_rank_mul)
    #     features_in = features_in.masked_fill(mask == 1,-10000) # 真的是塞进了
    #     features_k_softmax = nn.Softmax(dim=1)(features_in-torch.max(features_in,dim=1)[0].unsqueeze(1)) # 这为什么全是同样的数？这还有用吗
    #     cap_glo = torch.sum(features_k_softmax * cap_emb, dim=1) # (5bs,dim)

    #     if self.txt_attention_head == 'pvse':
    #         out, attn, residual = self.spm(local_feat=local_cap_emb, global_feat=cap_glo, pad_mask=pie_attention_mask)
    #     elif self.txt_attention_head == 'slot':
    #         out, attn, residual = self.spm(
    #             local_feat=cap_rank_mul,
    #             global_feat=cap_glo, 
    #             pad_mask=pie_attention_mask,
    #             lengths=lengths
    #         )
        
    #     out = l2norm(out) # (5bs,4,1024)
        
    #     return out, attn, residual
