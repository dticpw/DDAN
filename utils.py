import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import matplotlib.pyplot as plt
import pickle



def print_options(args):
    print("")
    print("----- options -----".center(120, '-'))
    args = vars(args)
    string = ''
    for i, (k, v) in enumerate(sorted(args.items())):
        string += "{}: {}".format(k, v).center(40, ' ')
        if i % 3 == 2 or i == len(args.items()) - 1:
            print(string)
            string = ''
    print("".center(120, '-'))
    print("")

# def rl(mask2d,tn):
#     # 将布尔掩码转换为0和1的数组
#     heatmap = mask2d.float().numpy()
    
#     # 使用matplotlib显示热力图
#     plt.imshow(heatmap, cmap='cool')
#     plt.colorbar(label='Intensity')
#     plt.title(tn)
#     plt.show()

def visualize_and_save(tensor, img_folder, tensor_file=None):

    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    existing_files = os.listdir(img_folder) # 存储文件夹名
    # 遍历existing_files列表中以"tensor_"开头的文件名，切掉'_'前的和'.'后的，转成int
    existing_indices = [int(filename.split('_')[1].split('.')[0]) for filename in existing_files if filename.startswith('tensor_')]
    if existing_indices:
        index = max(existing_indices) + 1
    else:
        index = 0

    # 如果张量在GPU上，将其移到CPU
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        # 将张量转换为ndarray类型，确保可以与matplotlib兼容
        tensor = tensor.detach().numpy()  # 转换为numpy.ndarray

    plt.imshow(tensor, cmap='viridis')  # You can choose different colormaps as per your preference
    plt.colorbar()  # Add color bar for reference
    # plt.xticks(range(0, tensor.shape[1], 4))  # 以4为单位设置x轴刻度
    # plt.yticks(range(0, tensor.shape[0], 4))  # 以4为单位设置y轴刻度
    img_file = os.path.join(img_folder, f'tensor_{index}.png')
    plt.savefig(img_file, dpi=300)
    plt.close()
    print("image_loaded(visualize_and_save)")

    ## 先给下了，占的空间实在太大了
    # with open(tensor_file, 'ab') as f:
    #     pickle.dump(tensor, f)



    # #### 可视化一波
    # img_emb2d_5times = img_embs.reshape(-1,img_embs.shape[-1])
    # img_emb2d = img_emb2d_5times[::5]
    # txt_emb2d = txt_embs.reshape(-1,txt_embs.shape[-1])
    # cos_mat = cosine_sim(img_emb2d, txt_emb2d)
    # # step = 32
    # # ministep = 8
    # # for i in range(img_emb2d.shape[0]//step):
    # #     sim_pic = torch.tensor([]).cuda()
    # #     for j in range(step//ministep):
    # #         cos_mat_temp = cos_mat[i*step+j*ministep:i*step+(j+1)*ministep, i*(step*5)+j*ministep:i*(step*5)+(j+1)*ministep]
    # #         sim_pic = torch.cat((sim_pic, cos_mat_temp), dim=0)
    # #     visualize_and_save(cos_mat_temp, args.img_folder)
    # # print("image loaded!")

    # step = 8
    # count = 0 
    # sim_pic = torch.tensor([]).cuda()
    # for i in range(img_emb2d.shape[0]//step):
    #     count += 1
    #     cos_mat_temp = cos_mat[i*step:(i+1)*step, i*(step*5):(i+1)*(step*5)]
    #     sim_pic = torch.cat((sim_pic, cos_mat_temp), dim=0)
    #     if count % 4 == 0:
    #         visualize_and_save(sim_pic, args.img_folder)
    #         sim_pic = torch.tensor([]).cuda()
    # ####


