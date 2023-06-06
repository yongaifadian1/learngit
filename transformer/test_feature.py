#测试自己提的特征
from prac_encoder import Net_4_layer
import torch
import os

mat_path='/148_Dataset_in188/data-mai.jialong/iemocap_extract/wavlm_extract_prac/wavlm_extract_prac.mat'
net=Net_4_layer(1024,8)
device=torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES']='2'
