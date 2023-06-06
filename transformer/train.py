

import torch
from torch import nn
from prac_encoder import Net_4_layer
from data import Iemocap_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score
from tqdm import tqdm
from config import Cfg,modify_config
from utils import environment
import argparse
iemocap_path='/148Dataset/data-mai.jialong/iemocap_extract'
device=torch.device('cuda')
# os.environ['CUDA_VISIBLE_DEVICES']='2'
def create_PositionalEncoding(input_dim, max_seq_len=400): #为（T，C）张量创建位置编码
    position_encoding = np.array([ 
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)] 
        for pos in range(max_seq_len)]) 
    
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False) 
    
    return position_encoding
def initialize(fold_round):
    encoder=Net_4_layer(1024,8).to(device)
    opt=torch.optim.SGD(encoder.parameters(),lr=0.0005)
    loss_func=torch.nn.CrossEntropyLoss()
    epoch=1 
    writer=SummaryWriter(log_dir=f'logs/fold{fold_round}')
    return encoder,opt,loss_func,epoch,writer
def load_weight(opt,encoder):
    if os.path.exists('weight/weight.pt'):
        weight=torch.load('weight/weight.pt')
        opt.load_state_dict(weight['opt'])
        encoder.load_state_dict(weight['model'])
        # epoch=weight['epoch']
        print('loading')
    else:
        print('nothing to load')
def train_test(encoder,fold_round,loss_func,opt,epoch,writer,position_code=None,sum_epoch=120):
    if position_code is not None:
        position_code=position_code.to(device)
    train_dataset=Iemocap_dataset(iemocap_path,state='train',fold_round=fold_round)
    test_dataset=Iemocap_dataset(iemocap_path,state='test',fold_round=fold_round)
    train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=True)

    scheduler=CosineAnnealingLR(optimizer=opt,T_max=sum_epoch,eta_min=0.0005/100)
    wa_list=[]
    ua_list=[]
    f1_list=[]
    while epoch<sum_epoch+1:
        #进度条前缀
        train_pbar=tqdm(train_dataloader)
        train_pbar.set_description(f'第{fold_round}折-->训练轮次：{epoch}/{sum_epoch}')

        loss_sum=0

        for i,[data,label,mask] in enumerate(train_pbar):
            data,label,mask=data.to(device),label.to(device),mask.to(device)
            output=encoder(data,mask=mask,position_code=position_code)
            loss=loss_func(output,label)
            loss_sum=loss_sum+float(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            #进度条后缀
            train_pbar_dict={}
            train_pbar_dict['lr']=opt.param_groups[0]['lr']
            train_pbar_dict['loss']=float(loss)
            train_pbar.set_postfix(train_pbar_dict)

        #每折最后一轮存一下
        # if epoch==119:
        #     weight={'epoch':epoch,'model':encoder.state_dict(),'opt':opt.state_dict()}
        #     torch.save(weight,f'weight/weight{fold_round}.pt')    

        loss_ave=loss_sum/len(train_pbar)
        writer.add_scalar(
            tag=f'flod{fold_round}-train loss',
            scalar_value=loss_ave,
            global_step=epoch
        )

        wa,ua,f1=test(test_dataloader,encoder,loss_func,epoch,sum_epoch,writer,fold_round,position_code=position_code)#每个epoch都test一次
        
        epoch+=1
        scheduler.step()
        wa_list.append(wa)
        ua_list.append(ua)
        f1_list.append(f1)
    wa_plus_ua=[wa+ua for wa,ua in zip(wa_list,ua_list)]
    wa_plus_ua=[float(i) for i in wa_plus_ua]#将列表元素数据类型由np.float转成float
    max_wa_plus_ua=max(wa_plus_ua)
    best_index=wa_plus_ua.index(max_wa_plus_ua)#因为index不接收np.float
    best_wa=wa_list[best_index]
    best_ua=ua_list[best_index]
    best_f1=f1_list[best_index]
    return best_wa,best_ua,best_f1
def test(test_dataloader,encoder,loss_func,epoch,sum_epoch,writer,fold_round,position_code=None):
    if position_code is not None:
        position_code=position_code.to(device)
    with torch.no_grad():
        loss_sum=0
        test_pbar=tqdm(test_dataloader)
        test_pbar.set_description(f'第{fold_round}折-->测试轮次：{epoch}/{sum_epoch}')
        for i,[data,label,mask] in enumerate(test_pbar):
            data,label,mask=data.to(device),label.to(device),mask.to(device)
            output=encoder(data,mask=mask,position_code=position_code)
            output_label=torch.argmax(input=output,dim=1)#返回一个一维张量
            if i==0:
                output_label_tensor=output_label
                label_tensor=label
            else:
                output_label_tensor=torch.cat((output_label_tensor,output_label))
                label_tensor=torch.cat((label_tensor,label))
            loss=loss_func(output,label)
            
            loss_sum=loss+loss_sum
            wa_temp=accuracy_score(label.cpu(),output_label.cpu())
            ua_temp=recall_score(label.cpu(),output_label.cpu(),average='macro',zero_division=0)
            test_pbar_dict={}
            test_pbar_dict['score']=wa_temp+ua_temp
            test_pbar_dict['loss']=float(loss)
            test_pbar.set_postfix(test_pbar_dict)
        
        wa=accuracy_score(label_tensor.cpu(),output_label_tensor.cpu())#accuracy接收np数组和列表，tensor只有在cpu中才能转成np数组，返回np类型浮点数
        ua=recall_score(label_tensor.cpu(),output_label_tensor.cpu(),average='macro',zero_division=0)
        f1=f1_score(label_tensor.cpu(),output_label_tensor.cpu(),average='weighted',zero_division=0)
        loss_ave=loss_sum/len(test_pbar)
        writer.add_scalar(
            tag=f'flod{fold_round}-test loss',
            scalar_value=loss_ave,
            global_step=epoch
        )
        writer.add_scalar(
            tag=f'flod{fold_round}-test wa',
            scalar_value=wa,
            global_step=epoch
        )
        writer.add_scalar(
            tag=f'flod{fold_round}-test ua',
            scalar_value=ua,
            global_step=epoch
        )
        writer.add_scalar(
            tag=f'flod{fold_round}-test f1',
            scalar_value=f1,
            global_step=epoch
        )

    return float(wa),float(ua),float(f1)#将np类型浮点数转成浮点数

def main(cfg):
    # position_code=create_PositionalEncoding(1024,max_seq_len=400)
    environment.visible_gpus(cfg.train.device_id)

    fold_best_data={
        'fold_round_1':{
            'wa_best':None,
            'ua_best':None,
            'f1_best':None
        },
        'fold_round_2':{
            'wa_best':None,
            'ua_best':None,
            'f1_best':None
        },
        'fold_round_3':{
            'wa_best':None,
            'ua_best':None,
            'f1_best':None
        },
        'fold_round_4':{
            'wa_best':None,
            'ua_best':None,
            'f1_best':None
        },
        'fold_round_5':{
            'wa_best':None,
            'ua_best':None,
            'f1_best':None
        }
    }
    for fold_round in range(1,6):
        encoder,opt,loss_func,epoch,writer=initialize(fold_round)
        best_wa,best_ua,best_f1=train_test(encoder,fold_round,loss_func,opt,epoch,writer,sum_epoch=120)
        fold_best_data[f'fold_round_{fold_round}']['wa_best']=best_wa
        fold_best_data[f'fold_round_{fold_round}']['ua_best']=best_ua
        fold_best_data[f'fold_round_{fold_round}']['f1_best']=best_f1
    wa_average=sum([fold_best_data['fold_round_1']['wa_best'],fold_best_data['fold_round_2']['wa_best'],fold_best_data['fold_round_3']['wa_best'],fold_best_data['fold_round_4']['wa_best'],fold_best_data['fold_round_5']['wa_best']])/5
    ua_average=sum([fold_best_data['fold_round_1']['ua_best'],fold_best_data['fold_round_2']['ua_best'],fold_best_data['fold_round_3']['ua_best'],fold_best_data['fold_round_4']['ua_best'],fold_best_data['fold_round_5']['ua_best']])/5
    f1_average=sum([fold_best_data['fold_round_1']['f1_best'],fold_best_data['fold_round_2']['f1_best'],fold_best_data['fold_round_3']['f1_best'],fold_best_data['fold_round_4']['f1_best'],fold_best_data['fold_round_5']['f1_best']])/5
    # print('each fold best data is:',fold_best_data)
    print('the average WA is:',wa_average)
    print('the average UA is:',ua_average)
    print('the average F1 is:',f1_average)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--train.device_id", help="modify cfg.train.device_id", type=str)
    args = parser.parse_args()
    modify_config(Cfg, args)
    main(Cfg)


# tqdm进度条 ✔
# 指标 WA UA F1 sklearn.metrics ✔
# 学习率下降的策略：scheduler ✔
# 测试 test 五折flod  ✔
# getitem多输出个mask  ✔
# 分类器T维度平均掉真实长度 ✔
# tensorboard记录 ✔
# https://github.com/microsoft/unilm/tree/master/wavlm 提特征再跑一次