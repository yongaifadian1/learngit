from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from scipy import io
import pandas as pd
import torch
class Iemocap_dataset(Dataset):
    def __init__(self,iemocap_path,do_uniform=True,state=None,fold_round=None) -> None:
        #fold_round为轮数，进行n折时需要填，第1轮将session05作为测试集，以此类推
        super().__init__()
        self.do_uniform=do_uniform
        self.iemocap_dataset_path=os.path.join(iemocap_path,'wavlm_large_L12_mat')
        csv_path=os.path.join(iemocap_path,'name_label_text.csv')
        self.dataframe=pd.read_csv(csv_path,header=0)#header=0表示第一行为列名
        self.dataframe.drop(index=self.dataframe[self.dataframe['label'].isin(['xxx','fru','sur','fea','dis','oth'])].index,inplace=True)#删除无意义标签的行
        self.name_label_series=self.dataframe.set_index('name')['label']#name为索引，label为值的series
        self.data_name_list=self.dataframe.iloc[:,1].tolist()#筛除过的文件名列表
        if state is not None:
            temp_data_name_list=self.data_name_list
            self.data_name_list=[]
        
        #对文件名列表做5折处理
        if state=='test':
            test_name_include=f'Ses0{6-fold_round}'#测试的数据名包含的字符串
            for i,data_name in enumerate(temp_data_name_list):
                if test_name_include in data_name:
                    self.data_name_list.append(data_name)
        elif state=='train':
            test_name_include=f'Ses0{6-fold_round}'
            for i,data_name in enumerate(temp_data_name_list):
                if test_name_include not in data_name:
                    self.data_name_list.append(data_name)
        elif state is not None:
            raise KeyError('输入合法的数据集状态')

        self.dict_label_2_index={'ang':0,'neu':1,'hap':2,'exc':2,'sad':3}
    def __len__(self):
        return len(self.data_name_list)
    def __getitem__(self, index):
        data_name=self.data_name_list[index]#文件名（从已经筛除过的,做过n折的，文件名列表中索引的）
        data_path=os.path.join(self.iemocap_dataset_path,data_name)
        data_out=np.float32(io.loadmat(data_path)['wavlm'])#加载索引的mat文件成二维numpy数组
        
        #统一dataout时间步为400,并根据data_out的时间步生成mask
        mask=None
        if self.do_uniform==True:
            if data_out.shape[0]>400:
                mask=torch.zeros(400,dtype=torch.bool)#生成一维长度400的bool型张量，全false

                data_out=data_out[:400,:]       
            else:
                mask=torch.zeros(data_out.shape[0],dtype=torch.bool)
                mask_expand=torch.ones(400-data_out.shape[0],dtype=torch.bool)
                mask=torch.cat((mask,mask_expand))#生成一维长度400的bool张量，空白时间步为true其余false

                data_out=np.pad(data_out,((0,400-data_out.shape[0]),(0,0)),mode='constant') 

        label_name=self.name_label_series[data_name]#索引的情感标签名
        label_out=self.dict_label_2_index[label_name]
        return torch.tensor(data_out),torch.tensor(label_out),mask
if __name__=='__main__':
    # iemocap_path=r'/148Dataset/data-mai.jialong/iemocap_extract'
    # dataset=Iemocap_dataset(iemocap_path,state='train',fold_round=1)
    # print(len(dataset))
    
    #查看mat文件
    data_path='/148Dataset/data-mai.jialong/iemocap_extract/wavlm_large_L12_mat/Ses01F_script01_2_M017'
    data_out=io.loadmat(data_path)[ '__header__']
    print(data_out)

