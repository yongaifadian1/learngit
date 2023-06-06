from torch import nn
import torch
from torch.nn import functional as F
class Multihead_self_attention(nn.Module):
    def __init__(self,embed_dim,head_num):
        super(Multihead_self_attention,self).__init__()
        self.project_QKV=nn.Linear(embed_dim,embed_dim*3)
        self.project_output=nn.Linear(embed_dim,embed_dim)
        self.scaling_factor=float(embed_dim/head_num)**-0.5
        self.head_num=head_num
    def forward(self,query,mask=None):#query:(batch_size,target_length,embed_dim)    mask:(batch_size,target_length)
        batch_size,target_length,embed_dim=query.size()
        query,key,value=self.project_QKV(query).chunk(3,dim=-1)
        query=query.transpose(0,1).contiguous().view(target_length,batch_size*self.head_num,int(embed_dim/self.head_num)).transpose(0,1)
        key=key.transpose(0,1).contiguous().view(target_length,batch_size*self.head_num,int(embed_dim/self.head_num)).transpose(0,1)
        value=value.transpose(0,1).contiguous().view(target_length,batch_size*self.head_num,int(embed_dim/self.head_num)).transpose(0,1)
        weight=torch.bmm(query,key.transpose(1,2))*self.scaling_factor#防止weight过大导致softmax梯度接近0
        #计算相似度
        if mask is not None:weight=weight.view(batch_size,self.head_num,target_length,target_length).masked_fill(mask.unsqueeze(1).unsqueeze(2),float('-inf')).view(batch_size*self.head_num,target_length,target_length)
        weight=F.softmax(weight, dim=-1)
        output=torch.bmm(weight,value)
        output=output.transpose(0,1).contiguous().view(target_length,batch_size,embed_dim).transpose(0,1)
        return self.project_output(output)
if __name__=='__main__':
    multi=Multihead_self_attention(1024,16)
    input=torch.randn(32,400,1024)
    print(multi(input).shape)
