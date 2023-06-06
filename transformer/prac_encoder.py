import torch
from torch import nn
from prac_mutihead_selfattention import Multihead_self_attention
class Net_4_layer(nn.Module):
    def __init__(self,embed_dim,head_num) -> None:
        super().__init__()
        self.encoder_layers=nn.ModuleList([
            Encoder(embed_dim,head_num),
            Encoder(embed_dim,head_num),
            Encoder(embed_dim,head_num),
            Encoder(embed_dim,head_num)]
        )
        
        self.classify_layer=nn.Linear(embed_dim,4)
    def forward(self,input_embed,position_code=None,mask=None):
        encoder_output = Encoder.add_position(self.encoder_layers,input_embed,position_code=position_code,mask=mask)
        #经过四层encoder
        # for m in self.encoder_layers:
        #     encoder_output=m(encoder_output,position_code=position_code,mask=mask)

        encoder_output=input_embed#不经过encoder，直接FC分类器

        ffn_output=torch.sum(encoder_output,dim=1)#(B,C)
        ffn_output=ffn_output/torch.sum(torch.eq(mask,0), dim=-1, keepdim=True)#sum指定的维度原本会被消掉，指定了keepdim后就可以保持

        ffn_output=self.classify_layer(ffn_output)
        return ffn_output #形状(B,4)
        
class Encoder(nn.Module):
    def __init__(self,embed_dim,head_num) -> None:
        super(Encoder,self).__init__()
        self.multihead=Multihead_self_attention(embed_dim,head_num)
        self.norm1=nn.LayerNorm(embed_dim)
        self.ffn1=nn.Linear(embed_dim,int(embed_dim/2))
        self.acti=nn.ReLU()
        self.ffn2=nn.Linear(int(embed_dim/2),embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)

        #分类器
        # self.classify=nn.Linear(embed_dim,4)
    def add_position(self,input_embed,position_code=None,mask=None):#input_embed:B,T,C   mask:B,T    position_code:T,C
        if position_code==None:
            return input_embed
        batch_size,target_lenth,channel=input_embed.size()
        position_code=position_code[:target_lenth].unsqueeze(0).repeat(batch_size,1,1)
        #repeat()表示在第一维重复b次，第二三维重复1次，也就是不改变。
        if mask is not None:
            mask=1-mask.unsqueeze(-1).repeat(1,1,channel).type(torch.float)
            position_code=mask*position_code
        return input_embed+position_code
    def forward(self,input_embed,position_code=None,mask=None):
        if position_code is not None:
            input_embed=self.add_position(input_embed,position_code,mask)
        residual_x1=input_embed
        multi_output=nn.functional.dropout(self.multihead(input_embed,mask),0.1)
        multi_output=self.norm1(residual_x1+multi_output)
        residual_x2=multi_output
        ffn_output=nn.functional.dropout(self.ffn2(self.acti(nn.functional.dropout(self.ffn1(multi_output),0.1))),0.1)
        ffn_output=self.norm2(residual_x2+ffn_output)

        #分类器
        # ffn_output = torch.sum(ffn_output, dim=1) / 400 
        # ffn_output=self.classify(ffn_output)

        return ffn_output
if __name__=='__main__':
    input=torch.rand(5,10,512)
    mask=torch.randn(5,10).ge(0)# rand:0~1均匀分布   randn：正态分布   ge：greater or equal 
    net=Net_4_layer(512,8)
    position=torch.rand(10,512)
    print(net(input,position_code=position,mask=mask).shape)

