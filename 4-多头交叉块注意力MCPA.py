# 论文题目：Multimodal Fusion Transformer for Remote Sensing Image Classification
# https://github.com/AnkurDeria/MFT?tab=readme-ov-file
# https://arxiv.org/pdf/2203.16952

import torch.nn as nn
import torch
from einops import rearrange

'''
题目：面向遥感图像分类的多模态融合变换器    IEEE 2023
多头交叉块注意力模块：MCPA
本文介绍了一种用于遥感图像（RS）数据融合的新型多头交叉补丁注意力（mCrossPA）机制。
类tokens还包含补充信息，源自多模态数据（例如 LiDAR、MSI、SAR 和 DSM），
这些数据与 HSI 补丁tokens信息一起馈送到vit网络。 新开发的 mCrossPA 
广泛使用了的注意力机制，可以有效地将来自 HSI 补丁tokens和现有 CLS tokens的信息
融合到一个集成了多模态特性的新tokens中。

在各种cv和nlp任务上是通用即插即用模块
'''

class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads # num_heads=8
        head_dim = dim // num_heads # dim=32    head_dim=4
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim , bias=qkv_bias) # nn.Linear(head_dim=4, dim=32, bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim) # nn.Linear(dim * num_heads 32*8=256, dim=32)
        self.proj_drop = nn.Dropout(proj_drop)  #proj_drop=0.1

    def forward(self, x):
        B, N, C = x.shape # B=1 N=1024 C=32
       
        # x[:, 0:N, ...].reshape(B=1, N=1024, self.num_heads=8, C // self.num_heads=4) = [1,1024,8,4]
        # self.wq = nn.Linear(head_dim=4, dim=32, bias=qkv_bias) 线性变换 从4→32
        # q.shape = [1, 8, 1024, 32];       self.num_heads 8； C // self.num_heads 4 
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:N, ...].reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  
        
        # out：k.shape = [1, 8, 1024, 32] BNC -> BNH(C/H) -> BHN(C/H)   
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  
        
        # v.shape = [1, 8, 1024, 32]
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        
        # in : q.shape = [1 B, 8 H, 1024 N , 32 C] k.shape = [1, 8, 1024, 32]
        # deal : self.scale=0.5
        # out: attn.shape [1 B, 8 H, 1024 N, 1024 N] 
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # attn.shape [1 B, 8 H, 1024 N, 1024 N]
        attn = attn.softmax(dim=-1)
        
        # in attn [1 B, 8 H, 1024 N, 1024 N] v [1, 8, 1024, 32]
        # deal torch.einsum
        # out x [1B, 1024N, 8H, 32] 
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)    # x.shape = [1, 1024, 8, 32] 
        
        #  C = 32 self.num_heads=8
        x = x.reshape(B, N, C * self.num_heads)   # x.shape = torch.Size([1, 1024, 256]) (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        
        # self.proj Linear(in_features=256, out_features=32, bias=True)
        x = self.proj(x) # x.shape = [1, 1024, 32]

        # self.proj_drop = nn.Dropout(proj_drop)  proj_drop=0.1
        x = self.proj_drop(x)   # x.shape = [1, 1024, 32]
       
        return x # x.shape = [1, 1024, 32]

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

if __name__ == '__main__':
    model = MCrossAttention(dim=32)
#   1.如果输入的是图片四维数据需转化为三维数据，演示代码如下
#     H, W = 32, 32
#     input = torch.randn(1, 32, H, W)
#     input_3d = to_3d(input)
#     output_3d = model(input_3d)
#     output_4d = to_4d(output_3d, H, W)
#     print('input_size:', input.size())
#     print('output_size:', output_4d.size())

#   2.如果输入的是三维数据演示代码如下
    input = torch.randn(1,1024,32) #B ,L,N
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())

