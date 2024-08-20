import math
import torch
from torch import nn
# https://github.com/Lose-Code/UBRFC-Net
# https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
'''
 用于图像去雾的无监督双向对比重建和自适应细粒度信道注意力网络     SCI 一区 2024 顶刊
捕捉全局和局部信息交互即插即用注意力模块：FCAttention

无监督算法在图像去雾领域取得了显著成果。此外，SE通道注意力机制仅利用全连接层捕捉全局信息，
缺乏与局部信息的互动，导致图像去雾时特征权重分配不准确。

为克服这些挑战，我们开发了一种自适应细粒度通道注意力（FCA）机制，
利用相关矩阵在不同粒度级别捕获全局和局部信息之间的关联，促进了它们之间的互动，实现了更有效的特征权重分配。

在图像去雾方面超越了当前先进的方法。本研究成功地引入了一种增强型无监督图像去雾方法，有效解决了现有技术的局限，实现了更优的去雾效果。
适用于：图像增强，暗光增强，图像去雾，图像去噪等所有CV2维任务通用的即插即用注意力模块
'''

# init也很重要

# 定义一个 Mix 模块，它接受一个参数 m，用于初始化权重 w。
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()

        # 创建一个可学习的参数 w，初始值为 m。
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w

        # 定义一个 Sigmoid 激活函数模块。
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        '''
        in(type shape): fea1  [1, 64, 1, 1]  fea2 [1, 64, 1, 1]
        deal(function meaning): self.mix_block  加权和
        out(type shape): out
          
        '''

        # 计算 w 的 Sigmoid 激活值作为混合因子。
        mix_factor = self.mix_block(self.w)
        '''
        in(type shape):  self.w = [-0.8000], requires_grad=True
        deal(function meaning): 
        out(type shape): mix_factor = [0.3100], grad_fn=<SigmoidBackward0>
          
        '''

        # 将混合因子应用于输入特征 fea1 和 fea2，并进行加权求和。
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        '''
        in(type shape): 
        deal(function meaning):
        out(type shape): 
          
        '''


        '''
        in(type shape): expand_as

        deal(function meaning):
            用于将一个张量扩展到与另一个张量相同的形状。
            将一个较小的张量沿着指定的维度进行扩展，以匹配目标张量的形状。

            mix_factor.expand_as(fea1) 和 mix_factor.expand_as(fea2) 的作用
            将 mix_factor 扩展到与 fea1 和 fea2 相同的形状
            确保在进行加权求和时，mix_factor 的形状与 fea1 和 fea2 匹配

        out(type shape): 
            假设 mix_factor 的值为 [0.3100]，fea1 和 fea2 的形状为 [1, 64, 1, 1]，
            那么 mix_factor.expand_as(fea1) 的结果将是一个形状为 [1, 64, 1, 1] 的张量，
            所有元素的值都是 0.3100。

            
            通过 expand_as 方法，mix_factor 被扩展到与 fea1 和 fea2 相同的形状，
            从而可以进行逐元素的加权求和操作。

        '''
        return out

class FCAttention(nn.Module):

    # 定义一个全连接注意力（FCAttention）模块，接受通道数 channel 和其他参数
    def __init__(self,channel,b=1, gamma=2): # channel = 64
        super(FCAttention, self).__init__()

        # 应用全局平均池化，输出尺寸为 1。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma)) # t=3
        k = t if t % 2 else t + 1 # k=3

        # 计算一维卷积的 kernel_size，基于输入通道数。
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        # 定义一个卷积层，用于在特征图上应用全连接操作。
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)

        # 定义 Sigmoid 激活函数模块。
        self.sigmoid = nn.Sigmoid()

        # 实例化一个 Mix 模块。
        self.mix = Mix()


    def forward(self, input):
        '''
        in(type shape): 
            input= [1,64,256,256] 
        
        deal(function meaning):论文结构图

        out(type shape): 
            return input*out
            out.shape [1, 64, 1, 1] input.shape [1, 64, 256, 256]
            input*out (1,64,256,256)
            
        '''


        x = self.avg_pool(input) 
        '''
        in(type shape): input= [1,64,256,256] 
        deal(function meaning):AdaptiveAvgPool2d(output_size=1)
        out(type shape): x [1, 64, 1, 1]
          
        '''

        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        '''
        in(type shape): x [1, 64, 1, 1]
        deal(function meaning):
            x.squeeze(-1).shape [1, 64, 1]
            transpose(-1, -2) [1, 1, 64]
            【不变卷积】self.conv1 Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
        out(type shape):x1  (1,64,1)
          
        '''

        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)

        '''
        in(type shape): x [1, 64, 1, 1]
        deal(function meaning): 
            
            self.fc = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            self.fc(x).shape [1, 64, 1, 1]
            squeeze(-1) [1, 64, 1]
            transpose(-1, -2) [1, 1, 64]
        out(type shape): 
            x2.shape = [1, 1, 64]
        
        '''
        
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)
        '''
        in(type shape): 
            x1  (1,64,1) \\ x2.shape = [1, 1, 64]
        deal(function meaning):
            计算 x1 和 x2 的外积，并通过 sum 求和，得到 out1.shape [1, 64, 1, 1] 的注意力图。
            torch.matmul(x1,x2) [1, 64, 64]
            torch.sum(torch.matmul(x1,x2),dim=1).shape [1, 64]
            unsqueeze(-1)  [1, 64,1] 
            unsqueeze(-1) [1,64,1,1]
        out(type shape): out1 (1,64,1,1)
          
        '''
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        
        out1 = self.sigmoid(out1)
        '''
        in(type shape): out1 (1,64,1,1)
        deal(function meaning):self.sigmoid
        out(type shape): out1 (1,64,1,1)
          
        '''


        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        '''
        in(type shape): x1.shape [1, 64, 1] x2.shape = [1, 1, 64]
        deal(function meaning):
            x2.transpose(-1, -2) [1, 64, 1] x1.transpose(-1, -2) [1, 1, 64]
            matmul [1,64,64]
            sum ,dim=1 [1,64] 沿哪个维度相加 哪个维度消失? 我觉得应该是变成1
            .unsqueeze(-1).unsqueeze(-1)  [1, 64, 1, 1]
        out(type shape): 
            out2.shape  [1, 64, 1, 1]
          
        '''

        out2 = self.sigmoid(out2)

        '''
        in(type shape): out2.shape [1, 64, 1, 1]
        deal(function meaning):self.sigmoid
        out(type shape):  out2.shape [1, 64, 1, 1]
          
        '''

        # 应用 Mix 模块，结合 out1 和 out2。 out.shape  [1, 64, 1, 1]
        out = self.mix(out1,out2)

        '''
        in(type shape): 
            out1 (1,64,1,1)  out2 [1, 64, 1, 1]
        deal(function meaning):
            nn.Sigmoid()
        out(type shape): 
            out.shape  [1, 64, 1, 1]
          
        '''

        # 再次应用一维卷积，并调整维度，得到 [1, 64, 1] 的特征图。
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        '''
        in(type shape): 
            out [1, 64, 1, 1]
        deal(function meaning):
            out.squeeze(-1) [1, 64, 1] transpose(-1, -2)  [1, 1, 64]
            【ksp 311 不变卷积】self.conv1  Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
            self.conv1([1, 1, 64])  → [1, 1, 64]
        out(type shape): out.shape  [1, 64, 1, 1]
          
        '''        
        # 应用 Sigmoid 激活函数，得到最终的输出。
        out = self.sigmoid(out)

        '''
        in(type shape): out.shape  [1, 64, 1, 1]
        deal(function meaning):self.sigmoid
        out(type shape): out.shape  [1, 64, 1, 1]
          
        '''        
        
        '''
        in(type shape): input.shape [1, 64, 256, 256] out.shape [1, 64, 1, 1] 
        deal(function meaning) : FCAttention
        out(type shape): input * out    [1, 64, 256, 256]
          
        '''
        
        # 将输入特征图与计算得到的权重相乘，得到增强后的输出特征图。
        return input * out   

# 适用于：图像增强，暗光增强，图像去雾，图像去噪
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,64,256,256)
    model = FCAttention(channel=64)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
