import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/Zheng-MJ/SMFANet
'''
SMFANet：一种轻量级的自调制特征聚合网络，以实现高效的图像超分辨率   ECCV 2024 顶会

即插即用特征融合模块：SMFA
探索非局部信息，从而更好地进行高分辨率的图像重建。
然而，SA机制需要大量的计算资源，这限制了其在低功耗设备中的应用。
此外，SA机制限制了其捕获局部细节的能力，从而影响图像重建效果。
为了解决这些问题，我们提出了一个自研特征融合（SMFA）模块，
以协同利用局部和非局部特征交互来进行更准确的高分辨率的图像重建。

具体来说，SMFA模块采用了一种有效的自注意近似（EASA）分支来捕获非局部信息，
并使用一个局部细节估计（LDE）分支来捕获局部细节。

适用于：高分辨率图像重建，暗光增强，图像恢复，等所有CV任务上通用特征融合模块
'''
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


# class PCFN(nn.Module):
#     def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
#         super().__init__()
#         hidden_dim = int(dim * growth_rate)
#         p_dim = int(hidden_dim * p_rate)
#         self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
#         self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

#         self.act = nn.GELU()
#         self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

#         self.p_dim = p_dim
#         self.hidden_dim = hidden_dim

#     def forward(self, x):
#         if self.training:
#             x = self.act(self.conv_0(x))
#             x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
#             x1 = self.act(self.conv_1(x1))
#             x = self.conv_2(torch.cat([x1, x2], dim=1))
#         else:
#             x = self.act(self.conv_0(x))
#             x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
#             x = self.conv_2(x)
#         return x
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape  
        # f.shape=（1,32,256,256）

        y, x = self.linear_0(f).chunk(2, dim=1) 
        # self.linear_0 Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)) 
        #  self.linear_0(f).shape = [1, 64, 256, 256]
        # y.shape = x.shape = [1, 32, 256, 256]

        x_s = self.dw_conv(
            F.adaptive_max_pool2d(
                x, (h // self.down_scale, w // self.down_scale)))
        '''
        in(type shape): 
            x[1,32,256,256];
            h // self.down_scale = 256//8 (32) = w // self.down_scale

        deal(function meaning):
            F.adaptive_max_pool2d 将输入张量的高度和宽度从 256 调整为 32，而批次大小和通道数保持不变。
                out [1,32,32,32]   缩小 self.down_scale 8倍
            self.dw_conv = nn.Conv2d(32, 32, 3 k, 1 s, 1 p, groups= 32)  深度可分离卷积

        out(type shape):  
            x_s [1,32，32,32]
          
        '''


        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        '''
        in(type shape): 
            x[1,32,256,256];
        deal(function meaning):
            torch.var 计算张量方差
            dim=(-2, -1) 倒数第二个和最后一个维度
            keepdim=True: 保持被计算的维度，结果张量的形状与输入张量相同

            x 的形状为 (batch_size, channels, height, width)，
            计算每个通道在 height 和 width 维度上的方差
        out(type shape): 
            x_v (batch_size = 1, channels =32, 1, 1)
          
        '''

        x_l = x * F.interpolate(
            self.gelu(
                self.linear_1(
                    x_s * self.alpha + x_v * self.belt)
                    ), 
                    size=(h, w),
                    mode='nearest')
        '''
        in(type shape): 
            x_s * self.alpha [1,32，32,32] *(1, dim=32, 1, 1) ==> [1, 32, 32, 32]
            x_v * self.belt  (1, 32,1, 1) *(1, dim=32, 1, 1) ==> [1, 32, 1, 1]
             ==> [1, 32, 32, 32]   size=(h 256, w 256)

        deal(function meaning):
            self.linear_1   Conv2d(dim 32, dim 32, 1 k, 1 s, 0 p) ==> [1,32,32,32]
            self.gelu   ==> [1,32,32,32]
            F.interpolate   ==> [1, 32, 256, 256]
        out(type shape):            
            x.shape = [1, 32, 256, 256]
            x_l.shape = [1, 32, 256, 256]
                    
        '''
       
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,32,256,256)
    model = SMFA(dim=32)
    output = model (input)
    
    print('input_size:', input.size())
    print('output_size:', output.size())
