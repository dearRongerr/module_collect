# --------------------------------------------------------
# 论文：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# GitHub地址：https://github.com/cecret3350/DEA-Net/tree/main
# --------------------------------------------------------
'''
DEA-Net：基于细节增强卷积和内容引导注意力的单图像去雾 (IEEE TIP 2024顶会论文)

我们提出了一种新的注意机制，称为轮廓引导注意（CGA），以一种从粗到细的方式生成特定于通道的sim。
通过使用输入的特征引导SIM的生成，CGA为每个通道分配唯一的SIM，
使模型关注每个通道的重要区域。因此，可以强调用特征编码的更多有用的信息，以有效地提高性能。
此外，还提出了一种基于cga的混合融合方案，可以有效地将编码器部分的低级特征与相应的高级特征进行融合。
'''
import torch
from torch import nn
from einops.layers.torch import Rearrange

# 二维空间注意力（Spatial Attention）模块
class SpatialAttention(nn.Module): 
    # 定义一个名为 SpatialAttention 的类，它继承自 PyTorch 的 nn.Module，用于创建一个空间注意力模块
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 在模块中定义一个二维卷积层，用于空间注意力的计算。
        # 输入通道数为 2，输出通道数为 1。
        # 卷积核大小为 7x7，使用 'reflect' 填充模式，填充大小为 3。
        # 卷积层包含偏置项（bias）。
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        # 定义前向传播函数，它接受输入 x 并返回空间注意力的输出。

        x_avg = torch.mean(x, dim=1, keepdim=True)  # torch.Size([3, 1, 64, 64])
        # 计算输入 x 的通道平均值，dim=1 表示沿着通道维度计算平均值。
        # keepdim=True 表示保持输出的维度与输入相同。

        x_max, _ = torch.max(x, dim=1, keepdim=True) # torch.Size([3, 1, 64, 64])
        # 计算输入 x 的通道最大值，dim=1 表示沿着通道维度寻找最大值。
        # _ 表示我们不关心最大值的索引。

        x2 = torch.cat([x_avg, x_max], dim=1)   # torch.Size([3, 2, 64, 64])
        # 将平均值和最大值沿着通道维度拼接起来，形成一个新的张量 x2。

        sattn = self.sa(x2) # torch.Size([3, 1, 64, 64])
        # 将拼接后的张量 x2 输入到之前定义的卷积层 self.sa 中，得到空间注意力图。

        # 返回空间注意力图，它将作为模块的输出。
        return sattn
    


# 定义一个名为 ChannelAttention 的类，继承自 PyTorch 的 nn.Module，用于创建通道注意力模块。
class ChannelAttention(nn.Module):

    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()

        # 使用全局平均池化层，将输入特征图的每个通道的空间维度压缩到一个单一的数值。
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 定义一个通道注意力子网络，使用顺序容器来堆叠层。
        self.ca = nn.Sequential(

            # 第一个卷积层，输入通道数为 dim，输出通道数为 dim // reduction，用于降维。
            # 卷积核大小为 1x1，没有填充（padding=0），包含偏置项（bias=True）。
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),

            # 使用 ReLU 激活函数，inplace=True 表示在原地进行计算，减少内存使用。
            nn.ReLU(inplace=True),

            # 第二个卷积层，输入通道数为经过降维的 dim // reduction，输出通道数恢复为原始的 dim。
            # 同样使用 1x1 卷积核，没有填充，包含偏置项。
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    # 定义前向传播函数，接受输入 x 并返回通道注意力的输出。
    def forward(self, x): 
        
        # x.shape = torch.Size([3, 32, 64, 64]) 

        # 通过全局平均池化层处理输入 x，得到每个通道的全局空间信息。
        x_gap = self.gap(x) # torch.Size([3, 32, 1, 1])
       
        cattn = self.ca(x_gap) # torch.Size([3, 32, 1, 1])
        # 将全局平均池化的结果输入到通道注意力子网络中，计算每个通道的权重。

        return cattn
        # 返回通道注意力的权重，这些权重将用于调节输入特征图的通道响应

# 定义一个名为 PixelAttention 的类，继承自 PyTorch 的 nn.Module，用于创建像素注意力模块。
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        # 调用父类的构造函数。

        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        # 定义一个二维卷积层，用于像素注意力的计算。
        # 输入通道数为 2 * dim，输出通道数为 dim
        # 卷积核大小为 7x7，使用 'reflect' 填充模式，填充大小为 3。
        # 卷积层使用分组卷积，groups=dim，每个组独立卷积。
        # 卷积层包含偏置项（bias=True）。

        self.sigmoid = nn.Sigmoid()
        # 定义 Sigmoid 激活函数，用于将卷积层的输出转换为概率分布。

    # 定义前向传播函数，接受输入特征图 x 和第一个注意力特征 pattn1，并返回像素注意力的输出。
    def forward(self, x, pattn1):

        # 获取输入特征图 x 的形状。
        B, C, H, W = x.shape

        x = x.unsqueeze(dim=2)  # B, C, 1, H, W torch.Size([3, 32, 1, 64, 64])
        # 扩展输入特征图的维度，增加一个维度，用于与 pattn1 拼接。

        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W torch.Size([3, 32, 1, 64, 64])
        # 扩展 pattn1 的维度，与 x 进行拼接。
        # x2.shape = torch.Size([3, 32, 2, 64, 64])
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W 
        # 将 x 和 pattn1 沿着通道维度拼接。
        # x2.shape = torch.Size([3, 64, 64, 64])
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2) 
        # 使用 Rearrange 函数重新排列 x2 的形状，这里应该是一个错误，因为 PyTorch 没有内置的 Rearrange 函数。
        # 正确的操作可能是使用 view 或 permute 来重新排列张量。


        pattn2 = self.pa2(x2) # pattn2 torch.Size([3, 32, 64, 64])
        # 将拼接和重新排列后的张量 x2 输入到之前定义的卷积层 self.pa2 中。

        pattn2 = self.sigmoid(pattn2) # pattn2 torch.Size([3, 32, 64, 64])
        # 通过 Sigmoid 激活函数处理 self.pa2 的输出，得到像素级别的注意力权重。

        return pattn2
        # 返回像素注意力权重。

class CGAFusion(nn.Module):
    # 定义 CGAFusion 类，继承自 PyTorch 的 nn.Module，用于创建特征融合模块。
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()

        self.sa = SpatialAttention()
        # 创建空间注意力模块。

        self.ca = ChannelAttention(dim, reduction)
        # 创建通道注意力模块，dim 表示输入特征的维度，reduction 用于控制通道降维的比例。

        self.pa = PixelAttention(dim)
        # 创建像素注意力模块，dim 表示输入特征的维度。

        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        # 创建一个二维卷积层，用于处理融合后的特征图，卷积核大小为 1。

        self.sigmoid = nn.Sigmoid()
        # 创建 Sigmoid 激活函数，用于将输出转换为概率分布。

    def forward(self, x, y):
        # 定义前向传播函数，接受两个输入特征图 x 和 y。
        # x.shape = torch.Size([3, 32, 64, 64]) 
        initial = x + y # initial.shape torch.Size([3, 32, 64, 64])
        # 将两个输入特征图相加，作为初始融合特征。

        cattn = self.ca(initial) # cattn torch.Size([3, 32, 1, 1])
        # 计算初始融合特征的通道注意力。

        sattn = self.sa(initial) # sattn torch.Size([3, 1, 64, 64])
        # 计算初始融合特征的空间注意力。

        pattn1 = sattn + cattn # torch.Size([3, 32, 64, 64])
        
        # 将空间注意力和通道注意力相加
        # pattn2 torch.Size([3, 32, 64, 64]) self.pa(initial, pattn1) torch.Size([3, 32, 64, 64])
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        # 计算像素级别的注意力权重，并通过 Sigmoid 函数进行归一化。

        result = initial + pattn2 * x + (1 - pattn2) * y 
        # result torch.Size([3, 32, 64, 64])
        # 根据像素注意力权重调整初始融合特征和输入特征的组合，生成最终的融合结果。

        result = self.conv(result)
         # result torch.Size([3, 32, 64, 64])
        # 使用卷积层进一步处理融合结果。

        return result
        # 返回最终的特征融合结果

# 双分支特征融合
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = CGAFusion(32)
    # TODO input1和input2会起到什么作用
    input1 = torch.rand(3, 32, 64, 64)
    input2 = torch.rand(3, 32, 64, 64)
    output = block(input1, input2) # torch.Size([3, 32, 64, 64])
    print(output.size())
