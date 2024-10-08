import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# 定义一个函数，用于对神经网络模块的权重进行正态分布初始化
def normal_init(module, mean=0, std=1, bias=0):
    # 检查模块是否有权重属性，并且权重不为None
    if hasattr(module, 'weight') and module.weight is not None:
        # 使用正态分布初始化权重，均值为mean，标准差为std
        nn.init.normal_(module.weight, mean, std)
    # 检查模块是否有偏置属性，并且偏置不为None
    if hasattr(module, 'bias') and module.bias is not None:
        # 将偏置初始化为bias指定的值
        nn.init.constant_(module.bias, bias)

# 定义一个函数，用于将神经网络模块的权重初始化为一个常数值
def constant_init(module, val, bias=0):
     # 检查模块是否有权重属性，并且权重不为None
    if hasattr(module, 'weight') and module.weight is not None:
        # 将权重初始化为val指定的常数值
        nn.init.constant_(module.weight, val)
    # 检查模块是否有偏置属性，并且偏置不为None    
    if hasattr(module, 'bias') and module.bias is not None:
        # 将偏置初始化为bias指定的值
        nn.init.constant_(module.bias, bias)

'''
题目：Learning to Upsample by Learning to Sample
即插即用的上采样模块：DySample

我们推出了 DySample，这是一款超轻且高效的动态上采样器。
虽然最近基于内核的动态上采样器（如 CARAFE、FADE 和 SAPA）取得了令人印象深刻的性能提升，
但它们引入了大量工作负载，主要是由于耗时的动态卷积和用于生成动态内核的额外子网络。
我们实现了一个新上采样器 DySample。

该上采样适用于：语义分割、目标检测、实例分割、全景分割。
style='lp' / ‘pl’ 用该模块上采样之前弄清楚这两种风格
'''

# 构造函数初始化DySample_UP模块
class DySample_UP(nn.Module):
    # in_channels=64 style='lp'
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super(DySample_UP,self).__init__()
        self.scale = scale  # 上采样的尺度因子，默认为2
        self.style = style # 上采样的风格，可以是'lp'或'pl'
        self.groups = groups # 组数，用于分组卷积

        # 确保上采样风格是有效的
        assert style in ['lp', 'pl']
        # 如果风格是'pl'，则输入通道数必须是scale的平方，并且是scale的倍数
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        # 输入通道数必须至少等于组数，并且是组数的倍数
        assert in_channels >= groups and in_channels % groups == 0

        # 根据风格设置输入和输出通道数
        if style == 'pl':
            # 对于'pl'风格，调整输入通道数
            in_channels = in_channels // scale ** 2
            # 输出通道数为组数的两倍
            out_channels = 2 * groups
        else:
            # 对于'lp'风格，输出通道数为组数乘以scale的平方 32 = 2 * groups 4 * scale 2 ** 2
            out_channels = 2 * groups * scale ** 2

        # 定义一个卷积层用于生成偏移量 self.offset = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        self.offset = nn.Conv2d(in_channels, out_channels, 1) # in_channels=64 out_channels=32 kernel_size=(1,1)
        # 使用标准差为0.001的正态分布初始化偏移量卷积层
        normal_init(self.offset, std=0.001)

        # 如果启用了dyscope（动态作用域），则添加一个额外的卷积层 default = 'False'
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            # 使用常数0初始化作用域卷积层
            constant_init(self.scope, val=0.)

        # 注册一个缓冲区init_pos，用于存储初始化的偏移位置
        self.register_buffer('init_pos', self._init_pos())
    
    # 初始化偏移位置的方法
    def _init_pos(self):
        # 使用arange生成一个从-self.scale/2到self.scale/2的序列，然后除以scale进行归一化
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        # 使用meshgrid生成网格，然后stack和transpose组合成一个2D偏移量矩阵
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    # sample 方法是上采样过程中对输入特征图 x 进行采样的核心函数
    def sample(self, x, offset):
        # 获取offset的尺寸，B是批次大小，H和W分别是特征图的高度和宽度
        B, _, H, W = offset.shape
        # 调整offset的视角，使其适用于后续的采样过程
        offset = offset.view(B, 2, -1, H, W)
        # 创建一个网格坐标，表示特征图中每个像素的位置
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        # 归一化网格坐标，使其范围在[-1, 1]，这是F.grid_sample所需的坐标范围
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        # 使用pixel_shuffle调整coords的维度，以匹配后续的采样操作
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        # 使用grid_sample根据调整后的coords对x进行采样
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    # forward_lp是局部感知（Local Perception）风格的上采样方法
    def forward_lp(self, x):
        # 如果定义了scope，则使用scope调整offset hasattr(self, 'scope') = False
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            # 否则，直接使用offset并加上初始化偏移 
            # x.shape = [1, 256, 4, 4] 
            # self.offset(x).shape=[1, 32, 4, 4] 
            # self.init_pos.shape=torch.Size([1, 32, 1, 1])
            offset = self.offset(x) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("lp",i.shape)
         # 调用sample方法进行上采样 
         # x.shape = [1,256,4,4] 
         # offset.shape = [1, 32, 4, 4] 
         #  self.sample(x, offset).shape = [1,256,8,8]
        return self.sample(x, offset)

    # forward_pl是像素洗牌后局部感知（Pixel Shuffle then Local Perception）风格的上采样方法
    def forward_pl(self, x):
        # 首先使用pixel_shuffle对x进行像素洗牌
        x_ = F.pixel_shuffle(x, self.scale)
        # 如果定义了scope，则使用scope调整offset
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        # 否则，直接使用offset并加上初始化偏移
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        # 调用sample方法进行上采样
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
 

if __name__ == '__main__':
    input = torch.rand(4, 256, 64, 64)
    # in_channels=64, scale=4, style='lp'/‘pl’,
    DySample_UP = DySample_UP(in_channels=256,scale=2,style='lp')
    output = DySample_UP(input) # output.shape = [1,256,8,8]
    print('input_size:', input.size())
    print('output_size:', output.size())

#     print(DySample_UP)
#     # 查看模型名称和大小
#     for name, param in DySample_UP.named_parameters():
#         print(name, param.size())
#     # 查看模块的参数数量
#     param_count = sum(p.numel() for p in DySample_UP.parameters())
#     print(f"Total number of parameters: {param_count}") 


# input_size: torch.Size([1, 64, 4, 4])
# output_size: torch.Size([1, 64, 8, 8])
# DySample_UP(
#   (offset): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
# )
# offset.weight torch.Size([32, 64, 1, 1])
# offset.bias torch.Size([32])
# Total number of parameters: 2080
