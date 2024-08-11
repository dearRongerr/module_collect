import itertools
import torch
# https://arxiv.org/pdf/2305.07027
'''
EfficientViT：具有级联群注意力的内存高效视觉转换器    cvpr 2023 顶会论文

Cascaded Group Attention (CGA) 本文中提出的一种新型注意力机制。
其核心思想是增强输入到注意力头的特征的多样性。与以前的自注意力不同，
它为每个头提供不同的输入分割，并跨头级联输出特征。
这种方法不仅减少了多头注意力中的计算冗余，而且通过增加网络深度来提升模型容量。

CGA注意力是 将输入特征分成不同的部分，每部分输入到一个注意力头
每个头计算其自注意力映射，然后将所有头的输出级联起来，
并通过一个线性层将它们投影回输入的维度。通过这样的方式，
CGA 在不增加额外参数的情况下提高了模型的计算效率。
另外，通过串联的方式，每个头的输出都会添加到下一个头的输入中，从而逐步精化特征表示。

'''

# TODO 明白输入、输出、Function 形状以及含义

# 定义一个带有批量归一化（Batch Normalization）的2D卷积模块
class Conv2d_BN(torch.nn.Sequential):
    # TODO 归一化方法改并且融合
    '''
    Conv2d_BN 类是一个卷积和批量归一化的组合模块，用于在训练时使用动态权重，
    并提供了一个方法 switch_to_deploy 用于将动态权重转换为静态权重，以便于模型部署。    
    '''

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        # 初始化卷积和批量归一化层，并添加到序列模块中
        # 初始化方法，创建一个卷积层和批量归一化层的序列
        super().__init__()# 调用基类的初始化方法

        # 添加一个2D卷积层到序列中
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('c', torch.nn.Conv2d(
            in_channels=a,  # 输入通道数
            out_channels=b,  # 输出通道数
            kernel_size=ks,  # 卷积核大小
            stride=stride,  # 步长
            padding=pad,  # 补零填充
            dilation=dilation,  # 扩张率
            groups=groups,  # 组数，用于分组卷积
            bias=False))  # 没有偏置项，因为批量归一化层会学习偏置
        

        # 添加一个2D批量归一化层到序列中
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        
        # 初始化批量归一化的权重为1，偏置为0
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()# 装饰器表示该方法内部不需要梯度计算
    def switch_to_deploy(self):
        # 将训练时使用的动态权重转换为静态权重，用于部署
        c, bn = self._modules.values() # 获取卷积层和批量归一化层的引用
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5 # 计算批量归一化的权重缩放因子
        w = c.weight * w[:, None, None, None]   # 将缩放因子应用于卷积层权重
        # 计算新的偏置项，考虑了批量归一化的偏置和均值
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        # 创建一个新的静态卷积层，融合了卷积和批量归一化的权重和偏置
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w) # 复制融合后的权重到新卷积层
        m.bias.data.copy_(b)    # 复制融合后的偏置到新卷积层
        return m    # 返回融合后的静态卷积层
# 定义级联群注意力模块
class CascadedGroupAttention(torch.nn.Module):
    '''
    CascadedGroupAttention 类实现了级联群注意力机制，用于增强特征多样性，并逐步精化特征表示。

    CascadedGroupAttention 类实现了级联群注意力机制，
    它通过将输入特征分割成不同的部分并输入到不同的注意力头中来增强特征的多样性。
    每个头计算自注意力映射后，将所有头的输出级联起来，并通过一个线性层将它们投影回原始输入的维度。
    这个过程不仅减少了计算冗余，而且通过串联的方式逐步精化特征表示。
    '''
    r""" Cascaded Group Attention.
    Args:
        dim (int): Number of input channels.输入通道数。
        key_dim (int): The dimension for query and key.查询和键的维度。
        num_heads (int): Number of attention heads.注意力头的数量。
        attn_ratio (int): Multiplier for the query dim for value dimension.值维度与查询维度的比例。
        resolution (int): Input resolution, correspond to the window size.输入分辨率，对应于窗口大小。
        kernels (List[int]): The kernel size of the dw conv on query.应用于查询的深度卷积的核大小。
    """

    def __init__(self, dim, num_heads=4,
                 attn_ratio=4,
                 resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()# 调用基类的构造函数

        key_dim = dim //16 # 计算每个头的键（key）维度，这里是输入通道维度除以16
        self.num_heads = num_heads # 注意头的数量
        self.scale = key_dim ** -0.5    # 缩放因子，用于调整注意力分数
        self.key_dim = key_dim  # 每个头的键（key）维度
        self.d = int(attn_ratio * key_dim)  # 值（value）维度，是键维度的attn_ratio倍
        self.attn_ratio = attn_ratio    # 值维度与查询维度的比例


        # 为每个注意力头创建查询、键、值和深度卷积模块
        # 初始化查询、键、值和深度卷积模块的列表
        qkvs = []
        dws = []

        # 对于每个注意力头
        for i in range(num_heads):

            # 创建一个Conv2d_BN模块，用于生成每个头的查询、键和值
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            # 创建一个Conv2d_BN模块，用于在每个头中应用深度卷积
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        
        # 将查询、键、值模块封装成ModuleList
        self.qkvs = torch.nn.ModuleList(qkvs)
        # 将深度卷积模块封装成ModuleList
        self.dws = torch.nn.ModuleList(dws)
        # 定义输出的投影层，用于将注意力输出映射回原始维度
        self.proj = torch.nn.Sequential(
            # 激活函数
            torch.nn.ReLU(),  
            # 投影卷积和批量归一化
            Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        # 构建注意力偏置索引，用于在每个头中添加不同的偏置  
        points = list(itertools.product(range(resolution), range(resolution)))  # 产生窗口内所有点的组合
        N = len(points)  # 点的总数
        attention_offsets = {}  # 存储偏置的字典
        idxs = []   # 存储偏置索引的列表
        for p1 in points:   # 对于窗口内每对点
            for p2 in points:   # 计算每对点之间的偏移
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                # 如果偏移未记录，则添加到字典
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                
                # 将偏置索引添加到列表
                idxs.append(attention_offsets[offset])
        
        # 将偏置转换为模型参数
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        
        # 注册偏置索引为常驻缓冲区
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))


    # CascadedGroupAttention 类的训练模式切换方法和前向传播方法的实现
    @torch.no_grad()    # 装饰器，表示这个函数执行时不计算梯度
    def train(self, mode=True):
        # 训练模式切换，如果mode为True，则切换到训练模式，否则切换到评估模式
        super().train(mode) # 调用基类的train方法

        # 如果处于训练模式且之前已经计算过静态权重，则删除静态权重
        if mode and hasattr(self, 'ab'):
            del self.ab
        # 否则，计算静态权重并保存
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    
    # 前向传播方法，输入x的尺寸为(B,C,H,W)，分别代表批次大小、通道数、高度和宽度
    def forward(self, x):  # x (B,C,H,W)
        # 分割输入特征
        B, C, H, W = x.shape

        # 如果处于训练模式，使用动态计算的注意力偏置
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        # 将输入特征沿通道方向分割，每部分输入到一个注意力头
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []  # 初始化输出特征列表
        feat = feats_in[0]  # 获取第一个头的输入特征
        # 遍历每个注意力头
        for i, qkv in enumerate(self.qkvs):
            if i > 0:   # 如果不是第一个头，将前一个头的输出添加到输入中
                feat = feat + feats_in[i]
            
            # 通过查询、键、值生成模块处理特征
            feat = qkv(feat)

            # 分离查询、键、值，其中每个头的查询、键、值通过卷积层生成后分割出来
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
            
            # 对查询进行深度卷积处理
            q = self.dws[i](q)

             # 展平查询、键、值，从(B, C/h, H, W)变为(B, C/h, N)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N # 展平查询、键、值
            # 计算注意力分数并应用偏置
            
            # 计算注意力分数并应用偏置，其中N是展平后的长度
            attn = ((q.transpose(-2, -1) @ k) * self.scale+(trainingab[i] if self.training else self.ab[i]))
            
            # 对注意力分数应用softmax归一化
            attn = attn.softmax(dim=-1)  # BNN  # 应用softmax归一化
            # 计算加权的值，通过矩阵乘法更新特征表示，并重塑回原始空间维度
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
            # 将当前头的输出特征添加到输出列表
            feats_out.append(feat)
        
        # 将所有头的输出特征在通道方向上进行拼接，并通过投影层
        x = self.proj(torch.cat(feats_out, 1))
        return x    # 返回最终的输出特征

if __name__ == '__main__':

    # 创建一个Conv2d_BN模块实例，例如：输入通道16，输出通道24，卷积核大小3x3，步长1，填充1
    input = torch.randn(1,64,32,32)
    model = CascadedGroupAttention(dim=64,resolution=32) #resolution要求和图片大小一样
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

    #查看模型结构（每个层的结构） 
    print(model)

    # 查看模型名称和大小
    for name, param in model.named_parameters():
        print(name, param.size())
    # 查看模块的参数数量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {param_count}")

'''
input_size: torch.Size([1, 64, 32, 32])
output_size: torch.Size([1, 64, 32, 32])
CascadedGroupAttention(
  (qkvs): ModuleList(
    (0-3): 4 x Conv2d_BN(
      (c): Conv2d(16, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (dws): ModuleList(
    (0-3): 4 x Conv2d_BN(
      (c): Conv2d(4, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=4, bias=False)
      (bn): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (proj): Sequential(
    (0): ReLU()
    (1): Conv2d_BN(
      (c): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
)
attention_biases torch.Size([4, 1024])
qkvs.0.c.weight torch.Size([24, 16, 1, 1])
qkvs.0.bn.weight torch.Size([24])
qkvs.0.bn.bias torch.Size([24])
qkvs.1.c.weight torch.Size([24, 16, 1, 1])
qkvs.1.bn.weight torch.Size([24])
qkvs.1.bn.bias torch.Size([24])
qkvs.2.c.weight torch.Size([24, 16, 1, 1])
qkvs.2.bn.weight torch.Size([24])
qkvs.2.bn.bias torch.Size([24])
qkvs.3.c.weight torch.Size([24, 16, 1, 1])
qkvs.3.bn.weight torch.Size([24])
qkvs.3.bn.bias torch.Size([24])
dws.0.c.weight torch.Size([4, 1, 5, 5])
dws.0.bn.weight torch.Size([4])
dws.0.bn.bias torch.Size([4])
dws.1.c.weight torch.Size([4, 1, 5, 5])
dws.1.bn.weight torch.Size([4])
dws.1.bn.bias torch.Size([4])
dws.2.c.weight torch.Size([4, 1, 5, 5])
dws.2.bn.weight torch.Size([4])
dws.2.bn.bias torch.Size([4])
dws.3.c.weight torch.Size([4, 1, 5, 5])
dws.3.bn.weight torch.Size([4])
dws.3.bn.bias torch.Size([4])
proj.1.c.weight torch.Size([64, 64, 1, 1])
proj.1.bn.weight torch.Size([64])
proj.1.bn.bias torch.Size([64])
Total number of parameters: 10480

'''
