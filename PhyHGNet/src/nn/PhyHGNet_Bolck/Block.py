import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from einops import rearrange
from collections import OrderedDict
from timm.layers import trunc_normal_
from timm.layers import DropPath

class BasicBlock_Faster_Block_CGLU(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = Faster_Block_CGLU(ch_out, ch_out)
class Faster_Block_CGLU(nn.Module):
    """
    [轻量化重构版] Faster_Block_CGLU
    适用场景: Backbone 主干网络
    核心思想: 
    1. 移除沉重的 Self-Attention 和 多分支卷积。
    2. 引入 Physics-Informed (物理感知) 的特征提取：
       - Laplacian Operator: 提取高频边缘 (Edge/Detail)
       - Diffusion Filter: 提取低频背景 (Context/Background)
    3. Math-Prior: 利用 GLU (Gated Linear Unit) 实现通道间的非线性选择。
    """
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,           # 兼容参数，保留接口
                 mlp_ratio=2,       # 控制 CGLU 的膨胀比
                 drop_path=0.0,
                 layer_scale_init_value=1e-6,
                 pconv_fw_type='split_cat', # 兼容参数
                 # 遥感/物理参数
                 use_spectral_attention=False, # 为了轻量化，建议在Backbone中设为False
                 spectral_bands=None,
                 **kwargs # 吸收多余参数
                 ):
        super().__init__()
        
        # 1. 维度对齐 (Projection)
        # 如果输入输出通道不一致，使用 1x1 卷积调整
        self.proj = nn.Conv2d(inc, dim, 1, bias=False) if inc != dim else nn.Identity()
        
        # 2. 物理先验空间混合器 (Physics-Informed Spatial Mixer)
        # 替代了原有的 MultiScalePartialConv
        self.spatial_mixer = PhysicsSpatialMixer(dim)

        # 3. 卷积门控单元 (CGLU) - 轻量化版
        # 替代了原有的 MLP 和 ConvolutionalGLU
        self.cglu = LightweightCGLU(dim, expansion=mlp_ratio)

        # 4. DropPath 和 LayerScale (训练稳定性)
        self.drop_path = nn.Identity() # 假设外部会处理，或者这里可以使用 timm 的 DropPath
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale = None

    def forward(self, x):
        # 1. 维度对齐
        x = self.proj(x)
        shortcut = x
        
        # 2. 空间混合 (提取边缘 + 扩散背景)
        x = self.spatial_mixer(x)
        
        # 3. 通道混合与门控 (CGLU)
        x = self.cglu(x)
        
        # 4. Layer Scale
        if self.layer_scale is not None:
            x = x * self.layer_scale.unsqueeze(-1).unsqueeze(-1)
            
        # 5. 残差连接
        x = shortcut + x
        
        return x


class PhysicsSpatialMixer(nn.Module):
    """
    物理空间混合器
    原理：信号处理中的 [高频提取] + [低频扩散]
    """
    def __init__(self, dim):
        super().__init__()
        # A. 拉普拉斯锐化算子 (Laplacian Edge Detector)
        # 这是一个固定的物理算子，不需要训练参数 (Parameter-Free)，极致轻量
        # 算子形状: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        self.laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('fixed_laplacian', self.laplacian_kernel)
        
        # B. 扩散卷积 (Diffusion Convolution)
        # 模拟热扩散，获取局部上下文。使用 Depthwise Conv，开销极小。
        self.diffusion = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # 初始化为高斯模糊核类似的权重，利于冷启动
        nn.init.constant_(self.diffusion.weight, 1.0/9.0)

        # C. 特征融合权重
        # 学习如何平衡 高频(Edge) 和 低频(Context)
        self.fusion_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)

    def forward(self, x):
        # x: [B, C, H, W]
        
        # Path 1: 物理边缘提取 (High Frequency)
        # 使用 F.conv2d 直接计算，groups=C 对每个通道独立计算
        # 注意：这里为了速度，我们复用输入x的通道作为groups
        b, c, h, w = x.shape
        laplacian_feat = F.conv2d(x, self.fixed_laplacian.repeat(c, 1, 1, 1), padding=1, groups=c)
        
        # Path 2: 扩散背景提取 (Low Frequency)
        diffusion_feat = self.diffusion(x)
        
        # 融合: 原图 + α * 边缘 + (1-α) * 背景
        # 这比单纯的卷积更能捕捉遥感图像中的纹理细节
        return x + laplacian_feat * self.fusion_weight + diffusion_feat * (1 - self.fusion_weight)


class LightweightCGLU(nn.Module):
    """
    轻量级卷积门控单元 (CGLU)
    数学原理: y = (x * W1) * sigmoid(x * W2)
    """
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = int(dim * expansion)
        
        # 为了极致轻量，我们在 CGLU 内部不使用大卷积，只使用 1x1
        # 上下文已经在 PhysicsSpatialMixer 中获取了
        self.fc1 = nn.Conv2d(dim, hidden_dim * 2, 1) # 升维
        self.dwc = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1, groups=hidden_dim * 2, bias=False) # 极轻量的深度卷积
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1) # 降维

    def forward(self, x):
        x_in = x
        
        # 1. 升维投影
        x = self.fc1(x)
        
        # 2. 深度卷积 (提取特征用于门控)
        x = self.dwc(x)
        
        # 3. 门控机制 (Split -> Element-wise Multiply)
        x_res, x_gate = x.chunk(2, dim=1)
        x = x_res * self.act(x_gate) # GLU 变体
        
        # 4. 降维投影
        x = self.fc2(x)
        
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange

# ----------------- 1. 必要的数学辅助函数 -----------------

def rotate_every_two(x):
    """
    数学含义：在复平面上对每两个元素进行旋转操作的辅助步骤
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    """
    数学含义：应用欧拉公式 e^(ix) = cos(x) + i*sin(x) 进行旋转位置编码
    """
    return (x * cos) + (rotate_every_two(x) * sin)

# ----------------- 2. RoPE 类定义 -----------------

class RoPE(nn.Module):
    """
    Rotary Positional Embedding (旋转位置编码)
    物理含义：将绝对位置映射为高维空间中的旋转角度，保持相对距离的旋转不变性。
    """
    def __init__(self, embed_dim, num_heads):
        '''
        embed_dim: 特征维度
        num_heads: 注意力头数
        '''
        super().__init__()
        # 生成不同频率的角度 (Angle), 对应傅里叶频域中的不同分量
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)
    
    def forward(self, slen: Tuple[int]):
        '''
        slen: (h, w) 输入特征图的高和宽
        '''
        # 生成网格坐标
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)

        # 计算正弦 (Sin) 和 余弦 (Cos) 分量
        # 广播机制生成 2D 频率栅格
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)

        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        # 返回 sin 和 cos 用于后续的 theta_shift
        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos
        
        
class LaplacianThermalSharpening(nn.Module):
    """
    物理模块：基于拉普拉斯算子的热斑增强
    数学公式：Out = In - \lambda * \nabla^2(In)
    物理含义：逆向热扩散，还原模糊前的热源形态
    """
    def __init__(self, channels):
        super().__init__()
        # 定义拉普拉斯卷积核 (边缘/斑点检测算子)
        #  0  -1   0
        # -1   4  -1
        #  0  -1   0
        kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.groups = channels
        # 可学习的扩散系数 lambda
        self.diff_coeff = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x):
        # 计算拉普拉斯响应 (二阶导数)
        laplacian = F.conv2d(x, self.kernel, padding=1, groups=self.groups)
        # 原始特征 + 逆扩散 (锐化)
        return x + self.diff_coeff * laplacian

class LiteFFN(nn.Module):
    """
    [轻量化] 替代笨重的标准 FFN
    使用 Depthwise Separable Conv 结构，大幅减少参数量
    结构: 1x1 Conv (升维) -> 3x3 DW-Conv (局部提取) -> 1x1 Conv (降维)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 1. Pointwise Conv (升维/特征融合)
        self.pw1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        
        # 2. Depthwise Conv (低成本局部特征提取)
        self.dw1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        
        self.act = act_layer()
        
        # 3. Pointwise Conv (降维/输出)
        self.pw2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.act(self.bn1(self.pw1(x)))
        x = self.act(self.bn2(self.dw1(x)))
        x = self.bn3(self.pw2(x))
        return x

class TransformerEncoderLayer_MALA(nn.Module):
    """
    [轻量化 + 物理增强] Lite-Thermo-MALA Layer
    特点：
    1. 保留 Laplacian 物理锐化 (针对红外)
    2. 使用 LiteFFN 替代标准 FFN (减少70%参数)
    3. 使用 BatchNorm 替代 LayerNorm (推理加速)
    """
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        # 1. 核心注意力 (MALA)
        self.attn = MALA(c1, num_heads=num_heads)
        
        # 2. 物理增强 (保留!)
        self.thermal_sharpener = LaplacianThermalSharpening(c1)
        
        # 3. 轻量化 FFN
        # 注意：这里我们限制中间层维度 cm。如果 cm 很大(如2048)，建议强制缩小以减重
        # 对于红外任务，c1*2 或 c1*4 足够，c1=256时，cm建议设为 512 或 1024
        lite_cm = min(cm, c1 * 3) # 限制膨胀系数不超过 3 倍
        self.ffn = LiteFFN(c1, hidden_features=lite_cm, act_layer=type(act))

        # 4. 归一化 (使用 BN 替代 LN，适合 CNN 架构)
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c1)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 增加一个可学习的比例系数，控制物理锐化的强度 (可选)
        self.phys_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """
        结构优化：Post-Norm 结构，且移除繁琐的 permute
        """
        # --- Block 1: Attention ---
        residual = src
        # MALA 期望输入 (B, C, H, W)
        x = self.attn(src)
        x = self.dropout1(x)
        x = residual + x
        x = self.norm1(x) # BN 直接处理 (B, C, H, W)

        # --- Block 2: Physics + FFN ---
        residual = x
        
        # [物理步骤]：先对特征进行"热扩散逆转"(锐化)
        # 加上 residual 形成 Skip Connection，防止特征退化
        # 公式: FFN( x + alpha * Laplacian(x) )
        x_phys = x + self.phys_alpha * self.thermal_sharpener(x)
        
        # 送入轻量化 FFN
        x = self.ffn(x_phys)
        x = self.dropout2(x)
        x = residual + x
        x = self.norm2(x)

        return x
    
class Converse2DC3(RepC3):
    def __init__(self, c1, c2, n=3, e=1):
        super().__init__(c1, c2, n, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Converse2D(c_, c_, 3) for _ in range(n)])

class Converse2D(nn.Module):
    """
    [物理重构版] Converse2D -> PhysicsConverse2D
    功能：利用热扩散的逆过程（Inverse Diffusion）增强红外小目标。
    兼容性：输入输出接口与标准 Conv 模块一致，支持 c1 != c2。
    """
    def __init__(self, c1, c2, k=5, e=0.5):
        """
        参数:
        c1: 输入通道数
        c2: 输出通道数
        k:  扩散卷积核大小 (kernel_size), 默认 5
        e:  预留参数 (expansion/ratio), 保持与其他 YOLO 模块参数对齐，此处暂未使用或可用于控制内部通道
        """
        super().__init__()
        
        # 1. 物理增强核心 (在 c1 维度上进行，保持特征完整性)
        # Depthwise Convolution 模拟热传导
        self.diffusion = nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False)
        
        # 初始化为平滑核 (Gaussian-like)，模拟物理模糊
        nn.init.constant_(self.diffusion.weight, 1.0 / (k ** 2))

        # 逆扩散强度 (可学习参数)
        self.alpha = nn.Parameter(torch.zeros(1, c1, 1, 1)) 
        
        # 2. 通道投影 (Projection)
        # 如果 c1 != c2，通过 1x1 卷积调整通道；如果相等，则使用 Identity (直连)
        # 这样保证了模块可以替换任何位置的 Conv 层
        self.project = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
        
        # 3. 激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, c1, H, W]
        
        # --- 物理层: 逆热扩散增强 ---
        # Step 1: 模拟热扩散 (获取低频背景)
        heat_background = self.diffusion(x)
        
        # Step 2: 提取热对比度 (原始图 - 背景 = 高频细节/小目标)
        local_contrast = x - heat_background
        
        # Step 3: 注入增强 (原图 + alpha * 对比度)
        enhanced_feat = x + self.alpha * local_contrast
        
        # --- 结构层: 维度调整与激活 ---
        # 调整通道数 c1 -> c2，并进行非线性激活
        return self.act(self.project(enhanced_feat))
class HyperComputeModule(nn.Module):
    """
    [极轻量化] 基于平均场理论(Mean Field)的近似玻尔兹曼超图
    复杂度从 O(N^2) 降低到 O(N)，同时修复通道不匹配问题。
    """
    def __init__(self, c1, c2, threshold=None):
        super().__init__()
        self.c_in = c1
        self.c_out = c2
        
        # 1. 自动适配通道数 (Shortcut)
        if c1 != c2:
            self.shortcut = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        else:
            self.shortcut = nn.Identity()

        # 2. 降维以减少计算量 (Squeeze ratio = 4)
        mid_c = c2 // 4
        self.fc_in = nn.Conv2d(c1, mid_c, 1, 1) # 降维
        
        # 3. 全局场 (Global Field) - 模拟超图中的"超节点"
        # 使用自适应池化获取全局上下文
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_global = nn.Conv2d(mid_c, mid_c, 1, 1)
        
        # 4. 局部场 (Local Field) - 模拟邻居传播
        # 使用大核深度卷积模拟局部热扩散
        self.local_diffuse = nn.Conv2d(mid_c, mid_c, 5, 1, 2, groups=mid_c, bias=False)
        
        # 5. 玻尔兹曼能量门控
        # 温度参数 T
        self.T = nn.Parameter(torch.tensor(1.0))
        
        # 6. 还原通道
        self.fc_out = nn.Conv2d(mid_c, c2, 1, 1)
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, C_in, H, W]
        
        # --- 1. 处理 Shortcut (解决通道不匹配的核心) ---
        residual = self.shortcut(x)
        
        # --- 2. 降维特征空间 ---
        x_embed = self.fc_in(x) # [B, mid_c, H, W]
        
        # --- 3. 物理场计算 (O(N) Complexity) ---
        
        # A. 全局势能 (Global Potential)
        # 任何一个像素与全局平均状态的差异
        global_vec = self.global_pool(x_embed) # [B, mid_c, 1, 1]
        global_context = self.fc_global(global_vec)
        
        # B. 局部热扩散 (Local Diffusion)
        local_context = self.local_diffuse(x_embed)
        
        # --- 4. 玻尔兹曼分布激活 ---
        # Energy = -(Local + Global)
        # Probability = Sigmoid(Energy / T)
        # 这里我们模拟 "能量越低(差异越小)，连接越紧密" 或者 "能量越高(热斑)，激活越强"
        # 直接利用点积相似度作为能量的近似
        energy = (x_embed * global_context) + local_context
        
        # 玻尔兹曼门控 (Soft-Mask)
        # T 控制激活的"锐度"
        gate = torch.sigmoid(energy / (torch.abs(self.T) + 1e-6))
        
        # 应用门控：特征重加权
        x_activated = x_embed * gate
        
        # --- 5. 还原与融合 ---
        x_out = self.fc_out(x_activated)
        
        # 残差连接
        return self.act(self.bn(residual + x_out))