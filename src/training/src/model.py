import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """标准的 3x3 卷积块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块 (Depthwise Separable Convolution)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. Depthwise: 逐通道卷积，提取空间特征
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 2. Pointwise: 1x1 卷积，融合通道特征
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class RMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入尺寸: (Batch, 3, 320, 320)
        
        # Stage 1: 浅层特征，使用标准卷积快速降低分辨率并提取基础色彩/边缘
        # 3 -> 16, stride=2, 输出尺寸: 160x160
        self.stage1 = ConvBNReLU(3, 16, stride=2)
        
        # Stage 2: 16 -> 32, stride=2, 输出尺寸: 80x80
        self.stage2 = DepthwiseConvBlock(16, 32, stride=2)
        
        # Stage 3: 32 -> 64, stride=2, 输出尺寸: 40x40 (用于输出高分辨率特征)
        self.stage3 = DepthwiseConvBlock(32, 64, stride=2)
        
        # Stage 4: 64 -> 128, stride=2, 输出尺寸: 20x20
        self.stage4 = DepthwiseConvBlock(64, 128, stride=2)
        
        # Stage 5: 128 -> 256, stride=2, 输出尺寸: 10x10 (用于输出高语义特征)
        self.stage5 = DepthwiseConvBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        
        # 获取 Stage 3 的输出 (提供给 Neck 用于高精度角点定位)
        feat_stage3 = self.stage3(x)
        
        x = self.stage4(feat_stage3)
        
        # 获取 Stage 5 的输出 (提供给 Neck 用于高语义分类)
        feat_stage5 = self.stage5(x)
        
        return feat_stage3, feat_stage5
    
class RMNeck(nn.Module):
    """特征融合层 (FPN 变体)"""
    def __init__(self, in_channels_s3=64, in_channels_s5=256, out_channels=256):
        super().__init__()
        # 为了将 Stage 3 的 40x40 降维到 10x10 以对齐 Stage 5，
        # 我们使用一个步长为 4 的深度可分离卷积，避免直接 MaxPool 带来的细粒度位置信息丢失。
        self.downsample_s3 = nn.Sequential(
            nn.Conv2d(in_channels_s3, in_channels_s3, kernel_size=4, stride=4, 
                      groups=in_channels_s3, bias=False),
            nn.BatchNorm2d(in_channels_s3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_s3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 将 64 通道的 s3_down 与 256 通道的 s5 拼接后，通道数为 320
        # 再通过 1x1 卷积和 3x3 卷积融合特征，降低通道数
        self.fuse = nn.Sequential(
            ConvBNReLU(64 + in_channels_s5, out_channels),
            DepthwiseConvBlock(out_channels, out_channels)
        )

    def forward(self, feat_s3, feat_s5):
        # feat_s3: [B, 64, 40, 40] -> downsample -> [B, 64, 10, 10]
        feat_s3_down = self.downsample_s3(feat_s3)
        
        # Concat 拼接
        # feat_s5: [B, 256, 10, 10]
        out = torch.cat([feat_s3_down, feat_s5], dim=1) # [B, 320, 10, 10]
        
        # 特征融合
        out = self.fuse(out) # [B, 256, 10, 10]
        return out


class RMHead(nn.Module):
    """解耦检测头 (Decoupled Head)"""
    def __init__(self, in_channels=256):
        super().__init__()
        # Box Head: 分支 A，输出 1 维置信度 + 4 维边界框偏移 (tx, ty, tw, th)
        self.box_head = nn.Conv2d(in_channels, 5, kernel_size=1, stride=1)
        
        # Pose Head: 分支 B，输出 8 维关键点坐标偏移量
        self.pose_head = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)

    def forward(self, x):
        box_out = self.box_head(x)   # [B, 5, 10, 10]
        pose_out = self.pose_head(x) # [B, 8, 10, 10]
        
        # 按照此前设定的 13 维张量格式拼接
        # 输出形状: [B, 13, 10, 10]
        out = torch.cat([box_out, pose_out], dim=1)
        return out


class RMDetector(nn.Module):
    """完整的单阶段装甲板检测模型"""
    def __init__(self):
        super().__init__()
        self.backbone = RMBackbone()
        self.neck = RMNeck(in_channels_s3=64, in_channels_s5=256, out_channels=256)
        self.head = RMHead(in_channels=256)

    def forward(self, x):
        # 1. 主干特征提取
        feat_s3, feat_s5 = self.backbone(x)
        # 2. 特征融合
        fused_feat = self.neck(feat_s3, feat_s5)
        # 3. 检测头解耦预测
        out = self.head(fused_feat)
        
        return out

# 测试完整模型
if __name__ == "__main__":
    model = RMBackbone()
    dummy_input = torch.randn(1, 3, 320, 320)
    out3, out5 = model(dummy_input)
    print(f"Stage 3 Output Shape: {out3.shape}") # 预期: [1, 64, 40, 40]
    print(f"Stage 5 Output Shape: {out5.shape}") # 预期: [1, 256, 10, 10]
    # 实例化模型
    model = RMDetector()
    
    # 模拟输入：Batch Size 为 2，3 通道，320x320
    dummy_input = torch.randn(2, 3, 320, 320)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") 
    # 预期输出形状: torch.Size([2, 13, 10, 10])，与 dataset 中生成的 target_tensor 完全对应
    