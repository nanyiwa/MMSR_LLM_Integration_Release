import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CBAM 注意力机制
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # AvgPool path
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # MaxPool path
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 相加后 Sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)  # 先通道注意
        result = out * self.sa(out)  # 再空间注意
        return result


# ==========================================
# 2. 动态残差块
# ==========================================
class DynamicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False, dropout=0.0):
        super(DynamicResBlock, self).__init__()

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 内部正则化
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # 主路径
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # 插入注意力
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ==========================================
# 3. Transformer 风格注意力头
# ==========================================
class TransformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(TransformerAttention, self).__init__()
        # Multi-head Self Attention
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # LayerNorm (Add & Norm 结构)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        attn_output, _ = self.mha(x, x, x)

        # Residual + Norm
        return self.layer_norm(x + attn_output)