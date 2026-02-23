import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DynamicResBlock, TransformerAttention


class ACRNNv3(nn.Module):
    def __init__(self, config):

        super(ACRNNv3, self).__init__()
        self.config = config

        # ====================================
        # Part 1: CNN Backbone (特征提取器)
        # ====================================

        self.cnn_layers = nn.ModuleList()

        # 1. 初始处理
        in_c = config.input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, config.cnn_filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(config.cnn_filters[0]),
            nn.ReLU(),
        )
        current_c = config.cnn_filters[0]

        # 2. 动态残差
        for i, out_c in enumerate(config.cnn_filters):
            is_last_layer = (i == len(config.cnn_filters) - 1)
            stride = (2, 1) if is_last_layer else (2, 2)

            block = DynamicResBlock(
                in_channels=current_c,
                out_channels=out_c,
                stride=stride,
                use_cbam=config.use_cbam,  # 启用通道+空间注意力
                dropout=config.cnn_dropout  # 防止过拟合
            )
            self.cnn_layers.append(block)
            current_c = out_c

        # 记录 CNN 最终的通道数 (例如 512)
        self.cnn_out_channels = current_c

        # ====================================
        # Part 2: Feature Bridge (特征桥接 - 新增复杂度)
        # ====================================

        final_freq_dim = 128 // (2 ** len(config.cnn_filters))
        if final_freq_dim < 1: final_freq_dim = 1

        self.flatten_dim = self.cnn_out_channels * final_freq_dim  # e.g., 512 * 8 = 4096

        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.rnn_input_dim = 1024

        # ====================================
        # Part 3: Temporal Modeling (RNN Neck)
        # ====================================

        if config.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.rnn_input_dim,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.rnn_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.rnn_dropout
            )
        elif config.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.rnn_input_dim,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.rnn_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.rnn_dropout
            )

        rnn_out_dim = config.rnn_hidden_size * (2 if config.bidirectional else 1)

        # ====================================
        # Part 4: Attention Head
        # ====================================
        if config.use_attention:
            self.attention = TransformerAttention(
                embed_dim=rnn_out_dim,
                num_heads=config.num_heads,
                dropout=config.attn_dropout
            )

        # ====================================
        # Part 5: Classifier
        # ====================================
        self.fc = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.fc_dropout),
            nn.Linear(128, config.num_classes)
        )

    def forward(self, x):
        # Input: (Batch, 1, Freq=128, Time=130)

        # 1. CNN Backbone
        x = self.stem(x)
        for layer in self.cnn_layers:
            x = layer(x)
        # Current Shape: (Batch, 512, 8, Time_Reduced)

        # 2. Reshape & Projection (The Bridge)
        x = x.permute(0, 3, 1, 2)  # -> (Batch, Time, Channel, Freq)
        B, T, C, F = x.size()
        x = x.reshape(B, T, C * F)  # -> (Batch, Time, 4096)

        x = self.projection(x)  # -> (Batch, Time, 1024)

        # 3. RNN Modeling
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # -> (Batch, Time, Hidden*2)

        # 4. Attention Mechanism
        if self.config.use_attention:
            x = self.attention(x)  # -> (Batch, Time, Hidden*2)

        # 5. Global Pooling
        x = torch.mean(x, dim=1)  # -> (Batch, Hidden*2)

        # 6. Classification
        x = self.fc(x)  # -> (Batch, Num_Classes)

        return x

