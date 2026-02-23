import os
from dataclasses import dataclass, field
from typing import List

# ==========================================
# 1. 情感标签 (7分类)
# ==========================================
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
LABEL_TO_ID = {label: i for i, label in enumerate(EMOTION_LABELS)}


# ==========================================
# 2. 音频特征参数
# ==========================================
@dataclass
class AudioConfig:
    duration: int = 3
    sample_rate: int = 22050
    n_mels: int = 128
    max_time_steps: int = 130
    n_fft: int = 2048
    hop_length: int = 512


# ==========================================
# 3. 模型架构参数
# ==========================================
@dataclass
class ModelConfig:
    input_channels: int = 1
    # 默认使用 Wider 架构 (512 filters)
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    use_cbam: bool = True
    cnn_dropout: float = 0.3

    # RNN 部分
    rnn_type: str = 'LSTM'
    rnn_hidden_size: int = 128
    rnn_layers: int = 2
    bidirectional: bool = True
    rnn_dropout: float = 0.3

    # Attention 部分
    use_attention: bool = True
    num_heads: int = 4
    attn_dropout: float = 0.3

    # Classifier
    num_classes: int = 7
    fc_dropout: float = 0.5