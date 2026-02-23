import os
import sys
import numpy as np
import librosa
import torch
import torch.nn.functional as F

# ==========================================
# 1. 路径修复与环境设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from config import AudioConfig, ModelConfig, EMOTION_LABELS
    from .models.acrnn_v3 import ACRNNv3
except ImportError as e:
    print(f"❌ [Audio] 导入模块失败: {e}")
    from src.modules.audio.config import AudioConfig, ModelConfig, EMOTION_LABELS
    from src.modules.audio.models.acrnn_v3 import ACRNNv3


class EmotionPredictor:
    def __init__(self, model_path=None, device=None):
        """
        初始化音频情感预测器
        :param model_path: 权重文件路径 (默认自动查找 weights/audio/...)
        :param device: 'cuda' or 'cpu'
        """
        # 1. 配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. 智能路径查找
        if model_path is None:

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            default_weights = os.path.join(project_root, "weights", "audio", "best_acrnn_v3_1.pth")

            if os.path.exists(default_weights):
                model_path = default_weights
            else:
                # 备用方案
                local_weights = os.path.join(current_dir, "best_acrnn_v3_1.pth")
                if os.path.exists(local_weights):
                    model_path = local_weights
                else:
                    raise FileNotFoundError(
                        f"❌ 无法自动定位权重文件！\n"
                        f"请检查路径: {default_weights}\n"
                        f"或者在初始化时手动传入 model_path 参数。"
                    )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到指定的权重文件: {model_path}")

        print(f"🧠 [Audio] 加载模型权重: {os.path.basename(model_path)}")

        # 3. 加载配置与模型架构
        self.audio_cfg = AudioConfig()
        self.model_cfg = ModelConfig()

        self.model = ACRNNv3(config=self.model_cfg).to(self.device)

        # 4. 加载权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            # print("✅ 音频模型加载就绪")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e

    def _preprocess_single(self, audio_path):
        try:
            # A. Load & Resample
            y, sr = librosa.load(audio_path, sr=self.audio_cfg.sample_rate)

            # B. Trim Silence
            y, _ = librosa.effects.trim(y)

            # C. Pad/Truncate
            target_len = int(self.audio_cfg.sample_rate * self.audio_cfg.duration)
            if len(y) > target_len:
                y = y[:target_len]
            else:
                padding = target_len - len(y)
                y = np.pad(y, (0, padding), mode='constant')

            # D. Log-Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_mels=self.audio_cfg.n_mels,
                n_fft=self.audio_cfg.n_fft,
                hop_length=self.audio_cfg.hop_length
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # E. Fix Time Steps (Width=130)
            target_width = self.audio_cfg.max_time_steps
            if log_mel_spec.shape[1] > target_width:
                log_mel_spec = log_mel_spec[:, :target_width]
            else:
                padding = target_width - log_mel_spec.shape[1]
                log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, padding)), mode='constant')

            # F. To Tensor (Batch, Channel, Freq, Time)
            spec_tensor = torch.tensor(log_mel_spec, dtype=torch.float32)
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 130)
            return spec_tensor.to(self.device)

        except Exception as e:
            print(f"❌ 音频预处理错误: {e}")
            return None

    def predict(self, audio_path):
        """
        :return: {'happy': 0.85, 'sad': 0.05, ...} (按概率降序)
        """
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}

        # 1. 预处理
        tensor = self._preprocess_single(audio_path)
        if tensor is None:
            return {"error": "Audio processing failed"}

        # 2. 推理
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 3. 格式化结果
        result = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}

        # 4. 概率降序排列
        sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return sorted_result


# --- 独立测试 ---
if __name__ == "__main__":
    print(" [Audio Module] 独立测试")
    try:
        predictor = EmotionPredictor()
        print("✅ 初始化成功")

        # 交互测试
        while True:
            path = input("\n🔊 输入音频文件路径 (输入 q 退出): ").strip().strip('"')
            if path.lower() == 'q': break

            res = predictor.predict(path)
            print("预测结果:")
            print(res)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
