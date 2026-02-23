import os
import sys
import torch

# 屏蔽警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline


class TextPredictor:
    def __init__(self, model_path=None, device=None):
        """
        初始化文本情感预测器
        :param model_path: 本地模型文件夹路径 (包含 config.json, model.safetensors 等)
        :param device: 运行设备 (CPU=-1, GPU=0)
        """
        # 1. 自动定位权重路径
        if model_path is None:
            # 当前文件: src/models/text/inference.py
            # 目标: weights/text/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            model_path = os.path.join(project_root, "weights", "text")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ 文本模型未找到: {model_path}\n")

        # 2. 确定设备
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        print(f"[Text] 加载 BERT 模型 (Device: {self.device})...")

        # 3. 加载 Pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                top_k=None,
                device=self.device
            )
            print("✅ 文本模型加载成功")
        except Exception as e:
            print(f"❌ 文本模型加载失败: {e}")
            raise e

    def predict(self, text):
        """
        执行预测
        :param text: 输入字符串
        :return: {'happy': 0.85, 'sad': 0.05, ...} (按概率降序)
        """
        if not text or not isinstance(text, str):
            return {}

        # 推理
        results = self.classifier(text)[0]

        # 格式化输出
        scores = {item['label'].lower(): item['score'] for item in results}

        # 按概率降序排列
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return sorted_scores


# 自测
if __name__ == "__main__":
    try:
        predictor = TextPredictor()
        text = "I am so happy today!"
        print(f"Input: {text}")
        print(f"Result: {predictor.predict(text)}")
    except Exception as e:
        print(f"Error: {e}")