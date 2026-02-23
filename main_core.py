import os
import sys
import time
import json
import torch

# 确保项目根目录在系统路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. 导入三大模态预测器
from src.modules.audio.inference import EmotionPredictor as AudioModel
from src.modules.text.inference import TextPredictor as TextModel
from src.modules.vision.inference import VisionPredictor as VisionModel
from src.fusion.weighted_fusion import WeightedFusion


class MultimodalSystem:
    def __init__(self):
        print("\n🚀 [System] 正在初始化多模态情感识别核心 (纯净版)...")
        start_time = time.time()

        # --- 加载模态模型 ---
        try:
            self.audio_engine = AudioModel()
        except Exception as e:
            print(f"⚠️ 音频模块加载失败: {e}")
            self.audio_engine = None

        try:
            self.text_engine = TextModel()
        except Exception as e:
            print(f"⚠️ 文本模块加载失败: {e}")
            self.text_engine = None

        try:
            self.vision_engine = VisionModel()
        except Exception as e:
            print(f"⚠️ 视觉模块加载失败: {e}")
            self.vision_engine = None

        self.fusion_engine = WeightedFusion()

        print(f"✅ 系统初始化完成! 耗时: {time.time() - start_time:.2f}s\n")

    def analyze(self, audio_path=None, text_content=None, image_path=None):
        """
        全流程分析: 单模态 -> 加权融合 (不再经过 LLM 判决)
        """
        # --- Step 1: 单模态推理 ---
        audio_res = {}
        text_res = {}
        vision_res = {}

        if self.audio_engine and audio_path:
            audio_res = self.audio_engine.predict(audio_path)
            if "error" in audio_res: audio_res = {}

        if self.text_engine and text_content:
            text_res = self.text_engine.predict(text_content)

        if self.vision_engine and image_path:
            vision_res = self.vision_engine.predict(image_path)
            if "error" in vision_res: vision_res = {}

        # --- Step 2: 加权融合 ---
        # 你的权重策略
        weights = {'audio': 0.3, 'text': 0.2, 'vision': 0.5}
        fused_result = self.fusion_engine.fuse(audio_res, text_res, vision_res, weights)

        # 确保 fused_result 是按分数从高到低排序的
        # 这样我们可以直接取第一个作为最终结果
        if fused_result:
            fused_result = dict(sorted(fused_result.items(), key=lambda x: x[1], reverse=True))
            # 【关键修改】最终决定直接取融合分数的 No.1
            top_emotion = list(fused_result.keys())[0]
        else:
            top_emotion = "neutral"

        # --- Step 3: 构造返回包 ---
        response = {
            "final_decision": top_emotion,
            "details": {
                "fused_scores": fused_result,
                "audio": audio_res,
                "text": text_res,
                "vision": vision_res
            }
        }
        return response

if __name__ == "__main__":
    # 简单测试
    core = MultimodalSystem()
    print("核心已启动，等待调用...")