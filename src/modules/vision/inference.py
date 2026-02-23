import os
import sys
import logging
import cv2
import numpy as np

# 1. 强制防冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


class VisionPredictor:
    def __init__(self):
        print("👁️ [Vision] 初始化 DeepFace 引擎...")
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            print("✅ 视觉引擎就绪")
        except ImportError:
            print("❌ 错误: 未安装 deepface 库，运行 pip install deepface")
            self.DeepFace = None

    # ==========================================
    # 核心接口 A: 路径分析
    # ==========================================
    def predict(self, img_path):
        """
        接收图片路径，返回情感分数
        被 main_core.py 的 analyze() 方法调用
        """
        if self.DeepFace is None: return {"error": "DeepFace not installed"}

        if img_path is None: return {}
        if not os.path.exists(img_path): return {"error": "File not found"}

        try:
            # 1. 调用 DeepFace 分析
            objs = self.DeepFace.analyze(
                img_path=img_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if not objs: return {"error": "No face detected"}

            # 2. 格式化结果
            result = objs[0]['emotion']
            # 归一化 (0-100 -> 0-1)
            scores = {k.lower(): v / 100.0 for k, v in result.items()}

            # 排序
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            # 出错时不崩，返回空
            return {"error": str(e)}

    # ==========================================
    # 核心接口 B: 实时流检测 (兼容 Live Vision)
    # ==========================================
    def detect_face_and_emotion(self, img_array):
        """
        接收 numpy 数组 (OpenCV 格式)，返回坐标和主导情绪
        被 backend/main.py 的 live_vision_analysis() 调用
        """
        if self.DeepFace is None: return None

        try:
            objs = self.DeepFace.analyze(
                img_path=img_array,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if not objs: return None

            face_data = objs[0]
            return {
                "region": face_data.get('region', {}),  # {x, y, w, h}
                "emotion": face_data.get('dominant_emotion', 'neutral')
            }
        except:
            return None