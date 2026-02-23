import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
import json
import re
import torch
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from deep_translator import GoogleTranslator

logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# 1. 路径与环境配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

UPLOAD_DIR = os.path.join(current_dir, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("[Backend] 正在导入核心模块...")
try:
    from main_core import MultimodalSystem
    from src.llm.service import LLMService
    from src.utils.prompts import get_system_prompt, PERSONA_CONFIG, GLOBAL_INSTRUCTIONS
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 2. 初始化 FastAPI 应用
# ==========================================
app = FastAPI(title="EmoChat Pro API", description="多模态情感交互后端接口")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

core_system = None
llm_service = None
stt_pipe = None


@app.on_event("startup")
async def startup_event():
    global core_system, llm_service, stt_pipe
    print("\n[Backend] 服务器启动中，正在加载模型...")

    core_system = MultimodalSystem()
    llm_service = LLMService()

    print("[System] 加载 Whisper STT...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
    except Exception as e:
        print(f"⚠️ Whisper 降级为 CPU: {e}")
        stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")

    print("✅ [Backend] 服务已就绪！\n")


# ==========================================
# 3. 辅助函数
# ==========================================
def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fa5]', text))


def translate_to_english(text):
    try:
        if not text or not text.strip(): return ""
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        # print(f"🔤 [Translation] {text} -> {translated}")
        return translated
    except Exception:
        return text


# ==========================================
# 4. API 接口
# ==========================================

@app.get("/")
async def root():
    return {"message": "EmoChat Pro Backend is Running!"}


# --- 接口 A: 生成开场白 ---
class GreetingRequest(BaseModel):
    mode: str
    custom_role: str = ""


@app.post("/api/greeting")
async def generate_greeting(request: GreetingRequest):
    try:
        if request.mode == "自定义" or request.mode == "自定义智能体":
            role_def = f"你扮演：{request.custom_role}" if request.custom_role else "你是一个助手"
            intro_sys_prompt = f"{role_def}。\n{GLOBAL_INSTRUCTIONS}\n请根据你的人设，向用户做个自我介绍并开启话题。"
        else:
            config = PERSONA_CONFIG.get(request.mode, PERSONA_CONFIG["日常闲聊"])
            intro_sys_prompt = f"{config['role_def']}\n{GLOBAL_INSTRUCTIONS}\n请根据你的人设，用最符合你风格的方式向用户打招呼并开启话题。"

        messages = [{"role": "system", "content": intro_sys_prompt}]
        reply = llm_service.chat(messages, temperature=0.8)

        return {"status": "success", "reply": reply}
    except Exception as e:
        print(f"❌ [Error] Greeting: {e}")
        return {"status": "error", "message": str(e)}


# --- 接口 B: 纯语音转文字 (STT) ---
@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        audio_path = os.path.join(UPLOAD_DIR, f"temp_stt_{audio.filename}")
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        stt_res = stt_pipe(audio_path, generate_kwargs={"language": "chinese"})

        return {"status": "success", "text": stt_res["text"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- 接口 C: 全模态分析 (接收前端传来的最佳照片) ---
@app.post("/api/analyze")
async def analyze_sentiment(
        audio: UploadFile = File(...),
        image: UploadFile = File(None),
        text: str = Form(""),
        mode: str = Form("日常闲聊"),
        custom_role: str = Form(""),
        history: str = Form("[]")
):
    try:
        # 1. 保存音频
        audio_path = os.path.join(UPLOAD_DIR, f"temp_{audio.filename}")
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # 2. 保存图片
        image_path = None
        if image:
            image_path = os.path.join(UPLOAD_DIR, f"temp_{image.filename}")
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        # 3. 核心分析
        text_for_model = text
        if contains_chinese(text):
            text_for_model = translate_to_english(text)

        analysis_res = core_system.analyze(
            audio_path=audio_path,
            image_path=image_path,
            text_content=text_for_model
        )

        final_emotion = analysis_res['final_decision'].upper()
        fused_scores = analysis_res['details']['fused_scores']
        confidence = max(fused_scores.values()) if fused_scores else 0.0

        # 4. LLM 生成
        sys_prompt = get_system_prompt(mode, final_emotion, custom_role)
        try:
            client_history = json.loads(history)
        except:
            client_history = []

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(client_history)
        messages.append({"role": "user", "content": text})

        ai_reply = llm_service.chat(messages)

        return {
            "status": "success",
            "data": {
                "text": text,
                "emotion": final_emotion,
                "confidence": confidence,
                "reply": ai_reply,
                "scores": fused_scores,
                "vision_score": analysis_res['details']['vision'],
                "audio_score": analysis_res['details']['audio'],
                "text_score": analysis_res['details']['text']
            }
        }

    except Exception as e:
        print(f"❌ [Error] Analyze: {e}")
        return {"status": "error", "message": str(e)}


# --- 接口 D: 实时视觉流  ---
import cv2
import numpy as np


@app.post("/api/live_vision")
async def live_vision_analysis(image: UploadFile = File(...)):
    """
    不做日志输出以防刷屏
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = core_system.vision_engine.detect_face_and_emotion(img)

        if result:
            return {"status": "success", "data": result}
        else:
            return {"status": "empty", "message": "No face"}

    except Exception:
        return {"status": "error"}


if __name__ == "__main__":
    # log_level="warning"减少日志
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")