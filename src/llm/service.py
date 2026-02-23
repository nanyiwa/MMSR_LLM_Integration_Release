import os
from openai import OpenAI

# ==========================================
# 1. 配置区
# ==========================================
API_KEY = "sk-260f27c9d9ed40618b568671d158854a"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ==========================================
# 2. 系统背景知识
# ==========================================

BASE_CONTEXT_KNOWLEDGE = """
【系统感知数据说明】
你将看到用户的情感标签和多模态概率数据。在理解用户当前状态时，请遵循以下**感知优先级**：

1. ** 视觉模态 (高信度)**：视觉模型通常很准确，请重点参考面部表情。
2. ** 实际文本内容 (核心)**：请直接通过阅读用户说的话来感受其情绪，这比模型分数更直观。
3. ** 音频模型 (低信度)**：这是自研模型，准确率可能不稳定，其生成的7情感标签仅供参考，请降低依赖。
4. ** 文本模型 (低信度)**：这是自研微调模型，准确率可能不稳定，其生成的7情感标签仅供参考，请降低依赖。

请综合上述信息，结合已给定的【最终状态标签】，进行自然的对话交互。
"""


class LLMService:
    def __init__(self):
        # 初始化
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        print(f" [LLM] DeepSeek 连接器已就绪")

    def chat(self, messages, temperature=0.7):
        """
        通用对话接口
        :param messages: 完整的对话历史列表 [{"role": "system",...}, {"role": "user",...}]
        :param temperature: 随机度 (0.3=严谨, 0.7=创意)
        :return: AI 的回复字符串
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"DeepSeek API 请求失败: {str(e)}"
            print(f"❌ {error_msg}")
            return "（AI 连接中断，请检查网络或 Key）"


# --- 简单测试 ---
if __name__ == "__main__":
    llm = LLMService()

    test_msgs = [
        {"role": "system", "content": "你是一个测试助手。" + BASE_CONTEXT_KNOWLEDGE},
        {"role": "user", "content": "你好，能听到我说话吗？"}
    ]

    print("正在测试 API")
    print(llm.chat(test_msgs))