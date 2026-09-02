"""模型配置：根据 .env 里的 Key 自动选择模型服务，供所有 demo 共用。"""
import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel

load_dotenv()


def build_model(temperature: float = 0.0):
    """优先 DeepSeek，其次阿里 Qwen，都没有则给出友好报错。

    temperature：采样温度，自一致性等多路采样时传 >0 以得到不同答案。
    """
    if os.getenv("DEEPSEEK_API_KEY"):
        return OpenAIServerModel(
            model_id=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_base="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature,
        )
    if os.getenv("QWEN_API_KEY"):
        return OpenAIServerModel(
            model_id=os.getenv("QWEN_MODEL", "qwen-plus"),
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("QWEN_API_KEY"),
            temperature=temperature,
        )
    raise SystemExit(
        "未检测到可用的 API Key。\n"
        "请复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY 或 QWEN_API_KEY 后重试。"
    )


def build_openai_client():
    """返回一个 OpenAI 兼容的原始客户端，用于直接 LLM 调用（重排/投票/反思等）。

    优先 DeepSeek，其次 Qwen，均需 OpenAI 兼容接口。
    """
    from openai import OpenAI

    if os.getenv("DEEPSEEK_API_KEY"):
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
    if os.getenv("QWEN_API_KEY"):
        return OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    raise SystemExit("未检测到可用的 API Key，无法直接调用 LLM。")


def get_model_id() -> str:
    """返回当前使用模型的 model_id（与 build_model 一致）。"""
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if os.getenv("QWEN_API_KEY"):
        return os.getenv("QWEN_MODEL", "qwen-plus")
    raise SystemExit("未检测到可用的 API Key。")
