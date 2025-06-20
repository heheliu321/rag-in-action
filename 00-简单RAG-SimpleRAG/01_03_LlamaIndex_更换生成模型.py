# 导入相关的库
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope

# https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
# Settings.llm = DeepSeek(model="deepseek-chat")
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载环境变量
import os

# 设置 DashScope LLM（例如 qwen-max）
Settings.llm = DashScope(
    model_name="deepseek-r1",
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6",
    api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    is_chat_model=True  # 开启推理模式
)

# 设置 Qwen3 为默认 Embedding 模型
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",  # 可根据 DashScope 支持的模型名调整
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

# 加载数据
documents = SimpleDirectoryReader(input_files=["C:\github\liuhehe-rag\\rag-in-action\90-文档-Data\黑悟空\设定.txt"]).load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建问答引擎
query_engine = index.as_query_engine()

# 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))