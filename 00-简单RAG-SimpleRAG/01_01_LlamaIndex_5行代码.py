"""
注意：运行此代码前，请确保已在环境变量中设置OpenAI API密钥。
在Linux/Mac系统中，可以通过以下命令设置：
export OPENAI_API_KEY='your-api-key'

在Windows系统中，可以通过以下命令设置：
set OPENAI_API_KEY=your-api-key

如果无法取得OpenAI API密钥，也没关系，我们有平替方案，请移步至其它程序。
"""
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
import os
import torchvision
print(torchvision.__version__)
import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # 应返回 True


# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["C:\github\liuhehe-rag\\rag-in-action\90-文档-Data\黑悟空\设定.txt"]).load_data()

# 设置 DashScope LLM（例如 qwen-max）
Settings.llm = DashScope(
    model_name="qwen-max",  # 可选：qwen-plus, qwen-turbo 等
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
)

# 设置 Qwen3 为默认 Embedding 模型
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",  # 可根据 DashScope 支持的模型名调整
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(documents)
# 第四行代码：创建问答引擎

# 创建 Deepseek LLM（通过API调用最新的DeepSeek大模型）
myllm = DeepSeek(
    model="deepseek-r1", # 使用最新的推理模型R1
    api_key='sk-71efd8a95f9d43b6a03f35abd074fee6',  # 从环境变量获取API key
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

query_engine = index.as_query_engine()
# 第五行代码: 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))
