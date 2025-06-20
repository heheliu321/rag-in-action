# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

Settings.llm = DashScope(
    model_name="deepseek-r1",  # 可选：qwen-plus, qwen-turbo 等
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
)

# 设置 Qwen3 为默认 Embedding 模型
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",  # 可根据 DashScope 支持的模型名调整
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=[r"C:\github\liuhehe-rag\rag-in-action\90-文档-Data\黑悟空\设定.txt"]).load_data()

# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(
    documents,
    # embed_model=embed_model
)

# 第四行代码：创建问答引擎
query_engine = index.as_query_engine(
    # llm=llm
)

# 第五行代码: 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))
