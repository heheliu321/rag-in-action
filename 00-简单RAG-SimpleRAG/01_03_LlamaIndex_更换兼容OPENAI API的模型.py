import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI  # 导入 OpenAI LLM 类
from llama_index.embeddings.openai import OpenAIEmbedding # 导入 OpenAI Embedding 类

# --- 开始配置你的自定义 API 地址和密钥 ---
# 请将下面的 placeholder 替换为你的实际 API Base URL 和 API Key
custom_api_base_url = "https://vip.apiyi.com/v1"  
custom_api_key = "XXX"            # 例如: "sk-yourkeyvalue"

# (可选) 确认你的第三方 API 支持并需要使用的模型名称

# OpenAI 默认模型:
llm_model_name = "gpt-4" # 或者你的 API 支持的其他聊天模型
embedding_model_name = "text-embedding-ada-002" # 或者你的 API 支持的其他嵌入模型

# 通过代码直接配置 (推荐，更清晰)
# 配置全局的 LLM (用于问答生成)
Settings.llm = OpenAI(
    model=llm_model_name,
    api_key=custom_api_key,
    api_base=custom_api_base_url,
    # 如果你的 API 端点有其他需要传递的参数，可以在这里添加
    # 例如: temperature=0.7
)

# 配置全局的 Embedding Model (用于文本向量化)
Settings.embed_model = OpenAIEmbedding(
    model=embedding_model_name,
    api_key=custom_api_key,
    api_base=custom_api_base_url,
    # 有些 embedding 端点可能也接受额外参数
)

# --- 配置结束 ---

# 第一行代码：导入相关的库 (部分已在上方导入)
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader (已导入)

# 第二行代码：加载数据
# 确保文件路径 "/Users/niumingjie.nmj/github/rag-in-action/90-文档-Data/黑悟空/设定.txt" 是正确的，并且程序有权限读取
try:
    documents = SimpleDirectoryReader(input_files=["/Users/niumingjie.nmj/github/rag-in-action/90-文档-Data/黑悟空/设定.txt"]).load_data()
except Exception as e:
    print(f"加载文档时出错: {e}")
    print("请检查文件路径和权限。")
    exit()

# 第三行代码：构建索引
# 由于我们已经通过 Settings 配置了全局的 llm 和 embed_model,
# VectorStoreIndex.from_documents() 会自动使用它们。
# 注意: 构建索引主要使用 embedding_model。
print("正在构建索引...")
index = VectorStoreIndex.from_documents(documents)
print("索引构建完成。")

# 第四行代码：创建问答引擎
# as_query_engine() 会自动使用 Settings 中配置的 llm (以及 embedding_model 用于对查询进行编码)。
query_engine = index.as_query_engine()
print("问答引擎已创建。")

# 第五行代码: 开始问答
question = "黑神话悟空中有哪些战斗工具?"
print(f"\n正在查询: {question}")
response = query_engine.query(question)
print("\n回答:")
print(response)