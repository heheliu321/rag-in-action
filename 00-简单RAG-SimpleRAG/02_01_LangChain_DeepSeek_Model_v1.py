# 1. 加载文档
import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader, TextLoader  # pip install langchain-community

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
# }
#
#
# loader = WebBaseLoader(
#     web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",),
#     header_template=headers  # 添加 User-Agent
# )
# 假设你已将网页保存为本地文件
loader = TextLoader("/Users/niumingjie.nmj/github/rag-in-action//Users/niumingjie.nmj/github/rag-in-action/90-文档-Data/黑悟空/黑悟空wiki.txt", encoding='utf-8')
docs = loader.load()

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface

# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 可根据 DashScope 支持的模型名调整
    dashscope_api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 构建用户查询
question = "黑悟空有哪些游戏场景？"

# 6. 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=10)
docs_content = "\n---------\n".join(doc.page_content for doc in retrieved_docs)

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )

# 8. 使用大语言模型生成答案
from langchain_deepseek import ChatDeepSeek  # pip install langchain-deepseek

# llm = ChatDeepSeek(
#     model="deepseek-chat",  # DeepSeek API 支持的模型名称
#     temperature=0.7,        # 控制输出的随机性
#     max_tokens=2048,        # 最大输出长度
#     api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
# )

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

llm = ChatTongyi(
    model_name="qwen-max",
    dashscope_api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
)

print(docs_content)
# answer = llm.invoke(prompt.format(question=question, context=docs_content))
prompt = """基于以下上下文，回答问题。如果上下文中没有相关信息，
            请说"我无法从提供的上下文中找到相关信息"。
            上下文: {context}
            问题: {question}
            回答:"""
prompt = (prompt.replace("{context}", docs_content).
          replace("{question}", "黑悟空有哪些游戏场景？"))
message = HumanMessage(content=prompt)
answer = llm.invoke([message])

print(answer)
