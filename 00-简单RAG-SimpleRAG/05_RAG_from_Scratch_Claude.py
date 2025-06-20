# 1. 准备文档数据
# 导入所需模块
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# 加载环境变量
load_dotenv()

# 初始化 DashScope 嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 根据实际支持的模型名调整
    dashscope_api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"  # 使用环境变量获取 API Key
)

# 初始化 ChromaDB 客户端并指定嵌入函数
client = PersistentClient(path="./chroma_db")  # 持久化路径

# 创建或获取集合时，直接使用 DashScopeEmbeddings 提供的向量
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=embeddings  # 直接使用 DashScopeEmbeddings 作为嵌入函数
)

# 准备文档数据
docs = [
    "黑神话悟空的战斗如同武侠小说活过来一般，当金箍棒与妖魔碰撞时，火星四溅，招式行云流水。悟空可随心切换狂猛或灵动的战斗风格，一棒横扫千军，或是腾挪如蝴蝶戏花。",
    "72变神通不只是变化形态，更是开启新世界的钥匙。化身飞鼠可以潜入妖魔巢穴打探军情，变作金鱼能够探索深海遗迹的秘密，每一种变化都是一段独特的冒险。",
    # 其他文档...
]

# 写入文档到 ChromaDB 集合
collection.add(
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))]  # 为每个文档分配唯一 ID
)


# 查询
question = "黑神话悟空的战斗系统有什么特点?"
results = collection.query(query_texts=[question], n_results=3)
context = results['documents'][0]

for i, doc in enumerate(context, 1):
    print(f"[{i}] {doc}")

# 5. 构建提示词
prompt = f"""根据以下参考信息回答问题，并给出信息源编号。
如果无法从参考信息中找到答案，请说明无法回答。
参考信息:
{chr(10).join(f"[{i+1}] {doc}" for i, doc in enumerate(context))}
问题: {question}
答案:"""

# 6. 使用Claude生成答案
from anthropic import Anthropic # pip install anthropic
claude = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
response = claude.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": prompt
    }],
    max_tokens=1024
)
print(f"\n生成的答案: {response.content[0].text}")
