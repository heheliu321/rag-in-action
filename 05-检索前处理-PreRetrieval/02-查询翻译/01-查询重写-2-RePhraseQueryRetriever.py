import logging
from pathlib import Path

from langchain.retrievers import RePhraseQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
# 设置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
# 加载游戏文档数据
current_dir = Path(__file__).resolve().parent.parent.parent
loader = TextLoader(f"{current_dir}/90-文档-Data/黑悟空/设定.txt", encoding='utf-8')
data = loader.load()
# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
# 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=all_splits, embedding= embed_model)
# 设置RePhraseQueryRetriever
llm = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
)
retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm # 使用DeepSeek模型做重写器
)
# 示例输入：游戏相关查询
query = "那个，我刚开始玩这个游戏，感觉很难，在普陀山那一关，嗯，怎么也过不去。先学什么技能比较好？新手求指导！"
# 调用RePhraseQueryRetriever进行查询重写
docs = retriever_from_llm.invoke(query)
print(docs)