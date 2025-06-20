# 1. 加载文档
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings

loader = TextLoader("/Users/niumingjie.nmj/github/rag-in-action//Users/niumingjie.nmj/github/rag-in-action/90-文档-Data/黑悟空/黑悟空wiki.txt", encoding='utf-8')
docs = loader.load()

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 可根据 DashScope 支持的模型名调整
    dashscope_api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
)

# 4. 创建向量存储aa
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 定义RAG提示词
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

# 6. 定义应用状态
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# 7. 定义检索步骤
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# 8. 定义生成步骤
from langchain_community.chat_models.tongyi import ChatTongyi


def generate(state: State):
    llm = ChatTongyi(
        model_name="qwen-max",
        dashscope_api_key="sk-71efd8a95f9d43b6a03f35abd074fee6"
    )
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# 9. 构建和编译应用
from langgraph.graph import START, StateGraph  # pip install langgraph

graph = (
    StateGraph(State)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)
print(type(graph))

# 10. 运行查询
question = "黑悟空有哪些游戏场景？"
response = graph.invoke({"question": question})
print(f"\n问题: {question}")
print(f"答案: {response['answer']}")
