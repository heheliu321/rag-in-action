# 需要LLAMA_CLOUD_API_KEY
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()   

# LlamaParse PDF reader for PDF Parsing
from llama_parse import LlamaParse
current_dir = Path(__file__).resolve().parent.parent.parent
file_path = f"{current_dir}/90-文档-Data/黑悟空/黑神话悟空.pdf"
documents = LlamaParse(result_type="markdown").load_data(
   file_path
)
print(documents)

from llama_index.core.node_parser import MarkdownElementNodeParser
node_parser = MarkdownElementNodeParser()
nodes = node_parser.get_nodes_from_documents(documents)

print(nodes)

