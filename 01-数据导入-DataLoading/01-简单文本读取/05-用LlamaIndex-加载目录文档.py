from pathlib import Path

from llama_index.core import SimpleDirectoryReader
# 使用 SimpleDirectoryReader 加载目录中的文件
current_dir = Path(__file__).resolve().parent.parent.parent
dir_reader = SimpleDirectoryReader(f"{current_dir}/90-文档-Data/黑悟空")
documents = dir_reader.load_data()
# 查看加载的文档数量和内容
print(f"文档数量: {len(documents)}")
print(documents[0].text[:100])  # 打印第一个文档的前100个字符

# 仅加载某一个特定文件
dir_reader = SimpleDirectoryReader(input_files=[f'{current_dir}/90-文档-Data/黑悟空/设定.txt'])
documents = dir_reader.load_data()
print(f"文档数量: {len(documents)}")
print(documents[0].text[:100])  # 打印第一个文档的前100个字符
