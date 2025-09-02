from pathlib import Path

from pdf2image import convert_from_path
import base64
import os
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key='sk-71efd8a95f9d43b6a03f35abd074fee6',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
output_dir = "temp_images"

# 1. PDF 转图片
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
current_dir = Path(__file__).resolve().parent.parent.parent
images = convert_from_path(f"{current_dir}/90-文档-Data/黑悟空/黑神话悟空.pdf")
image_paths = []
for i, image in enumerate(images):
    image_path = os.path.join(output_dir, f'page_{i+1}.jpg')
    image.save(image_path, 'JPEG')
    image_paths.append(image_path)
print(f"成功转换 {len(image_paths)} 页")


# 2. GPT-4o 分析图片
print("\n开始分析图片...")
results = []
for image_path in image_paths:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                        },
                    },
                    {"type": "text", "text": "图中描绘的是什么景象?"},
                ],
            },
        ],
    )
results.append(response.choices[0].message.content)


# 3. 转换为 LangChain 的 Document 数据结构
from langchain_core.documents import Document

documents = [
    Document(
        page_content=result,
        metadata={"source": "data/黑悟空/黑神话悟空.pdf", "page_number": i + 1}
    )
    for i, result in enumerate(results)
]

# 输出所有生成的 Document 对象
print("\n分析结果：")
for doc in documents:
    print(f"内容: {doc.page_content}\n元数据: {doc.metadata}\n")
    print("-" * 80)

# 清理临时文件
for image_path in image_paths:
    os.remove(image_path)
os.rmdir(output_dir)

