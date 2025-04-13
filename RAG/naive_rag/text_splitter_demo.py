# 文本切块脚本
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size: int = 100, chunk_overlap: int = 20):
    """
    将文本切成多个chunk，每个chunk长度不超过 chunk_size，保留 chunk_overlap 的重叠部分
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "。", "，", ",", " "]  # 从段落到词，递进切割
    )

    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    # 示例文本
    full_text = """
    猫是一种常见的家养动物，通常具有敏锐的听觉和视觉，尤其在夜间表现出很强的活动能力。
    它们是夜行性动物，常被人们当作宠物饲养。
    与猫相比，狗则更具群体性和服从性，常用于看家护院。
    狗通常通过训练来完成各种任务，例如搜救、导盲等。

    此外，动物与人类的互动在人类社会中发挥着重要作用，不仅提供情感陪伴，还可以协助执行各种任务。
    """

    chunks = split_text_into_chunks(full_text)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk}")

"""
--- Chunk 1 ---
猫是一种常见的家养动物，通常具有敏锐的听觉和视觉，尤其在夜间表现出很强的活动能力。
    它们是夜行性动物，常被人们当作宠物饲养。

--- Chunk 2 ---
与猫相比，狗则更具群体性和服从性，常用于看家护院。
    狗通常通过训练来完成各种任务，例如搜救、导盲等。

--- Chunk 3 ---
此外，动物与人类的互动在人类社会中发挥着重要作用，不仅提供情感陪伴，还可以协助执行各种任务。
"""
