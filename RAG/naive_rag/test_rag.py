# test_rag.py

from rag_engine import NaiveRAG
from openai_api_wrapper import call_chatgpt
from text_splitter_demo import split_text_into_chunks

if __name__ == "__main__":
    # Step 1: 构建小型知识库（每条为一个“文档块”）
    documents = [
        "猫是夜行动物，具有灵敏的听觉和视觉。",
        "狗被称为人类最好的朋友，通常被训练来看家。",
        "苹果富含维生素C和纤维，有助于健康。",
        "人工智能是指让计算机模拟人类智能的技术。",
        "地球是太阳系中唯一存在生命的行星。"
    ]
    # 切块的做法

    full_text = open("my_knowledge.txt", "r", encoding="utf-8").read()
    chunks = split_text_into_chunks(full_text)

    # Step 2: 初始化 RAG 系统
    rag = NaiveRAG(top_k=2)
    rag.build_index(chunks)

    # Step 3: 用户提问
    query = "猫的特点是什么？"

    # Step 4: 构造 Prompt
    prompt = rag.build_prompt(query)
    print("\n【生成的 Prompt】\n" + "=" * 50 + f"\n{prompt}")

    # Step 5: 调用 ChatGPT 生成回答
    answer = call_chatgpt(prompt, model="gpt-3.5-turbo")
    print("\n【ChatGPT 生成回答】\n" + "=" * 50)
    print(answer)
"""
[INFO] 已成功建立向量索引，文档数: 5，向量维度: 384

【生成的 Prompt】
==================================================
请根据以下内容回答问题：

[文档1]:
狗被称为人类最好的朋友，通常被训练来看家。

[文档2]:
猫是夜行动物，具有灵敏的听觉和视觉。

[用户问题]:
猫的特点是什么？


【ChatGPT 生成回答】
==================================================
猫的特点包括它们是夜行动物，拥有灵敏的听觉和视觉。

"""