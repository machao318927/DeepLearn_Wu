# 📦 post_retrieval/context_reranking.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import pipeline

class ContextReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = pipeline("text-classification", model=model_name)

    def rerank(self, query, contexts, top_k=3):
        # 每个 context 都与 query 拼接成一个 pair 进行评分
        inputs = [f"{query} [SEP] {ctx}" for ctx in contexts]
        scores = self.reranker(inputs, truncation=True)
        # 取出得分最高的 top_k 个 context
        ranked = sorted(zip(contexts, scores), key=lambda x: x[1]['score'], reverse=True)
        return [ctx for ctx, _ in ranked[:top_k]]


# ✅ 测试用例

def test_context_reranker():
    print("[TEST] 正在测试 ContextReranker")

    query = "RAG 框架的核心优势是什么？"
    contexts = [
        "RAG 能结合检索模块与生成模型。",
        "稠密检索基于向量空间建模。",
        "RAG 支持多跳问答，处理复杂查询。"
    ]

    reranker = ContextReranker()
    top_contexts = reranker.rerank(query, contexts, top_k=2)

    assert isinstance(top_contexts, list) and len(top_contexts) == 2
    print("[✅] ContextReranker 测试通过")
    print("Top reranked results:", top_contexts)


if __name__ == '__main__':
    test_context_reranker()
