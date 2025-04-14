# 📦 retrieval/retriever_router.py

import os
import sys

# 添加根路径，确保可以正确导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid_retriever import HybridRetriever


class RetrieverRouter:
    def __init__(self, strategy="dense", embedding_model="all-MiniLM-L6-v2"):
        self.strategy = strategy.lower()
        self.dense = DenseRetriever(embedding_model)
        self.sparse = SparseRetriever()
        self.hybrid = HybridRetriever(self.dense, self.sparse)

    def build_index(self, texts):
        if self.strategy == "dense":
            self.dense.build_index(texts)
        elif self.strategy == "sparse":
            self.sparse.build_index(texts)
        elif self.strategy == "hybrid":
            self.hybrid.build_dense_index(texts)
            self.hybrid.build_sparse_index(texts)
        else:
            raise ValueError(f"不支持的检索策略: {self.strategy}")

    def retrieve(self, query, top_k=5):
        if self.strategy == "dense":
            return self.dense.retrieve(query, top_k)
        elif self.strategy == "sparse":
            return self.sparse.retrieve(query, top_k)
        elif self.strategy == "hybrid":
            return self.hybrid.retrieve(query, top_k)
        else:
            raise ValueError(f"不支持的检索策略: {self.strategy}")


# ✅ 测试函数
def test_retriever_router():
    print("[TEST] 正在测试 RetrieverRouter")

    texts = [
        "RAG 是检索增强生成技术",
        "稠密检索使用向量匹配技术",
        "稀疏检索使用词项匹配，如 BM25",
        "混合检索结合了稠密与稀疏两种方式",
    ]

    router = RetrieverRouter(strategy="hybrid")
    router.build_index(texts)

    query = "什么是 RAG？"
    results = router.retrieve(query, top_k=3)

    assert isinstance(results, list) and len(results) > 0, "[❌] Router 检索失败"
    print("[✅] RetrieverRouter 测试通过")
    print("检索结果:", results)


if __name__ == "__main__":
    test_retriever_router()
