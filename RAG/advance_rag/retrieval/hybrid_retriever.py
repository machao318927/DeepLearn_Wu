# 📦 retrieval/hybrid_retriever.py

import os
import sys

# 添加项目根路径以便进行跨目录导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever

    def build_dense_index(self, texts):
        self.dense.build_index(texts)

    def build_sparse_index(self, texts):
        # 添加稀疏检索构建函数的兼容性调用（若不存在该方法，则跳过）
        if hasattr(self.sparse, "build_index"):
            self.sparse.build_index(texts)
        elif hasattr(self.sparse, "build_sparse_index"):
            self.sparse.build_sparse_index(texts)
        else:
            raise AttributeError("SparseRetriever 缺少 build_index 或 build_sparse_index 方法")

    def retrieve(self, query, top_k=5):
        dense_results = self.dense.retrieve(query, top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k * 2)

        # 合并结果并去重
        merged = list(dict.fromkeys(dense_results + sparse_results))
        return merged[:top_k]


# ✅ 测试用例

def test_hybrid_retriever():
    print("[TEST] 正在测试 HybridRetriever")

    texts = [
        "RAG 框架结合了检索和生成",
        "稠密检索使用向量匹配",
        "稀疏检索使用词项匹配，例如 BM25",
        "RAG 可用于多跳问答任务"
    ]

    dense = DenseRetriever("all-MiniLM-L6-v2")
    sparse = SparseRetriever()

    retriever = HybridRetriever(dense, sparse)
    retriever.build_dense_index(texts)
    retriever.build_sparse_index(texts)

    query = "RAG 是什么？"
    result = retriever.retrieve(query, top_k=3)

    assert isinstance(result, list) and len(result) > 0, "[❌] 检索失败"
    print("[✅] HybridRetriever 测试通过")
    print("检索结果:", result)


if __name__ == '__main__':
    test_hybrid_retriever()