# 📦 retrieval/test_retrieval.py

import os
import sys

# 设置根路径以便进行模块导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever_router import RetrieverRouter


def test_dense():
    print("\n[TEST] DenseRetriever")
    retriever = DenseRetriever("all-MiniLM-L6-v2")
    texts = [
        "什么是 RAG？",
        "RAG 是结合检索和生成的技术。",
        "它适合构建问答系统。"
    ]
    retriever.build_index(texts)
    result = retriever.retrieve("RAG 是什么？", top_k=3)
    print("结果:", result)


def test_sparse():
    print("\n[TEST] SparseRetriever")
    retriever = SparseRetriever()
    texts = [
        "RAG 框架结合了检索和生成",
        "稀疏检索算法如 BM25 非常常见",
        "它不依赖向量，而是基于词项匹配"
    ]
    retriever.build_index(texts)
    result = retriever.retrieve("什么是稀疏检索？", top_k=2)
    print("结果:", result)


def test_hybrid():
    print("\n[TEST] HybridRetriever")
    texts = [
        "RAG 非常适合问答系统",
        "稀疏检索基于 BM25",
        "稠密检索基于向量相似度",
        "RAG 框架结合了检索和生成"
    ]
    dense = DenseRetriever("all-MiniLM-L6-v2")
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense, sparse)
    hybrid.build_dense_index(texts)
    hybrid.build_sparse_index(texts)
    result = hybrid.retrieve("RAG 是什么？", top_k=3)
    print("结果:", result)


def test_router():
    print("\n[TEST] RetrieverRouter")
    texts = [
        "RAG 能结合检索模块与生成模块",
        "DenseRetriever 使用向量表示问题与文本",
        "SparseRetriever 使用 BM25 算法",
        "HybridRetriever 结合了两者优势"
    ]
    router = RetrieverRouter(strategy="hybrid")
    router.build_index(texts)
    result = router.retrieve("RAG 框架的特点是什么？", top_k=3)
    print("结果:", result)


if __name__ == '__main__':
    test_dense()
    test_sparse()
    test_hybrid()
    test_router()