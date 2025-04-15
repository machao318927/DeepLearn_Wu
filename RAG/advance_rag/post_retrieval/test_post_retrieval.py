# 📦 post_retrieval/test_post_retrieval.py

import os
import sys

# 添加项目根路径以便进行跨目录导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from post_retrieval.context_compression import ContextCompressor
from post_retrieval.context_reranking import ContextReranker
from post_retrieval.context_fusion import ContextFusion
from post_retrieval.context_rewriting import ContextRewriter

def test_context_compression():
    print("[TEST] ContextCompressor")
    docs = [
        "RAG 是一种结合检索与生成的 NLP 框架。",
        "RAG 框架可以有效提升问答系统的性能。",
        "RAG 使用向量检索和生成式模型。"
    ]
    compressor = ContextCompressor()
    compressed = compressor.compress(docs)
    assert isinstance(compressed, str) and len(compressed) > 0
    print("[✅] Compression result:", compressed)

def test_context_reranker():
    print("[TEST] ContextReranker")
    query = "RAG 的用途是什么？"
    docs = [
        "RAG 是一种 NLP 模型架构。",
        "RAG 可提升问答质量。",
        "稀疏检索不适合复杂任务。"
    ]
    reranker = ContextReranker()
    ranked = reranker.rerank(query, docs, top_k=2)
    assert isinstance(ranked, list) and len(ranked) == 2
    print("[✅] Reranking result:", ranked)

def test_context_fuser():
    print("[TEST] ContextFusion")
    chunks = [
        "RAG 框架由 Facebook 提出。",
        "它结合了检索器和生成模型。"
    ]
    fuser = ContextFusion()
    fused = fuser.fuse(chunks)
    assert isinstance(fused, str) and len(fused) > 0
    print("[✅] Fused result:", fused)

def test_context_rewriter():
    print("[TEST] ContextRewriter")
    docs = [
        "rag是检索增强生成模型。",
        "它结合了两个部分：retriever 和 generator。"
    ]
    rewriter = ContextRewriter()
    rewritten = rewriter.rewrite(docs)
    assert isinstance(rewritten, list) and all(isinstance(d, str) and len(d.strip()) > 0 for d in rewritten)
    print("[✅] Rewriting result:", rewritten)

if __name__ == '__main__':
    test_context_compression()
    test_context_reranker()
    test_context_fuser()
    test_context_rewriter()