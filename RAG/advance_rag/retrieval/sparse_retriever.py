# 📦 retrieval/sparse_retriever.py

import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'indexing')))

from indexing.index_saver_loader import IndexIOHandler

class SparseRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.texts = []
        self.tfidf_matrix = None

    def build_index(self, texts):
        self.texts = texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.texts[i] for i in top_indices]


# ✅ 测试用例
def test_sparse_retriever():
    print("[TEST] 正在测试 SparseRetriever")

    texts = [
        "稀疏检索使用 TF-IDF 或 BM25 等特征。",
        "TF-IDF 是常用的文本表示方法。",
        "RAG 可以结合稀疏和稠密检索。"
    ]

    retriever = SparseRetriever()
    retriever.build_index(texts)
    result = retriever.retrieve("什么是 TF-IDF？")

    assert isinstance(result, list) and len(result) > 0, "[❌] 稀疏检索失败"
    print("[✅] SparseRetriever 测试通过")
    print("检索结果:", result)


if __name__ == '__main__':
    test_sparse_retriever()
