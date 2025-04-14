# 📦 retrieval/dense_retriever.py

import faiss
import numpy as np
import os
import sys

# 添加项目根路径以便进行跨目录导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'indexing')))

from indexing.index_saver_loader import IndexIOHandler
from indexing.embedding_generator import EmbeddingGenerator
from indexing.index_builder import IndexBuilder

class DenseRetriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.encoder = EmbeddingGenerator(embedding_model)
        self.index = None
        self.texts = []

    def load_index(self, index_path, texts_path):
        self.index, self.texts = IndexIOHandler.load_index(index_path, texts_path)

    def build_index(self, texts):
        embeddings = self.encoder.encode(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.texts = texts

    def retrieve(self, query, top_k=3):
        if self.index is None:
            raise ValueError("索引未加载或构建，请先调用 load_index 或 build_index")
        query_vec = self.encoder.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        return [self.texts[i] for i in indices[0]]

    def get_index_and_texts(self):
        return self.index, self.texts


# ✅ 测试用例

def test_dense_retriever():
    print("[TEST] 正在测试 DenseRetriever")

    texts = ["RAG 是检索增强生成的代表方法。", "Dense Retriever 使用向量计算相似度。", "它适用于复杂的 QA 场景。"]
    retriever = DenseRetriever("all-MiniLM-L6-v2")
    retriever.build_index(texts)
    result = retriever.retrieve("什么是 RAG 方法？")

    assert isinstance(result, list) and len(result) > 0, "[❌] 检索失败"
    print("[✅] DenseRetriever 测试通过")
    print("检索结果:", result)


if __name__ == '__main__':
    test_dense_retriever()
