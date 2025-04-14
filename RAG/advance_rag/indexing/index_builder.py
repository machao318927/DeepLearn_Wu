# 🔧 indexing/index_builder.py

import faiss
import os
import pickle
from embedding_generator import EmbeddingGenerator

class IndexBuilder:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
        self.index = None
        self.texts = []

    def build_index(self, texts):
        """
        根据输入文本生成向量并构建Faiss索引。
        """
        self.texts = texts
        embeddings = self.embedding_generator.encode(texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        """
        对query进行编码，并从索引中检索前top_k个最相似的文本。
        """
        query_embedding = self.embedding_generator.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in indices[0]]

    def save(self, index_path, texts_path):
        """
        保存索引和对应文本。
        """
        faiss.write_index(self.index, index_path)
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self, index_path, texts_path):
        """
        加载已保存的索引和文本。
        """
        self.index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)


# 🧪 测试用例

def test_index_builder_full_pipeline():
    print("\n[TEST] IndexBuilder 模块测试开始...")

    # 示例文本
    texts = [
        "猫是一种夜行性动物，听觉和视觉都很敏锐。",
        "狗通常比较忠诚，适合看家护院。",
        "大象是一种大型哺乳动物，生活在热带地区。"
    ]

    # 构建索引
    embedder = EmbeddingGenerator()
    builder = IndexBuilder(embedder)
    builder.build_index(texts)

    # 保存并重新加载
    index_path = "test.index"
    texts_path = "test_texts.pkl"
    builder.save(index_path, texts_path)

    # 重新载入验证
    new_builder = IndexBuilder(embedder)
    new_builder.load(index_path, texts_path)

    # 检索
    results = new_builder.search("猫的听觉")
    print("[结果] 检索结果：")
    for res in results:
        print("-", res)

    # 清理测试文件
    os.remove(index_path)
    os.remove(texts_path)
    print("[TEST] 测试通过 ✅")


if __name__ == "__main__":
    test_index_builder_full_pipeline()
