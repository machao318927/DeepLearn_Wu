# indexing/embedding_generator.py

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化Embedding生成器
        :param model_name: 预训练向量模型名称
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        对多个文本进行向量编码
        :param texts: 文本列表
        :return: 对应的向量矩阵（NumPy数组）
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

# # 测试
# def test_embedding_generation():
#     # 初始化向量生成器
#     generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
#
#     # 测试文本
#     texts = [
#         "猫是一种常见的家养动物。",
#         "狗具有很强的服从性。"
#     ]
#
#     # 获取向量
#     embeddings = generator.encode(texts)
#
#     # 测试返回类型与维度
#     assert isinstance(embeddings, np.ndarray), "返回类型应为 numpy.ndarray"
#     assert embeddings.shape[0] == len(texts), "向量数量应与输入文本一致"
#     assert embeddings.shape[1] > 0, "向量维度应大于 0"
#
#     print("✅ test_embedding_generation 测试通过！")
#
# if __name__ == "__main__":
#     test_embedding_generation()