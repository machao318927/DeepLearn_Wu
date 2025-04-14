# 📦 indexing/index_saver_loader.py

import faiss
import pickle

class IndexIOHandler:
    @staticmethod
    def save_index(index, texts, index_path, texts_path):
        """
        保存向量索引和对应的文本数据到磁盘。
        :param index: faiss 索引对象
        :param texts: 原始文本列表
        :param index_path: faiss索引保存路径 (.index)
        :param texts_path: 文本保存路径 (.pkl)
        """
        faiss.write_index(index, index_path)
        with open(texts_path, 'wb') as f:
            pickle.dump(texts, f)

    @staticmethod
    def load_index(index_path, texts_path):
        """
        从磁盘加载索引和文本。
        :param index_path: faiss索引文件路径
        :param texts_path: 文本文件路径
        :return: Tuple (faiss index, list of texts)
        """
        index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)
        return index, texts


# ✅ 测试用例（可直接运行）
def test_index_io_handler():
    from sentence_transformers import SentenceTransformer
    import os

    # 临时数据
    texts = ["RAG 是一种检索增强生成方法。", "它结合了信息检索和语言模型的能力。"]
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(texts, convert_to_numpy=True)

    # 构建 faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 保存路径
    index_path = "test_index.index"
    texts_path = "test_texts.pkl"

    # 保存与加载
    IndexIOHandler.save_index(index, texts, index_path, texts_path)
    loaded_index, loaded_texts = IndexIOHandler.load_index(index_path, texts_path)

    # 检查有效性
    assert isinstance(loaded_index, faiss.IndexFlat), "加载的索引类型错误"
    assert loaded_texts == texts, "加载的文本内容不一致"
    print("[✅] IndexIOHandler 测试通过")

    # 清理临时文件
    os.remove(index_path)
    os.remove(texts_path)


if __name__ == '__main__':
    test_index_io_handler()