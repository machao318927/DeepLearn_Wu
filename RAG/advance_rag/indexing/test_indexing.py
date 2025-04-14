# 📦 indexing/test_indexing.py

import os
import faiss
import pickle
from embedding_generator import EmbeddingGenerator
from index_builder import IndexBuilder
from index_saver_loader import IndexIOHandler

def test_full_indexing_pipeline():
    print("[🔍] 开始测试完整索引流程...")

    # 1. 初始化模块
    encoder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    builder = IndexBuilder(encoder)

    # 2. 示例文本数据
    texts = ["RAG 是一种检索增强生成方法。", "它结合了信息检索和语言模型的能力。"]

    # 3. 构建索引
    builder.build_index(texts)

    # 4. 保存索引和文本
    index_path = "test.index"
    texts_path = "test_texts.pkl"
    IndexIOHandler.save_index(builder.index, builder.texts, index_path, texts_path)

    # 5. 加载索引和文本
    index_loaded, texts_loaded = IndexIOHandler.load_index(index_path, texts_path)

    # 6. 校验正确性
    assert isinstance(index_loaded, faiss.IndexFlat), "索引加载失败或类型不匹配"
    assert texts_loaded == texts, "加载文本内容不一致"

    print("[✅] 索引构建-保存-加载完整流程测试通过")

    # 7. 清理文件
    os.remove(index_path)
    os.remove(texts_path)

if __name__ == '__main__':
    test_full_indexing_pipeline()