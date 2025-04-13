# rag_engine.py

"""
Naive RAG 原型系统：从文档构建索引，用户查询后检索并生成 Prompt
依赖库：sentence-transformers, faiss-cpu
作者：马超专用 ChatGPT 助手
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class NaiveRAG:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", top_k: int = 3):
        """
        初始化 Naive RAG 系统
        :param embedding_model_name: 句子嵌入模型名称（SentenceTransformer库中的预训练模型）
        :param top_k: 检索文档返回前 top_k 个
        """
        self.model = SentenceTransformer(embedding_model_name)
        # 默认做的是 mean pooling，也就是将 BERT 最后一层所有 token 的向量平均，得到整句的向量表示
        self.index = None
        self.documents = []
        self.doc_embeddings = None
        self.top_k = top_k

    def build_index(self, documents: list[str]):
        """
        RAG原理：把所有待检索的文档转成向量，并组织到向量数据库中
        构建向量索引
        :param documents: 文本列表，每个字符串是一篇文档
        """
        self.documents = documents # 保存原始文档
        self.doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        # 将文档列表转成一个 [N, 384] 的 numpy 矩阵，N是文档数，384是句向量维度

        dim = self.doc_embeddings.shape[1]  # 获取向量维度
        self.index = faiss.IndexFlatL2(dim)  # 创建一个 FAISS 向量索引结构，支持快速相似度搜索（使用欧氏距离 L2）
        self.index.add(self.doc_embeddings) # 把文档向量加进索引中，相当于建立了一个“语义文档图书馆”

        print(f"[INFO] 已成功建立向量索引，文档数: {len(documents)}，向量维度: {dim}")

    def retrieve(self, query: str) -> list[str]:
        """
        RAG原理：将用户提问转化为向量，去语义向量数据库中找 Top-K 相似文档
        基于语义相似性检索最相关的文档
        :param query: 用户查询字符串
        :return: 最相似的 top_k 篇文档
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True) # 将用户 query 转成 [1, 384] 的向量
        distances, indices = self.index.search(query_embedding, self.top_k) # 进行L2距离最近的Top-K检索
        top_docs = [self.documents[i] for i in indices[0]] # 返回的是最相似的文档编号，你取出对应原始文本作为结果返回
        return top_docs

    def build_prompt(self, query: str) -> str:
        """
        RAG原理：把问题和检索到的文档拼接起来，送给生成器
        构造 Prompt：将检索到的文档与问题拼接为 Prompt
        :param query: 用户查询问题
        :return: 拼接后的 Prompt 字符串
        """
        top_docs = self.retrieve(query)
        prompt = "请根据以下内容回答问题：\n\n"
        for i, doc in enumerate(top_docs):
            prompt += f"[文档{i + 1}]:\n{doc}\n\n"
        prompt += f"[用户问题]:\n{query}\n"
        return prompt
