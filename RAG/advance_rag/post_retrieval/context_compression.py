# 📦 post_retrieval/context_compression.py
from transformers import pipeline
import torch


class ContextCompressor:
    def __init__(self, model_name="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=model_name, device=self.device)

    def compress(self, documents, max_length=64):
        """
        将多个文档压缩为一段摘要（适配中文）
        """
        combined = "。".join(documents)
        result = self.summarizer(combined, max_length=max_length, min_length=20, do_sample=False)
        return result[0]["summary_text"]


# ✅ 测试用例
def test_context_compressor():
    print("[TEST] 正在测试 ContextCompressor")
    docs = [
        "RAG 是一种结合检索与生成的 NLP 框架。",
        "RAG 框架可以有效提升问答系统的性能。",
        "RAG 使用向量检索和生成式模型。"
    ]
    compressor = ContextCompressor()
    summary = compressor.compress(docs)
    assert isinstance(summary, str) and len(summary) > 0
    print("[✅] Compression result:", summary)


if __name__ == "__main__":
    test_context_compressor()
