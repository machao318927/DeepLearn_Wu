# 📦 post_retrieval/context_rewriting.py

from transformers import pipeline
import torch


class ContextRewriter:
    def __init__(self, model_name="Langboat/mengzi-t5-base"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.rewriter = pipeline("summarization", model=model_name, device=self.device)

    def rewrite(self, contexts: list[str]) -> list[str]:
        rewritten = []
        for ctx in contexts:
            try:
                result = self.rewriter(ctx, max_length=64, min_length=10, do_sample=False)
                rewritten.append(result[0]["summary_text"])
            except Exception as e:
                rewritten.append(ctx)  # fallback
        return rewritten



# ✅ 测试用例
def test_context_rewriter():
    print("[TEST] 正在测试 ContextRewriter")
    contexts = [
        "rag是检索增强生成模型。",
        "它结合了两个部分：retriever 和 generator。"
    ]
    rewriter = ContextRewriter()
    rewritten = rewriter.rewrite(contexts)
    assert isinstance(rewritten, list) and all(isinstance(r, str) for r in rewritten)
    print("[✅] Rewriting result:", rewritten)


if __name__ == "__main__":
    test_context_rewriter()
