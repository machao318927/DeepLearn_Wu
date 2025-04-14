# 📦 pre_retrieval/query_rewriting.py

from transformers import pipeline
from typing import List

class QueryRewriter:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.generator = pipeline("text-generation", model=model_name)

    def rewrite(self, query: str, history: List[str] = None) -> str:
        context = " " .join(history) + " " + query if history else query
        rewritten = self.generator(context, max_new_tokens=32, do_sample=True, top_k=50, temperature=0.7)[0]['generated_text']
        return rewritten.strip()


# ✅ 测试用例
if __name__ == '__main__':
    rewriter = QueryRewriter()
    original_query = "苹果手机功能"
    history = ["你喜欢哪种手机？", "我喜欢苹果手机"]
    rewritten_query = rewriter.rewrite(original_query, history)
    print("[原始Query]", original_query)
    print("[历史记录]", history)
    print("[改写后Query]", rewritten_query)