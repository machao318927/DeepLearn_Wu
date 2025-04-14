# 📦 pre_retrieval/query_expansion.py

from transformers import pipeline

class QueryExpander:
    def __init__(self, model_name="google/flan-t5-base"):
        self.generator = pipeline("text2text-generation", model=model_name)
        print("Device set to use", self.generator.device)

    def expand(self, query: str) -> str:
        prompt = f"请扩展下面的查询：{query}"
        result = self.generator(prompt, max_length=64, clean_up_tokenization_spaces=True)
        if "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "text" in result[0]:
            return result[0]["text"]
        else:
            return ""


# ✅ 测试用例（可直接运行）
def test_query_expander():
    expander = QueryExpander()
    query = "电动汽车的续航问题"
    expanded = expander.expand(query)
    assert isinstance(expanded, str) and len(expanded) > 0
    print("[✅] QueryExpander 测试通过")
    print("[DEBUG] Expanded:", expanded)


if __name__ == '__main__':
    test_query_expander()