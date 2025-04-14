# 📦 pre_retrieval/query_classification.py

from transformers import pipeline
from typing import List

class QueryClassifier:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    def classify(self, query: str, candidate_labels: List[str]) -> str:
        result = self.classifier(query, candidate_labels)
        return result["labels"][0]


# ✅ 测试用例（可直接运行）
def test_query_classifier():
    classifier = QueryClassifier()
    query = "请告诉我最新的苹果手机有什么功能"
    labels = ["产品咨询", "售后服务", "投诉建议"]
    top_label = classifier.classify(query, labels)
    print(f"[🔍] 输入问题: {query}")
    print(f"[✅] 分类结果: {top_label}")


if __name__ == '__main__':
    test_query_classifier()
