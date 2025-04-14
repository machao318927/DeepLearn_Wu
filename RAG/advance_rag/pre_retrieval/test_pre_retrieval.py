# ✅ tests/test_pre_retrieval.py

from query_expansion import QueryExpander
from query_classification import QueryClassifier
from query_routing import QueryRouter

def test_query_expander():
    expander = QueryExpander()
    query = "人工智能的发展方向"
    expanded = expander.expand(query)
    assert isinstance(expanded, str) and len(expanded) > 0
    print("[✅] QueryExpander 测试通过")
    print(f"[DEBUG] Expanded: {expanded}")

def test_query_classifier():
    classifier = QueryClassifier()
    query = "我想了解苹果手机的性能"
    labels = ["产品咨询", "技术问题", "价格信息"]
    label = classifier.classify(query, labels)
    assert label in labels
    print("[✅] QueryClassifier 测试通过")

def test_query_router():
    router = QueryRouter({
        "金融": ["股票", "投资", "银行"],
        "科技": ["人工智能", "机器学习", "计算机"],
        "医疗": ["医生", "医院", "药品"]
    })
    query = "介绍一下人工智能在医疗领域的应用"
    route = router.route(query)
    assert route in ["金融", "科技", "医疗"]
    print("[✅] QueryRouter 测试通过")

if __name__ == '__main__':
    test_query_expander()
    test_query_classifier()
    test_query_router()