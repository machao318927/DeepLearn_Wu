# 📦 pre_retrieval/query_routing.py

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util

class QueryRouter:
    def __init__(self, route_map: Dict[str, List[str]], model_name="all-MiniLM-L6-v2"):
        """
        初始化路由器。
        :param route_map: 每个路由类别对应的文档集合（用于匹配相似度）
        :param model_name: 向量模型名
        """
        self.route_map = route_map
        self.encoder = SentenceTransformer(model_name)
        self.route_embeddings = {
            route: self.encoder.encode(samples, convert_to_tensor=True).mean(dim=0)
            for route, samples in route_map.items()
        }

    def route(self, query: str) -> str:
        """
        返回最匹配的路由类别。
        :param query: 用户查询
        :return: 匹配路由键
        """
        query_vec = self.encoder.encode(query, convert_to_tensor=True)
        best_score = -1
        best_route = None
        for route, rep_vec in self.route_embeddings.items():
            score = util.pytorch_cos_sim(query_vec, rep_vec).item()
            if score > best_score:
                best_score = score
                best_route = route
        return best_route


# ✅ 测试用例（可直接运行）
def test_query_router():
    routes = {
        "技术文档库": ["如何使用API", "模型部署方式", "安装指南"],
        "客户支持库": ["如何联系客服", "售后服务流程", "退款政策"],
        "市场营销库": ["产品优势介绍", "推广策略", "市场趋势"]
    }
    router = QueryRouter(route_map=routes)
    query = "请问怎么联系客服？"
    route = router.route(query)
    print(f"[🔍] 输入问题: {query}")
    print(f"[✅] 匹配路由: {route}")


if __name__ == '__main__':
    test_query_router()