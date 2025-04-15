# 📦 post_retrieval/context_fusion.py

from typing import List

class ContextFusion:
    def __init__(self, method: str = "concat"):
        """
        :param method: 融合方式，目前支持："concat"（简单拼接）、"paragraph"（换段落拼接）
        """
        self.method = method

    def fuse(self, chunks: List[str]) -> str:
        """
        将多个检索结果拼接为统一上下文。
        :param chunks: 检索片段列表
        :return: 融合后的文本
        """
        if not chunks:
            return ""

        if self.method == "concat":
            return " ".join(chunks)
        elif self.method == "paragraph":
            return "\n\n".join(chunks)
        else:
            raise ValueError(f"不支持的融合方式: {self.method}")


# ✅ 测试用例

def test_context_fusion():
    chunks = [
        "RAG 结合了信息检索和生成模型的能力。",
        "它适合用于问答、摘要等任务。",
        "多个检索片段可能需要融合以支持复杂回答。"
    ]

    print("[TEST] ContextFusion")

    fusion1 = ContextFusion(method="concat")
    fused1 = fusion1.fuse(chunks)
    print("[concat]", fused1)
    assert isinstance(fused1, str) and len(fused1) > 0

    fusion2 = ContextFusion(method="paragraph")
    fused2 = fusion2.fuse(chunks)
    print("[paragraph]\n", fused2)
    assert "\n\n" in fused2

    print("[✅] ContextFusion 测试通过")


if __name__ == '__main__':
    test_context_fusion()
