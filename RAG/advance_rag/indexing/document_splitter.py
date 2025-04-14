
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "。", "，", ",", " "]
        )

    def split(self, document: str) -> List[str]:
        """
        将长文本划分为多个片段（chunk）。

        参数：
            document: 原始文本内容
        返回：
            List[str]: 分割后的文本片段
        """
        return self.splitter.split_text(document)
