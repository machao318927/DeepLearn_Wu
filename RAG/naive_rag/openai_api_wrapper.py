from openai import OpenAI

# ✅ 第三方服务的 Key 和 Base URL
client = OpenAI(
    api_key="sk-tLpNDWa6j2n3MJaGAKQJ67EdtmDLVYndaIylcPASc7N1E6mn",
    base_url="https://chatapi.littlewheat.com/v1"
)

def call_chatgpt(prompt: str, model: str = "gpt-3.5-turbo"):
    """
    使用自定义 openai-compatible 接口调用 ChatGPT
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业问答助手，请根据文档认真回答用户的问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()
