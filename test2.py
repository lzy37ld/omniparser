import openai
import requests
import os

# 方法2a：使用 requests 直接查看响应头
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    }
)

# 查看限制相关的响应头
for key, value in response.headers.items():
    if 'ratelimit' in key.lower():
        print(f"{key}: {value}")