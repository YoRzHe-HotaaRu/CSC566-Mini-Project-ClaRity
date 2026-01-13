from src.vlm_analyzer import VLMAnalyzer
from src.config import VLM_CONFIG
import json

# Print config
print("=== VLM CONFIG ===")
print(json.dumps(VLM_CONFIG, indent=2))
print()

vlm = VLMAnalyzer()

# Test the actual API call with debug
import requests

headers = {
    "Authorization": f"Bearer {vlm.api_key}",
    "Content-Type": "application/json"
}

# Get the image
base64_image = vlm._encode_image_to_base64(
    r'C:\Users\ADMINI~1\AppData\Local\Temp\goose-pasted-images\pasted-img-1768268111575-cl4vsmq-247c2821eb9d35ab.png'
)

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What layer is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    ]
}]

payload = {
    "model": vlm.model,
    "messages": messages,
    "temperature": 0.3,
    "max_tokens": 500
}

print("=== MAKING API REQUEST ===")
print(f"Endpoint: {vlm.endpoint}")
print(f"Model: {vlm.model}")
print(f"Headers: {json.dumps({k: v[:10] + '...' if k == 'Authorization' else v for k, v in headers.items()})}")
print()

try:
    response = requests.post(vlm.endpoint, headers=headers, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print()
    print("=== RESPONSE ===")
    print(response.text[:1000])
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
