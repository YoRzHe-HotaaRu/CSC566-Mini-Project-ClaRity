from src.vlm_analyzer import VLMAnalyzer
import json

vlm = VLMAnalyzer()

# Simulate the response we got
test_response = {
    "id": "test",
    "model": "z-ai/glm-4.6v",
    "choices": [{
        "finish_reason": "stop",
        "message": {
            "content": "LAYER: Base Course\nCONFIDENCE: 85%\n\nThis appears to be a base course layer with coarse aggregate.",
            "role": "assistant"
        }
    }]
}

# Test parsing
content = test_response["choices"][0]["message"]["content"]
print("=== TEST CONTENT ===")
print(content)
print()

result = vlm._parse_analysis_response(content)
result["success"] = True
result["raw_response"] = content

print("=== PARSED RESULT ===")
print(json.dumps(result, indent=2))
