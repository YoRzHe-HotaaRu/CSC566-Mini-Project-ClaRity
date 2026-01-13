from src.vlm_analyzer import VLMAnalyzer
from src.config import ROAD_LAYERS
import json

vlm = VLMAnalyzer()
result = vlm.analyze_road_layer(
    r'C:\Users\ADMINI~1\AppData\Local\Temp\goose-pasted-images\pasted-img-1768268111575-cl4vsmq-247c2821eb9d35ab.png'
)

print('=== VLM RAW RESPONSE ===')
print(json.dumps(result, indent=2))
