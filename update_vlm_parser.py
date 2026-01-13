with open('src/vlm_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the _parse_analysis_response method
old_method = '''    def _parse_analysis_response(self, content: str) -> Dict:
        """
        Parse VLM response to extract structured data.
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed result dictionary
        """
        result = {
            "layer_number": None,
            "layer_name": None,
            "confidence": 0.0,
            "material": None,
            "texture_description": None,
            "additional_notes": None
        }
        
        content_lower = content.lower()
        
        # Try to extract layer number
        for layer_num in range(1, 6):
            layer_info = ROAD_LAYERS[layer_num]
            layer_name = layer_info["name"].lower()
            
            if f"layer {layer_num}" in content_lower or layer_name in content_lower:
                result["layer_number"] = layer_num
                result["layer_name"] = layer_info["name"]
                result["full_name"] = layer_info["full_name"]
                result["material"] = layer_info["material"]
                break
        
        # Try to extract confidence
        import re
        confidence_patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*confidence",
            r"confidence[:\s]+(\d+(?:\.\d+)?)"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content_lower)
            if match:
                conf_value = float(match.group(1))
                # Normalize to 0-1 if percentage
                result["confidence"] = conf_value / 100 if conf_value > 1 else conf_value
                break
        
        # If no confidence found, estimate based on certainty words
        if result["confidence"] == 0:
            if any(word in content_lower for word in ["definitely", "clearly", "certainly"]):
                result["confidence"] = 0.9
            elif any(word in content_lower for word in ["likely", "probably", "appears"]):
                result["confidence"] = 0.7
            elif any(word in content_lower for word in ["possibly", "might", "could"]):
                result["confidence"] = 0.5
            else:
                result["confidence"] = 0.6
        
        # Extract texture description
        texture_keywords = ["texture", "surface", "pattern", "roughness"]
        for keyword in texture_keywords:
            if keyword in content_lower:
                # Try to find sentence with keyword
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        result["texture_description"] = sentence.strip()
                        break
                break
        
        return result'''

new_method = '''    def _parse_analysis_response(self, content: str) -> Dict:
        """
        Parse VLM response to extract structured data.
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed result dictionary
        """
        import re
        
        result = {
            "layer_number": None,
            "layer_name": None,
            "full_name": None,
            "confidence": 0.0,
            "material": None,
            "texture_description": None,
            "additional_notes": None,
            "reasoning": None
        }
        
        content_stripped = content.strip()
        content_lower = content_stripped.lower()
        
        # Try to extract structured fields first (new format)
        # Extract LAYER number
        layer_match = re.search(r'LAYER:\s*(\d)', content_stripped, re.IGNORECASE)
        if layer_match:
            layer_num = int(layer_match.group(1))
            if 1 <= layer_num <= 5:
                result["layer_number"] = layer_num
                layer_info = ROAD_LAYERS[layer_num]
                result["layer_name"] = layer_info["name"]
                result["full_name"] = layer_info["full_name"]
                result["material"] = layer_info["material"]
        
        # If structured extraction failed, try keyword matching
        if result["layer_number"] is None:
            for layer_num in range(1, 6):
                layer_info = ROAD_LAYERS[layer_num]
                layer_name = layer_info["name"].lower()
                
                # Check for various name formats
                if (f"layer {layer_num}" in content_lower or 
                    f"layer {layer_num} -" in content_lower or
                    layer_name in content_lower or
                    layer_info["full_name"].lower() in content_lower):
                    result["layer_number"] = layer_num
                    result["layer_name"] = layer_info["name"]
                    result["full_name"] = layer_info["full_name"]
                    result["material"] = layer_info["material"]
                    break
        
        # Extract confidence (try multiple formats)
        confidence_patterns = [
            r'CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%',  # "CONFIDENCE: 85%"
            r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%',  # "confidence: 85%"
            r'(\d+(?:\.\d+)?)\s*%\s*confidence',     # "85% confidence"
            r'confidence[:\s]+(\d+(?:\.\d+)?)',        # "confidence: 0.85"
            r'(\d+(?:\.\d+)?)\s*%\s*confident',       # "85% confident"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content_stripped, re.IGNORECASE)
            if match:
                conf_value = float(match.group(1))
                # Normalize to 0-1 range
                result["confidence"] = conf_value / 100 if conf_value > 1 else conf_value
                break
        
        # If no confidence found, estimate based on certainty words
        if result["confidence"] == 0:
            high_certainty = ["definitely", "clearly", "certainly", "undoubtedly", "unquestionably"]
            medium_certainty = ["likely", "probably", "appears to be", "seems to be", "suggests"]
            low_certainty = ["possibly", "might be", "could be", "may be"]
            
            if any(word in content_lower for word in high_certainty):
                result["confidence"] = 0.9
            elif any(word in content_lower for word in medium_certainty):
                result["confidence"] = 0.7
            elif any(word in content_lower for word in low_certainty):
                result["confidence"] = 0.5
            else:
                result["confidence"] = 0.6  # Default moderate confidence
        
        # Extract reasoning if present
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n\n|\Z)', content_stripped, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # Try to find any explanation text
            for keyword in ["because", "since", "due to", "based on", "the texture", "the color"]:
                if keyword in content_lower:
                    sentences = content_stripped.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            result["reasoning"] = sentence.strip()
                            break
                    break
        
        # Extract texture/material description
        texture_keywords = ["texture", "surface", "material", "appearance"]
        for keyword in texture_keywords:
            if keyword in content_lower and not result["reasoning"]:
                sentences = content_stripped.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower() and len(sentence.strip()) > 10:
                        result["texture_description"] = sentence.strip()
                        break
                if result["texture_description"]:
                    break
        
        return result'''

# Replace the method
content = content.replace(old_method, new_method)

# Write back
with open('src/vlm_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated _parse_analysis_response method in vlm_analyzer.py')
print('Improvements:')
print('  - Better structured field extraction')
print('  - More confidence format patterns')
print('  - Extracts reasoning field')
print('  - Fallback keyword matching')
