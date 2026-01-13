# Fix VLM method field
with open('gui/main_window.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix VLM method field
old = '''                    vlm_result = analyzer.analyze_road_layer(str(temp_path))
                    result["classification"] = vlm_result
                    # Create smart VLM visualization'''

new = '''                    vlm_result = analyzer.analyze_road_layer(str(temp_path))

                    # Ensure method field is populated
                    if 'method' not in vlm_result or vlm_result.get('method') == 'N/A':
                        vlm_result['method'] = 'VLM Analysis (GLM-4.6V)'

                    result["classification"] = vlm_result

                    # Create smart VLM visualization'''

if old in content:
    content = content.replace(old, new)
    with open('gui/main_window.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: VLM method field fix applied")
else:
    print("INFO: VLM method field fix may already be applied")
    # Check if method field population exists
    if "VLM Analysis (GLM-4.6V)" in content:
        print("CONFIRMED: Method field population exists")
    else:
        print("WARNING: Method field population not found")
