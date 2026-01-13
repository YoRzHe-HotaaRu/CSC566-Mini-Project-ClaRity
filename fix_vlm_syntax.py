with open('src/vlm_analyzer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the broken regex at lines 292-294 (indices 291-293)
new_line = "        reasoning_match = re.search(r'REASONING:\\s*(.+?)(?:\\n\\n|\\Z)', content_stripped, re.IGNORECASE | re.DOTALL)\n"

lines[291] = new_line
del lines[292:294]  # Remove the next 2 broken lines

with open('src/vlm_analyzer.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed syntax error in vlm_analyzer.py!')
