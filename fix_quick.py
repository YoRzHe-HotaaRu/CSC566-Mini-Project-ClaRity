import sys
with open('gui/main_window.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific problematic setStyleSheet line
old_text = '''self.legend_placeholder.setStyleSheet(""
            color: #888; 
            font-style: italic; 
            padding: 10px;
        " ")'''

new_text = '''self.legend_placeholder.setStyleSheet("""color: #888; font-style: italic; padding: 10px;""")'''

content = content.replace(old_text, new_text)

# Also fix any remaining duplicate legend sections after return panel
lines = content.split('\n')
output = []
skip = False
for i, line in enumerate(lines):
    if '# Layer legend' in line and i > 350:
        # Check if this is after 'return panel'
        if any('return panel' in lines[j] for j in range(max(0, i-10), i)):
            skip = True
            continue
    if skip and 'def create_control_panel' in line:
        skip = False
    if not skip:
        output.append(line)

content = '\n'.join(output)

with open('gui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed!')
