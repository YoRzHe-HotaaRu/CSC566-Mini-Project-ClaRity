import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('gui/main_window.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the problematic setStyleSheet line
output = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Find the malformed setStyleSheet line
    if 'self.legend_placeholder.setStyleSheet' in line and len(line.strip()) < 80:
        # Skip this line and the malformed lines that follow
        # Add the correct version
        output.append('        self.legend_placeholder.setStyleSheet("""color: #888; font-style: italic; padding: 10px;""")\n')
        
        # Skip lines until we find setAlignment
        i += 1
        while i < len(lines) and 'setAlignment' not in lines[i]:
            i += 1
        
        # Add the setAlignment line
        if i < len(lines):
            output.append(lines[i])
        i += 1
        continue
    
    output.append(line)
    i += 1

with open('gui/main_window.py', 'w', encoding='utf-8') as f:
    f.writelines(output)

print("Fixed!")
