"""
FINAL VERIFICATION - GUI IMPROVEMENTS
=====================================
"""

import sys
from pathlib import Path

gui_file = Path(__file__).parent / "gui" / "main_window.py"

with open(gui_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("="*70)
print("GUI IMPROVEMENTS VERIFICATION")
print("="*70)

checks = []

# Check 1: Legend visibility
if 'legend_item.setVisible(True)' in content:
    checks.append(('PASS', 'Legend items set to visible'))
else:
    checks.append(('FAIL', 'Legend items NOT set to visible'))

if 'self.legend_widget.setVisible(True)' in content:
    checks.append(('PASS', 'Legend widget visibility enabled'))
else:
    checks.append(('FAIL', 'Legend widget visibility NOT enabled'))

# Check 2: VLM smart visualization
if 'border_width = min(h, w) // 20' in content:
    checks.append(('PASS', 'VLM smart visualization with border'))
else:
    checks.append(('FAIL', 'VLM smart visualization NOT found'))

if 'noise_mask = np.random.random((h, w)) > confidence' in content:
    checks.append(('PASS', 'VLM confidence-based overlay'))
else:
    checks.append(('FAIL', 'VLM confidence overlay NOT found'))

# Check 3: Rich results display
if 'texture_description' in content:
    checks.append(('PASS', 'Results include texture_description'))
else:
    checks.append(('WARN', 'texture_description not in results'))

if 'reasoning' in content:
    checks.append(('PASS', 'Results include reasoning'))
else:
    checks.append(('WARN', 'reasoning not in results'))

# Check 4: Method field
if "vlm_result['method'] = 'VLM Analysis (GLM-4.6V)'" in content:
    checks.append(('PASS', 'VLM method field population'))
else:
    checks.append(('FAIL', 'VLM method field NOT populated'))

# Check 5: Summary box
if 'QGroupBox("Summary")' in content or 'QGroupBox("Summary")' in content:
    checks.append(('PASS', 'Summary box created'))
else:
    checks.append(('FAIL', 'Summary box NOT created'))

if 'def generate_summary' in content:
    checks.append(('PASS', 'generate_summary method exists'))
else:
    checks.append(('FAIL', 'generate_summary method NOT found'))

if 'self.summary_text = QTextEdit()' in content:
    checks.append(('PASS', 'Summary text widget created'))
else:
    checks.append(('FAIL', 'Summary text widget NOT found'))

# Check 6: Mode tracking
if 'self.mode = mode' in content:
    checks.append(('PASS', 'Mode tracking implemented'))
else:
    checks.append(('FAIL', 'Mode tracking NOT found'))

# Check 7: Summary generation call
if 'self.generate_summary(result, classification)' in content:
    checks.append(('PASS', 'Summary generation called'))
else:
    checks.append(('FAIL', 'Summary generation NOT called'))

# Print results
print()
for status, description in checks:
    symbol = '[+]' if status == 'PASS' else '[X]' if status == 'FAIL' else '[!]'
    print(f"  {symbol} {description}")

print()
print("="*70)

# Summary
pass_count = sum(1 for s, _ in checks if s == 'PASS')
fail_count = sum(1 for s, _ in checks if s == 'FAIL')
warn_count = sum(1 for s, _ in checks if s == 'WARN')

print(f"RESULTS: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARN")
print("="*70)

if fail_count == 0:
    print("\n[+] ALL CRITICAL FIXES APPLIED!")
    print("    Ready for testing.")
else:
    print(f"\n[X] {fail_count} critical issue(s) remain")
    print("    Please review the FAIL items above.")

print()
print("Next steps:")
print("1. Launch GUI: python gui/main_window.py")
print("2. Load a road layer image")
print("3. Test VLM Analysis mode")
print("4. Verify all 5 layers in legend")
print("5. Verify rich results display")
print("6. Verify summary box appears")
print("7. Verify method field shows 'VLM Analysis (GLM-4.6V)'")
print("="*70)
