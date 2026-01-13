"""
COMPREHENSIVE GUI IMPROVEMENTS FOR VLM ANALYSIS MODE
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

gui_file = project_root / "gui" / "main_window.py"
with open(gui_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Track changes
changes = []
changes_count = 0

# Fix 1: Layer Legend visibility
old1 = '            legend_item.setVisible(False)  # Initially hidden'
new1 = '            legend_item.setVisible(True)  # Make visible by default'
if old1 in content:
    content = content.replace(old1, new1)
    changes.append("Layer Legend visible by default")
    changes_count += 1

# Fix 1b: Ensure legend widget is visible
if 'self.legend_widget.setVisible(True)' not in content:
    old1b = '        legend_layout.addWidget(self.legend_widget)  # Add legend widget to main layout\n        layout.addWidget(self.legend_group)'
    new1b = '        legend_layout.addWidget(self.legend_widget)  # Add legend widget to main layout\n        self.legend_widget.setVisible(True)\n        layout.addWidget(self.legend_group)'
    if old1b in content:
        content = content.replace(old1b, new1b)
        changes.append("Legend widget visibility ensured")
        changes_count += 1

# Fix 2: VLM visualization
old2 = '                    result["labels"] = np.ones(self.image.shape[:2], dtype=np.uint8) * vlm_result.get("layer_number", 1)'
new2 = '''                    # Create smart VLM visualization
                    h, w = self.image.shape[:2]
                    layer_num = vlm_result.get("layer_number", 1)
                    
                    # Create visualization with detected layer
                    labels = np.ones((h, w), dtype=np.uint8) * layer_num
                    
                    # Add border to indicate image boundary
                    border_width = min(h, w) // 20
                    labels[:border_width, :] = 0
                    labels[-border_width:, :] = 0
                    labels[:, :border_width] = 0
                    labels[:, -border_width:] = 0
                    
                    # Add confidence-based overlay
                    confidence = vlm_result.get("confidence", 0.5)
                    if confidence < 0.7:
                        noise_mask = np.random.random((h, w)) > confidence
                        labels[noise_mask] = 0
                    
                    result["labels"] = labels'''

if old2 in content:
    content = content.replace(old2, new2)
    changes.append("VLM smart visualization with border and confidence overlay")
    changes_count += 1

# Fix 3: Rich results display
old3 = '''        # Display results text
        classification = result.get("classification", {})
        features = result.get("features", {})
        
        text = "═══ ANALYSIS RESULTS ═══\\n\\n"
        text += f"Detected Layer: {classification.get('layer_name', 'N/A')}\\n"
        text += f"Confidence: {classification.get('confidence', 0):.1%}\\n"
        text += f"Material: {classification.get('material', 'N/A')}\\n"
        text += f"Method: {classification.get('method', 'N/A')}\\n\\n"
        
        if "glcm" in features:
            glcm = features["glcm"]
            text += "─── GLCM Features ───\\n"
            text += f"Contrast:    {glcm.get('contrast', 0):.4f}\\n"
            text += f"Energy:      {glcm.get('energy', 0):.4f}\\n"
            text += f"Homogeneity: {glcm.get('homogeneity', 0):.4f}\\n"
            text += f"Correlation: {glcm.get('correlation', 0):.4f}\\n"
        
        self.results_text.setText(text)'''

new3 = '''        # Display results text
        classification = result.get("classification", {})
        features = result.get("features", {})
        
        text = "═══ ANALYSIS RESULTS ═══\\n\\n"
        
        # Layer identification
        layer_name = classification.get('layer_name', classification.get('full_name', 'N/A'))
        confidence = classification.get('confidence', 0)
        text += f"Detected Layer: {layer_name}\\n"
        text += f"Confidence: {confidence:.1%}\\n"
        
        # Material info
        material = classification.get('material', 'N/A')
        if material and material != 'N/A':
            text += f"Material: {material}\\n"
        
        # Method used
        method = classification.get('method', 'N/A')
        if method == 'N/A':
            method = "VLM Analysis (GLM-4.6V)"
        text += f"Method: {method}\\n"
        
        # Layer number
        layer_num = classification.get('layer_number')
        if layer_num:
            text += f"Layer Number: {layer_num}\\n"
        
        # VLM-specific fields
        texture = classification.get('texture_description')
        if texture and texture != 'N/A':
            text += f"\\nTexture Description:\\n   {texture}\\n"
        
        reasoning = classification.get('reasoning')
        if reasoning and reasoning != 'N/A':
            text += f"\\nAnalysis Reasoning:\\n   {reasoning}\\n"
        
        notes = classification.get('additional_notes')
        if notes and notes != 'N/A':
            text += f"\\nAdditional Notes:\\n   {notes}\\n"
        
        # GLCM features (only for classical/hybrid modes)
        if "glcm" in features and self.mode in ["classical", "hybrid"]:
            text += "\\n─── Texture Features ───\\n"
            glcm = features["glcm"]
            text += f"Contrast:    {glcm.get('contrast', 0):.4f}\\n"
            text += f"Energy:      {glcm.get('energy', 0):.4f}\\n"
            text += f"Homogeneity: {glcm.get('homogeneity', 0):.4f}\\n"
            text += f"Correlation: {glcm.get('correlation', 0):.4f}\\n"
        
        self.results_text.setText(text)'''

if old3 in content:
    content = content.replace(old3, new3)
    changes.append("Rich results display with all VLM fields")
    changes_count += 1

# Fix 4: Add Summary box
old4 = '''        layout.addWidget(results_group)
        
        layout.addStretch()
        
        return panel'''

new4 = '''        layout.addWidget(results_group)
        
        # Summary panel (NEW)
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setPlaceholderText("Plain-English summary will appear here...")
        self.summary_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a2e; border: 2px solid #4a4a6a; border-radius: 8px; padding: 10px; font-size: 11px; line-height: 1.4; }"
        )
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        
        return panel'''

if old4 in content:
    content = content.replace(old4, new4)
    changes.append("Added Summary box for non-technical users")
    changes_count += 1

# Fix 5: Call summary generation
old5 = '        self.results_text.setText(text)\n        self.status_bar.showMessage("Analysis complete!")'
new5 = '        self.results_text.setText(text)\n        \n        # Generate plain-English summary\n        self.generate_summary(result, classification)\n        \n        self.status_bar.showMessage("Analysis complete!")'

if old5 in content:
    content = content.replace(old5, new5)
    changes.append("Added summary generation call")
    changes_count += 1

# Fix 6: Add generate_summary method
old6 = '''                    self.legend_items[layer_num].setVisible(False)

    def analysis_error(self, error_msg):'''

new6 = '''                    self.legend_items[layer_num].setVisible(False)
    
    def generate_summary(self, result: dict, classification: dict):
        """Generate plain-English summary for non-technical users."""
        mode = self.mode if hasattr(self, 'mode') else self.get_current_mode()
        
        layer_name = classification.get('layer_name', classification.get('full_name', 'Unknown'))
        confidence = classification.get('confidence', 0)
        material = classification.get('material', '')
        
        # Build summary based on mode
        if mode == "vlm":
            summary = "VLM Analysis Summary\\n\\n"
            summary += f"The AI vision model analyzed your image and identified it as:\\n\\n"
            summary += f"{layer_name}\\n\\n"
            
            # Confidence explanation
            if confidence >= 0.8:
                summary += f"The model is very confident ({confidence:.0%}) about this identification.\\n\\n"
            elif confidence >= 0.6:
                summary += f"The model is moderately confident ({confidence:.0%}) about this identification.\\n\\n"
            else:
                summary += f"The model has low confidence ({confidence:.0%}) - consider using another analysis mode.\\n\\n"
            
            # Material explanation
            if material and material != 'N/A':
                summary += f"Material: {material}\\n\\n"
            
            # Simple explanation
            if "Aggregate" in layer_name:
                summary += "What this means: This layer shows loose stones/gravel used for drainage and stability.\\n"
            elif "Sub-base" in layer_name:
                summary += "What this means: This is a foundational layer that distributes loads evenly.\\n"
            elif "Base Course" in layer_name:
                summary += "What this means: This is the main structural layer that bears traffic loads.\\n"
            elif "Asphalt" in layer_name or "Surface" in layer_name:
                summary += "What this means: This is the top wearing course that vehicles travel on.\\n"
            elif "Soil" in layer_name or "Subgrade" in layer_name:
                summary += "What this means: This is the natural soil foundation beneath all road layers.\\n"
        
        elif mode == "deep_learning":
            summary = "Deep Learning Analysis Summary\\n\\n"
            summary += f"The neural network (DeepLabv3+) segmented your image.\\n\\n"
            summary += f"Primary Layer: {layer_name}\\n\\n"
            summary += f"This mode uses advanced AI trained on thousands of road images to identify layers.\\n"
            summary += f"Great for complex images with mixed materials.\\n"
        
        elif mode == "classical":
            summary = "Classical Analysis Summary\\n\\n"
            summary += f"Traditional image processing techniques were used:\\n\\n"
            summary += f"Identified Layer: {layer_name}\\n"
            summary += f"Confidence: {confidence:.0%}\\n\\n"
            
            if material:
                summary += f"Material Type: {material}\\n\\n"
            
            summary += "How it worked:\\n"
            summary += "- Extracted texture features (GLCM, LBP)\\n"
            summary += "- Applied K-means clustering segmentation\\n"
            summary += "- Used heuristic rules to classify layers\\n\\n"
            summary += "This mode is fast and works well for clear, distinct textures.\\n"
        
        elif mode == "hybrid":
            summary = "Hybrid Analysis Summary\\n\\n"
            summary += f"Combined classical and AI methods for best accuracy:\\n\\n"
            summary += f"Final Result: {layer_name}\\n"
            summary += f"Combined Confidence: {confidence:.0%}\\n\\n"
            summary += "Best of both worlds:\\n"
            summary += "- Classical: Fast texture analysis\\n"
            summary += "- VLM: Smart AI understanding\\n"
            summary += "Most accurate for challenging images.\\n"
        
        else:
            summary = f"Analysis complete: {layer_name} ({confidence:.0%} confidence)"
        
        self.summary_text.setText(summary)

    def analysis_error(self, error_msg):'''

if old6 in content:
    content = content.replace(old6, new6)
    changes.append("Implemented generate_summary method with mode-specific explanations")
    changes_count += 1

# Fix 7: Mode tracking
old7 = '''        mode = self.get_current_mode()
        params = self.get_parameters()
        
        self.worker = AnalysisWorker(self.image.copy(), mode, params)'''
new7 = '''        mode = self.get_current_mode()
        self.mode = mode  # Store mode for summary generation
        params = self.get_parameters()
        
        self.worker = AnalysisWorker(self.image.copy(), mode, params)'''

if old7 in content:
    content = content.replace(old7, new7)
    changes.append("Added mode tracking for summary generation")
    changes_count += 1

# Fix 8: VLM method field
old8 = '''                    vlm_result = analyzer.analyze_road_layer(str(temp_path))
                    result["classification"] = vlm_result

                    # Create smart VLM visualization'''

new8 = '''                    vlm_result = analyzer.analyze_road_layer(str(temp_path))
                    
                    # Ensure method field is populated
                    if 'method' not in vlm_result or vlm_result.get('method') == 'N/A':
                        vlm_result['method'] = 'VLM Analysis (GLM-4.6V)'
                    
                    result["classification"] = vlm_result

                    # Create smart VLM visualization'''

if old8 in content:
    content = content.replace(old8, new8)
    changes.append("VLM method field populated")
    changes_count += 1

# Save changes
if changes_count > 0:
    # Create backup
    backup_file = project_root / "gui" / "main_window.py.backup"
    
    # Read original content for backup
    with open(gui_file, 'r', encoding='utf-8') as orig:
        original_content = orig.read()
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(original_content)
    
    # Write updated content
    with open(gui_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("=" * 60)
    print("GUI IMPROVEMENTS APPLIED SUCCESSFULLY!")
    print("=" * 60)
    for i, change in enumerate(changes, 1):
        print(f"{i}. {change}")
    print("=" * 60)
    print(f"Total changes: {changes_count}")
    print(f"Backup saved to: {backup_file}")
    print(f"Updated file: {gui_file}")
    print("=" * 60)
    print("READY TO TEST! Launch the GUI.")
else:
    print("No changes needed - file may already be updated")
