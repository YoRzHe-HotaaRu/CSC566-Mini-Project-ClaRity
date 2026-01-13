# Add generate_summary method
with open('gui/main_window.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert (before analysis_error method)
for i, line in enumerate(lines):
    if 'def analysis_error(self, error: str):' in line:
        # Insert the generate_summary method before this line
        summary_method = '''    def generate_summary(self, result: dict, classification: dict):
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

'''
        lines.insert(i, summary_method)
        print(f"SUCCESS: Added generate_summary method at line {i+1}")
        break

# Write back
with open('gui/main_window.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("File saved: gui/main_window.py")
print("The generate_summary method has been added!")
