"""
PDF Report Generator for Road Surface Layer Analyzer
Generates professional academic-style analysis reports.

CSC566 Image Processing Mini Project
"""

import os
import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import cv2
import numpy as np

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Charts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config import ROAD_LAYERS


class ReportGenerator:
    """Generate PDF analysis reports with academic formatting."""
    
    # ClaRity color scheme
    COLORS = {
        'primary_blue': colors.Color(0.13, 0.59, 0.95),      # #2196F3
        'primary_green': colors.Color(0.30, 0.69, 0.31),     # #4CAF50
        'dark': colors.Color(0.12, 0.16, 0.19),              # #1f2a30
        'text': colors.Color(0.2, 0.2, 0.2),
        'light_gray': colors.Color(0.9, 0.9, 0.9),
    }
    
    def __init__(self, result: Dict, image: np.ndarray, mode: str, params: Dict):
        """
        Initialize report generator.
        
        Args:
            result: Analysis result dictionary
            image: Original image (BGR)
            mode: Analysis mode used
            params: Analysis parameters
        """
        self.result = result
        self.image = image
        self.mode = mode
        self.params = params
        self.elements = []
        self.styles = self._create_styles()
        
    def _create_styles(self) -> Dict:
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()
        
        # Title style
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.COLORS['dark'],
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        # Section heading
        styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=self.COLORS['primary_blue'],
            spaceBefore=20,
            spaceAfter=10
        ))
        
        # Subsection
        styles.add(ParagraphStyle(
            name='SubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=self.COLORS['primary_green'],
            spaceBefore=15,
            spaceAfter=8
        ))
        
        # Body text (use different name to avoid conflict)
        styles.add(ParagraphStyle(
            name='ReportBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=self.COLORS['text'],
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14
        ))
        
        # Caption (use different name to avoid conflict)
        styles.add(ParagraphStyle(
            name='ReportCaption',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=4
        ))
        
        return styles
        
    def generate(self, output_path: str) -> bool:
        """
        Generate the PDF report.
        
        Args:
            output_path: Path to save the PDF
            
        Returns:
            True if successful
        """
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            self.elements = []
            
            # Build report sections
            self._add_header()
            self._add_images()
            self._add_layer_analysis()
            self._add_texture_features()
            self._add_settings()
            self._add_chart()
            self._add_vlm_analysis()
            self._add_discussion()
            self._add_conclusion()
            self._add_footer()
            
            # Build PDF
            doc.build(self.elements)
            return True
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return False
            
    def _add_header(self):
        """Add report header with title, date, and branding."""
        # Logo (if exists)
        logo_path = Path(__file__).parent.parent / "gui" / "assets" / "logo.png"
        if logo_path.exists():
            logo = Image(str(logo_path), width=1*inch, height=1*inch)
            self.elements.append(logo)
            self.elements.append(Spacer(1, 10))
        
        # Title
        self.elements.append(Paragraph(
            "Road Surface Layer Analysis Report",
            self.styles['ReportTitle']
        ))
        
        # Subtitle with branding
        self.elements.append(Paragraph(
            "<i>Generated by ClaRity - Road Surface Layer Analyzer</i>",
            self.styles['ReportCaption']
        ))
        
        # Course info
        self.elements.append(Paragraph(
            "CSC566 Image Processing Mini Project",
            self.styles['ReportCaption']
        ))
        
        self.elements.append(Spacer(1, 10))
        
        # Report metadata table
        now = datetime.now()
        source_filename = self.result.get("source_filename", "Unknown")
        processing_time = self.result.get("processing_time")
        
        metadata_data = [
            ["Report Generated", now.strftime('%B %d, %Y at %H:%M')],
            ["Source Image", source_filename],
        ]
        
        if processing_time:
            metadata_data.append(["Processing Time", f"{processing_time:.2f} seconds"])
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.COLORS['light_gray']),
            ('TEXTCOLOR', (0, 0), (0, -1), self.COLORS['dark']),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['primary_blue']),
        ]))
        self.elements.append(metadata_table)
        
        self.elements.append(Spacer(1, 15))
        
        # Horizontal line
        self.elements.append(HRFlowable(
            width="100%", thickness=1, color=self.COLORS['primary_blue']
        ))
        self.elements.append(Spacer(1, 20))
        
    def _add_images(self):
        """Add original and segmentation images."""
        self.elements.append(Paragraph(
            "1. Image Analysis",
            self.styles['SectionHeading']
        ))
        
        # Create temporary files for images
        temp_dir = tempfile.gettempdir()
        
        # Original image
        orig_path = os.path.join(temp_dir, "report_original.jpg")
        cv2.imwrite(orig_path, self.image)
        
        # Segmentation result - use pre-stored colored segmentation if available
        seg_colored = self.result.get("colored_segmentation")
        
        if seg_colored is None:
            # Fallback: generate from labels
            labels = self.result.get("labels")
            if labels is not None:
                from src.visualization import create_colored_segmentation
                seg_colored = create_colored_segmentation(labels)
        
        if seg_colored is not None:
            seg_path = os.path.join(temp_dir, "report_segmentation.png")
            cv2.imwrite(seg_path, seg_colored)
        else:
            seg_path = None
        
        # Create side-by-side table for images
        img_width = 3 * inch
        img_height = 2.5 * inch
        
        if seg_path and os.path.exists(seg_path):
            img_data = [
                [
                    Image(orig_path, width=img_width, height=img_height),
                    Image(seg_path, width=img_width, height=img_height)
                ],
                [
                    Paragraph("(a) Original Image", self.styles['ReportCaption']),
                    Paragraph("(b) Segmentation Result", self.styles['ReportCaption'])
                ]
            ]
        else:
            img_data = [
                [Image(orig_path, width=img_width, height=img_height)],
                [Paragraph("Original Image", self.styles['ReportCaption'])]
            ]
            
        img_table = Table(img_data, colWidths=[img_width + 20] * len(img_data[0]))
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        self.elements.append(img_table)
        self.elements.append(Spacer(1, 10))
        
        # Image info
        h, w = self.image.shape[:2]
        self.elements.append(Paragraph(
            f"<b>Image Dimensions:</b> {w} × {h} pixels",
            self.styles['ReportBody']
        ))
        
    def _add_layer_analysis(self):
        """Add layer analysis table."""
        self.elements.append(Paragraph(
            "2. Layer Classification Results",
            self.styles['SectionHeading']
        ))
        
        classification = self.result.get("classification", {})
        
        # Main result
        layer_name = classification.get("layer_name", "Unknown")
        confidence = classification.get("confidence", 0)
        material = classification.get("material", "N/A")
        method = classification.get("method", self.mode)
        
        self.elements.append(Paragraph(
            f"<b>Identified Layer:</b> {layer_name}",
            self.styles['ReportBody']
        ))
        self.elements.append(Paragraph(
            f"<b>Confidence:</b> {confidence:.1%}",
            self.styles['ReportBody']
        ))
        self.elements.append(Paragraph(
            f"<b>Material Type:</b> {material}",
            self.styles['ReportBody']
        ))
        self.elements.append(Paragraph(
            f"<b>Analysis Method:</b> {method}",
            self.styles['ReportBody']
        ))
        
        # Layer distribution table (if available)
        layer_dist = self.result.get("layer_distribution", {})
        labels = self.result.get("labels")
        
        if labels is not None:
            self.elements.append(Spacer(1, 10))
            self.elements.append(Paragraph(
                "2.1 Layer Distribution",
                self.styles['SubHeading']
            ))
            
            # Calculate distribution from labels
            unique, counts = np.unique(labels, return_counts=True)
            total = labels.size
            
            table_data = [["Layer", "Coverage", "Material"]]
            for layer_num, count in zip(unique, counts):
                pct = count / total
                layer_info = ROAD_LAYERS.get(int(layer_num), {})
                name = layer_info.get("name", f"Layer {layer_num}")
                mat = layer_info.get("material", "Unknown")
                table_data.append([name, f"{pct:.1%}", mat])
            
            table = Table(table_data, colWidths=[2.5*inch, 1.2*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['primary_blue']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), self.COLORS['light_gray']),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS['light_gray']]),
            ]))
            self.elements.append(table)
            
    def _add_texture_features(self):
        """Add texture features table."""
        features = self.result.get("features", {})
        
        if not features:
            return
            
        self.elements.append(Paragraph(
            "3. Texture Feature Analysis",
            self.styles['SectionHeading']
        ))
        
        # GLCM features
        glcm = features.get("glcm", {})
        if glcm:
            self.elements.append(Paragraph(
                "3.1 Gray Level Co-occurrence Matrix (GLCM)",
                self.styles['SubHeading']
            ))
            
            table_data = [["Property", "Value", "Interpretation"]]
            
            contrast = glcm.get("contrast", 0)
            table_data.append(["Contrast", f"{contrast:.4f}", 
                "High = rough texture" if contrast > 500 else "Low = smooth texture"])
            
            energy = glcm.get("energy", 0)
            table_data.append(["Energy", f"{energy:.4f}",
                "High = uniform" if energy > 0.1 else "Low = varied"])
            
            homogeneity = glcm.get("homogeneity", 0)
            table_data.append(["Homogeneity", f"{homogeneity:.4f}",
                "High = fine texture" if homogeneity > 0.5 else "Low = coarse texture"])
            
            correlation = glcm.get("correlation", 0)
            table_data.append(["Correlation", f"{correlation:.4f}",
                "High = structured" if correlation > 0.8 else "Low = random"])
            
            table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['primary_green']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS['light_gray']]),
            ]))
            self.elements.append(table)
            
    def _add_settings(self):
        """Add analysis settings used."""
        self.elements.append(Paragraph(
            "4. Analysis Parameters",
            self.styles['SectionHeading']
        ))
        
        # Mode name mapping for better display
        mode_names = {
            "classical": "Classical (Texture Analysis)",
            "vlm": "VLM (GLM-4.6V AI Vision)",
            "deep_learning": "Deep Learning (CNN Classifier)",
            "hybrid": "Hybrid (Classical + AI)"
        }
        
        # Create settings table
        table_data = [["Parameter", "Value"]]
        
        # Analysis mode (always shown first)
        table_data.append(["Analysis Mode", mode_names.get(self.mode, self.mode.replace("_", " ").title())])
        
        if self.mode == "classical":
            # Preprocessing settings
            table_data.append(["", ""])  # Separator
            table_data.append(["— PREPROCESSING —", ""])
            table_data.append(["Noise Filter", self.params.get("noise_filter", "median")])
            table_data.append(["Kernel Size", str(self.params.get("kernel_size", 3))])
            table_data.append(["Contrast Method", self.params.get("contrast_method", "clahe")])
            if self.params.get("contrast_method") == "clahe":
                table_data.append(["CLAHE Clip Limit", str(self.params.get("clahe_clip", 2.0))])
            
            # Feature extraction
            table_data.append(["", ""])
            table_data.append(["— FEATURE EXTRACTION —", ""])
            table_data.append(["GLCM Features", "Yes" if self.params.get("use_glcm", True) else "No"])
            table_data.append(["LBP Features", "Yes" if self.params.get("use_lbp", True) else "No"])
            table_data.append(["Gabor Features", "Yes" if self.params.get("use_gabor", False) else "No"])
            
            # Segmentation
            table_data.append(["", ""])
            table_data.append(["— SEGMENTATION —", ""])
            table_data.append(["Segmentation Method", self.params.get("segmentation_method", "K-Means")])
            table_data.append(["Number of Clusters", str(self.params.get("n_clusters", 5))])
            
            # Morphology
            table_data.append(["", ""])
            table_data.append(["— POST-PROCESSING —", ""])
            table_data.append(["Morphological Ops", "Yes" if self.params.get("use_morphology") else "No"])
            table_data.append(["Fill Holes", "Yes" if self.params.get("fill_holes") else "No"])
            
        elif self.mode == "vlm":
            # VLM-specific settings
            table_data.append(["", ""])
            table_data.append(["— VLM SETTINGS —", ""])
            table_data.append(["Analysis Type", self.params.get("vlm_analysis_type", "Layer ID")])
            table_data.append(["Temperature", str(self.params.get("vlm_temperature", 0.3))])
            table_data.append(["Model", "GLM-4.6V"])
            
            # Output options
            table_data.append(["", ""])
            table_data.append(["— OUTPUT OPTIONS —", ""])
            table_data.append(["Include Layer Name", "Yes" if self.params.get("vlm_include_layer", True) else "No"])
            table_data.append(["Include Confidence", "Yes" if self.params.get("vlm_include_confidence", True) else "No"])
            table_data.append(["Include Material", "Yes" if self.params.get("vlm_include_material", True) else "No"])
            table_data.append(["Include Texture", "Yes" if self.params.get("vlm_include_texture", True) else "No"])
            table_data.append(["Include Recommendations", "Yes" if self.params.get("vlm_include_recommendations", True) else "No"])
            
        elif self.mode == "deep_learning":
            # Deep Learning settings
            table_data.append(["", ""])
            table_data.append(["— MODEL SETTINGS —", ""])
            table_data.append(["Backbone Network", self.params.get("dl_backbone", "ResNet-101")])
            table_data.append(["Use Pretrained", "Yes" if self.params.get("dl_pretrained", True) else "No"])
            table_data.append(["Device", self.params.get("dl_device", "CPU")])
            
            # Inference settings
            table_data.append(["", ""])
            table_data.append(["— INFERENCE SETTINGS —", ""])
            table_data.append(["Input Resolution", self.params.get("dl_resolution", "512x512")])
            table_data.append(["Confidence Threshold", f"{self.params.get('dl_confidence_threshold', 0.1):.0%}"])
            
        elif self.mode == "hybrid":
            # Hybrid mode settings
            classical_weight = self.params.get("classical_weight", 0.7)
            ai_weight = 1 - classical_weight
            
            table_data.append(["", ""])
            table_data.append(["— WEIGHTING —", ""])
            table_data.append(["Classical Weight", f"{classical_weight:.0%}"])
            table_data.append(["AI Weight", f"{ai_weight:.0%}"])
            table_data.append(["VLM Validation", "Enabled" if self.params.get("hybrid_vlm_validation", True) else "Disabled"])
            table_data.append(["Conflict Rule", self.params.get("hybrid_conflict_rule", "Weighted Average")])
            
            # Classical settings used
            table_data.append(["", ""])
            table_data.append(["— CLASSICAL SETTINGS —", ""])
            table_data.append(["Noise Filter", self.params.get("noise_filter", "median")])
            table_data.append(["Contrast Method", self.params.get("contrast_method", "clahe")])
            table_data.append(["Segmentation", self.params.get("segmentation_method", "K-Means")])
            table_data.append(["Clusters", str(self.params.get("n_clusters", 5))])
            
            # Features used
            table_data.append(["", ""])
            table_data.append(["— FEATURES —", ""])
            table_data.append(["GLCM", "Yes" if self.params.get("use_glcm", True) else "No"])
            table_data.append(["LBP", "Yes" if self.params.get("use_lbp", True) else "No"])
            table_data.append(["Gabor", "Yes" if self.params.get("use_gabor", False) else "No"])
            
        table = Table(table_data, colWidths=[2.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['dark']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, self.COLORS['light_gray']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS['light_gray']]),
        ]))
        self.elements.append(table)
        
    def _add_chart(self):
        """Add layer distribution pie chart."""
        labels_array = self.result.get("labels")
        if labels_array is None:
            return
            
        self.elements.append(Paragraph(
            "5. Layer Distribution Visualization",
            self.styles['SectionHeading']
        ))
        
        # Calculate distribution
        unique, counts = np.unique(labels_array, return_counts=True)
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(5, 4))
        
        layer_names = []
        layer_colors = []
        for layer_num in unique:
            info = ROAD_LAYERS.get(int(layer_num), {})
            layer_names.append(info.get("name", f"Layer {layer_num}"))
            hex_color = info.get("hex_color", "#888888")
            layer_colors.append(hex_color)
        
        percentages = counts / counts.sum() * 100
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=layer_names,
            colors=layer_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75
        )
        
        ax.set_title("Layer Distribution", fontsize=12, fontweight='bold')
        
        # Save chart to temp file
        temp_dir = tempfile.gettempdir()
        chart_path = os.path.join(temp_dir, "report_chart.png")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Add to report
        self.elements.append(Image(chart_path, width=4*inch, height=3.2*inch))
        self.elements.append(Paragraph(
            "Figure 1: Pie chart showing the distribution of identified road layers",
            self.styles['ReportCaption']
        ))
        
    def _add_vlm_analysis(self):
        """Add VLM analysis if available."""
        vlm_result = self.result.get("vlm_result") or self.result.get("classification", {})
        reasoning = vlm_result.get("reasoning") or vlm_result.get("analysis", "")
        
        if not reasoning:
            return
            
        self.elements.append(Paragraph(
            "6. AI Vision Analysis",
            self.styles['SectionHeading']
        ))
        
        self.elements.append(Paragraph(
            "<b>AI Interpretation:</b>",
            self.styles['ReportBody']
        ))
        
        self.elements.append(Paragraph(
            reasoning,
            self.styles['ReportBody']
        ))
        
    def _add_discussion(self):
        """Add auto-generated discussion section."""
        self.elements.append(Paragraph(
            "7. Discussion",
            self.styles['SectionHeading']
        ))
        
        classification = self.result.get("classification", {})
        layer_name = classification.get("layer_name", "Unknown")
        confidence = classification.get("confidence", 0)
        material = classification.get("material", "")
        
        # Generate discussion based on results
        discussion = []
        
        discussion.append(
            f"The analysis identified the primary road layer as <b>{layer_name}</b> "
            f"with a confidence level of <b>{confidence:.1%}</b>. "
        )
        
        if confidence >= 0.8:
            discussion.append(
                "This high confidence suggests the texture patterns and visual characteristics "
                "closely match the expected properties of this layer type."
            )
        elif confidence >= 0.5:
            discussion.append(
                "The moderate confidence level indicates some ambiguity in the classification, "
                "which may be due to transitional zones, mixed materials, or image quality factors."
            )
        else:
            discussion.append(
                "The lower confidence suggests the image may contain mixed layers, unusual textures, "
                "or may benefit from additional analysis modes for verification."
            )
        
        if material and material != "N/A":
            discussion.append(
                f" The identified material type is <b>{material}</b>, "
                "which is consistent with standard road construction specifications."
            )
        
        # Mode-specific discussion
        if self.mode == "hybrid":
            classical_res = self.result.get("classical_result", {})
            vlm_res = self.result.get("vlm_result", {})
            if classical_res and vlm_res:
                discussion.append(
                    f"<br/><br/>The hybrid analysis combined classical texture analysis "
                    f"(identified: {classical_res.get('layer_name', 'N/A')}, {classical_res.get('confidence', 0):.0%}) "
                    f"with AI vision analysis (identified: {vlm_res.get('layer_name', 'N/A')}, {vlm_res.get('confidence', 0):.0%})."
                )
        
        self.elements.append(Paragraph(
            " ".join(discussion),
            self.styles['ReportBody']
        ))
        
    def _add_conclusion(self):
        """Add conclusion section."""
        self.elements.append(Paragraph(
            "8. Conclusion",
            self.styles['SectionHeading']
        ))
        
        classification = self.result.get("classification", {})
        layer_name = classification.get("layer_name", "Unknown")
        confidence = classification.get("confidence", 0)
        
        conclusion = (
            f"Based on the comprehensive image analysis using the {self.mode.replace('_', ' ')} method, "
            f"the road surface layer has been identified as <b>{layer_name}</b> "
            f"with a confidence of <b>{confidence:.1%}</b>. "
            f"This classification is based on texture features, color patterns, and structural characteristics "
            f"extracted from the input image. The results can be used for road condition assessment, "
            f"maintenance planning, and quality control purposes."
        )
        
        self.elements.append(Paragraph(conclusion, self.styles['ReportBody']))
        
    def _add_footer(self):
        """Add footer with branding."""
        self.elements.append(Spacer(1, 30))
        self.elements.append(HRFlowable(
            width="100%", thickness=1, color=self.COLORS['primary_green']
        ))
        self.elements.append(Spacer(1, 10))
        self.elements.append(Paragraph(
            "<b>ClaRity Group</b> | Road Surface Layer Analyzer v1.0.0",
            self.styles['ReportCaption']
        ))
        self.elements.append(Paragraph(
            "<i>\"Building the future with ClaRity\"</i>",
            self.styles['ReportCaption']
        ))


def generate_report(result: Dict, image: np.ndarray, mode: str, params: Dict, output_path: str) -> bool:
    """
    Convenience function to generate a PDF report.
    
    Args:
        result: Analysis result
        image: Original image
        mode: Analysis mode
        params: Parameters used
        output_path: Where to save the PDF
        
    Returns:
        True if successful
    """
    generator = ReportGenerator(result, image, mode, params)
    return generator.generate(output_path)

