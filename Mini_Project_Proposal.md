# CSC566 MINI PROJECT PROPOSAL

## 1. Project Title

**Automated Road Surface Layer Identification and Analysis using Multi-Method Image Processing Method**

---

## 2. Introduction

This project implements an **automated multi-method system** for analyzing and classifying road construction layers using Google Earth Pro aerial satellite images. The application addresses the complexity of road infrastructure monitoring by identifying **five distinct construction layers**, ranging from the foundational subgrade to the final surface course.

Built with **Python 3.10+** and a professional **PyQt5 GUI** with dark theme, the system integrates:

- **Classical Image Processing** (GLCM, LBP texture analysis with K-Means/SLIC/Watershed segmentation)
- **Deep Learning** (DeepLabv3+ semantic segmentation with ResNet backbone)
- **Vision Language Models** (GLM-4.6V for AI-powered layer identification)
- **Instance Segmentation** (YOLOv11 for real-time object detection with Live Preview)
- **Hybrid Analysis** (Classical + VLM cross-validation with configurable conflict resolution)

The system leverages **CUDA GPU acceleration** for high-performance inference and provides comprehensive visualization with overlays, contours, and detailed result dialogs. By combining these advanced technologies with an intuitive interface, the application serves as a powerful assistive tool for construction assessment and road health monitoring.

---

## 3. Problem Statement & Objectives

### Problem Statement

Accurate identification of road construction layers is essential for quality control in civil engineering. However, manual analysis of aerial imagery presents significant challenges:

1. **Time-Intensive**: Visual inspection of satellite images requires extensive human effort
2. **Prone to Error**: Subjective interpretation leads to inconsistent classifications
3. **Limited Scalability**: On-site surveys are expensive and impractical for large-scale monitoring
4. **Complex Textures**: Different road materials exhibit subtle visual and texture characteristics that require sophisticated analysis

### Objectives

| No. | Objective |
|:---:|-----------|
| 1 | To develop an automated pipeline for classifying **five road layers**: Subgrade, Subbase Course, Base Course, Binder Course, and Surface Course |
| 2 | To implement **texture-based feature extraction** using GLCM (Contrast, Energy, Homogeneity, Correlation) and Local Binary Patterns (LBP) |
| 3 | To provide a **multi-mode analysis interface** featuring Classical segmentation, Deep Learning (DeepLabv3+), VLM Analysis (GLM-4.6V), Hybrid validation, and YOLOv11 Instance Segmentation |
| 4 | To deliver a **professional PyQt5 GUI** with real-time processing, visualization overlays, result dialogs, and PDF report generation |
| 5 | To implement **CUDA GPU acceleration** for efficient processing of high-resolution satellite imagery |

---

## 4. Project Scope & Significance

### Scope

The scope of this project focuses on:

- **Layer Classification**: Segmentation and classification of five specific road layers found in a standard road cross-section
- **Image Source**: Analysis of aerial satellite imagery from Google Earth Pro
- **Processing Methods**: Implementation of 5 distinct analysis modes with configurable parameters
- **GPU Acceleration**: CUDA-enabled processing for DeepLabv3+ and YOLOv11 models
- **User Interface**: Professional PyQt5 application with dark theme, drag-drop support, and live preview capabilities
- **Output Generation**: Color-coded visualizations, statistical reports, and exportable PDF documents

### Significance

| Aspect | Impact |
|--------|--------|
| **Civil Engineering** | Assists engineers in remote road inspection, reducing the need for costly on-site surveys |
| **Urban Planning** | Enables rapid infrastructure health assessments across large areas |
| **Quality Control** | Provides objective, consistent layer classification with confidence metrics |
| **Research** | Demonstrates integration of classical and modern AI techniques for practical image processing applications |
| **Education** | Serves as a comprehensive example of multi-method image analysis for academic purposes |

---

## 5. Proposed Method/Modelling Approach

The system utilizes a **comprehensive multi-method approach** combining five distinct methodologies:

### 5.1 Preprocessing Pipeline

| Stage | Technique | Purpose |
|-------|-----------|---------|
| **Noise Reduction** | Median, Gaussian, Bilateral filtering | Remove sensor noise while preserving edges |
| **Contrast Enhancement** | CLAHE (Adaptive Histogram Equalization) | Improve local contrast for better feature extraction |
| **Sharpening** | Unsharp mask | Enhance texture details |
| **Color Conversion** | RGB → Grayscale, HSV, Lab | Prepare for feature extraction |

### 5.2 Analysis Modes

#### Mode 1: Classical Analysis (Texture-Based)

```
Image → Preprocessing → Feature Extraction → Segmentation → Morphology → Classification
                              ↓                    ↓
                        GLCM + LBP           K-Means/SLIC/Watershed
```

- **Features**: GLCM (Contrast, Energy, Homogeneity, Correlation, Entropy), LBP histograms, Gabor filters (optional)
- **Segmentation**: K-Means clustering, SLIC Superpixels, Watershed algorithm
- **Post-Processing**: Morphological operations (opening, closing), hole filling
- **Output**: Detailed result dialogs showing Image Segmentation steps and Texture Feature Extraction

#### Mode 2: Deep Learning (DeepLabv3+)

```
Image → Resize → DeepLabv3+ (ResNet backbone) → Semantic Labels → Visualization Overlay
```

- **Architecture**: DeepLabv3+ with ResNet-50/101 backbone
- **Pre-training**: ImageNet weights with optional fine-tuning
- **Inference**: CUDA GPU acceleration
- **Visualization**: Semi-transparent overlay on original image with instance contours and info banner

#### Mode 3: VLM Analysis (GLM-4.6V)

```
Image → ZenMux API → GLM-4.6V → Structured Response → Enhanced Visualization
```

- **Model**: GLM-4.6V Vision Language Model via ZenMux API
- **Analysis Types**: Layer ID, Detailed Analysis, Quick Scan
- **Output**: Layer identification, confidence, material description, recommendations
- **Visualization**: Muted image with green bounding box, edge overlay, and info banner

#### Mode 4: Hybrid (Classical + VLM)

```
Image → Classical Analysis ─┐
                            ├→ Conflict Resolution → Final Classification
Image → VLM Analysis ───────┘
```

- **Combination**: Runs both Classical and VLM analysis
- **Conflict Rules**: Higher Confidence Wins, Classical Priority, VLM Priority, Weighted Average
- **Weighting**: Configurable Classical/AI balance (0-100%)

#### Mode 5: YOLOv11 Instance Segmentation

```
Image/Live Feed → YOLOv11-seg → Instance Masks → Colored Visualization
```

- **Model**: YOLOv11 with segmentation head
- **Features**: Real-time inference, instance-level masks, confidence thresholding
- **Live Preview**: Window capture for real-time road analysis from any application
- **Visualization**: Colored masks with labels and confidence scores

### 5.3 Post-Processing

| Operation | Purpose |
|-----------|---------|
| **Morphological Opening** | Remove small noise regions |
| **Morphological Closing** | Fill small holes in regions |
| **Connected Components** | Identify and filter isolated regions |
| **Contour Detection** | Draw instance boundaries for visualization |

---

## 6. Performance Metrics

The effectiveness of the system is evaluated through specialized metrics:

### 6.1 Texture Metrics (Classical Mode)

| Metric | Description | Road Layer Correlation |
|--------|-------------|------------------------|
| **GLCM Contrast** | Measures local intensity variation | High → Coarse aggregate layers |
| **GLCM Energy** | Measures texture uniformity | High → Smooth surface course |
| **GLCM Homogeneity** | Measures closeness to diagonal | High → Uniform materials |
| **LBP Uniformity** | Measures pattern consistency | Differentiates layer textures |

### 6.2 Classification Metrics

| Metric | Description |
|--------|-------------|
| **Confidence Percentage** | Model certainty for each classified layer |
| **Layer Distribution** | Percentage of image covered by each layer |
| **Dominant Layer** | Primary layer identification with highest coverage |

### 6.3 System Performance Metrics

| Metric | Target |
|--------|--------|
| **Processing Time** | < 5 seconds for Classical, < 2 seconds for CNN (GPU) |
| **GPU Memory** | < 4GB for inference |
| **GUI Responsiveness** | Non-blocking analysis with progress feedback |

### 6.4 Deep Learning Metrics

| Metric | Description |
|--------|-------------|
| **Semantic IoU** | Intersection over Union for segmentation accuracy |
| **Instance Detection** | Precision and recall for YOLOv11 detections |
| **Inference FPS** | Frames per second for live preview mode |
