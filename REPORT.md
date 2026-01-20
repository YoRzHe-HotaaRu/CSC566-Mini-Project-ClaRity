# CSC566 MINI PROJECT REPORT

**Project Title:** Automated Road Surface Layer Identification and Analysis using Multi-Method Image Processing Method

**Course:** CSC566 - Image Processing  
**Group:** ClaRity Group (A4CDCS2306A)  
**Lecturer:** Ts. ZAABA BIN AHMAD

| No. | Name | Student ID |
|:---:|------|:----------:|
| 1 | AMIR HAFIZI BIN MUSA | 2024745815 |
| 2 | AQIL IMRAN BIN NORHIDZAM | 2024779269 |
| 3 | MUHAMMAD 'ADLI BIN MOHD ALI | 2024974573 |
| 4 | NIK MUHAMMAD HAZIQ BIN NIK HASNI | 2024741073 |

---

## 1. Introduction

This project implements an automated system for analyzing road construction layers from aerial satellite images. The system identifies **5 distinct road layers**:

1. **Subgrade** - Foundation soil layer
2. **Subbase Course** - Coarse crushed aggregate
3. **Base Course** - Finer crushed aggregate
4. **Binder Course** - Premix asphalt with aggregate
5. **Surface Course** - Smooth wearing surface

The application combines multiple analysis methods:
- **Classical Image Processing** - Texture analysis using GLCM and LBP
- **Deep Learning** - DeepLabv3+ semantic segmentation
- **Vision Language Model** - GLM-4.6V AI analysis
- **YOLOv11** - Instance segmentation with live preview
- **Hybrid** - Combined Classical + VLM validation

The system is built with Python and PyQt5, featuring a professional GUI with CUDA GPU acceleration.

---

## 2. Objectives

The objectives of this project are:

1. To develop an automated pipeline for classifying five road construction layers
2. To implement texture-based feature extraction using GLCM and LBP
3. To provide multiple analysis modes (Classical, CNN, VLM, Hybrid, YOLO)
4. To create a user-friendly GUI application with visualization features
5. To enable GPU-accelerated processing for efficient analysis

---

## 3. Data Collection

### Image Source
- **Google Earth Pro** aerial satellite images of road construction sites
- Resolution: High-resolution satellite imagery
- Format: JPEG, PNG images

### Dataset Organization
```
data/
├── subgrade/          # Layer 1 images
├── subbase/           # Layer 2 images
├── base_course/       # Layer 3 images
├── binder_course/     # Layer 4 images
└── surface_course/    # Layer 5 images
```

### Image Characteristics
| Layer | Visual Features |
|-------|-----------------|
| Subgrade | Brown earth tones, irregular texture |
| Subbase | Visible coarse stones, rough surface |
| Base Course | Uniform aggregate, medium texture |
| Binder Course | Dark with visible stones |
| Surface Course | Smooth, uniform dark surface |

---

## 4. Flowchart

### Main Processing Flow

```
┌─────────────────┐
│  Load Image     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Noise Filter │
│  - CLAHE        │
│  - Sharpen      │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Select Mode    │
└────────┬────────┘
         │
    ┌────┴────┬────────┬──────────┬──────────┐
    ▼         ▼        ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌────────┐ ┌───────┐
│Classic│ │  CNN  │ │  VLM  │ │ Hybrid │ │ YOLO  │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬────┘ └───┬───┘
    │         │        │          │          │
    └────┬────┴────────┴──────────┴──────────┘
         ▼
┌─────────────────┐
│ Classification  │
│ - Layer ID      │
│ - Confidence    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Visualization  │
│  - Overlay      │
│  - Results      │
└─────────────────┘
```

### Classical Mode Flow

```
Image → Grayscale → GLCM Features → K-Means Segmentation → Morphology → Classification
                  → LBP Features  ↗
```

### Deep Learning Mode Flow

```
Image → Resize (512x512) → DeepLabv3+ → Labels → Overlay on Original → Display
```

---

## 5. Results of Prototype

### Feature Extraction Results

The system successfully extracts texture features:

| Feature | Description | Use |
|---------|-------------|-----|
| GLCM Contrast | Intensity variation | Detect rough surfaces |
| GLCM Energy | Uniformity measure | Detect smooth surfaces |
| GLCM Homogeneity | Local similarity | Measure texture consistency |
| LBP Histogram | Local patterns | Capture micro-textures |

### Classification Performance

| Mode | Strengths |
|------|-----------|
| Classical | Fast processing, good for clear textures |
| CNN | Handles complex images, semantic understanding |
| VLM | AI-powered analysis with explanations |
| Hybrid | Highest accuracy with cross-validation |
| YOLO | Real-time detection with instance masks |

### Processing Speed

| Mode | Approximate Time |
|------|------------------|
| Classical | 2-4 seconds |
| CNN (GPU) | 1-2 seconds |
| VLM | 3-5 seconds (API) |
| YOLO (GPU) | < 1 second |

---

## 6. System Prototype in GUI

### Main Interface Components

```
┌─────────────────────────────────────────────────────────────┐
│  Road Surface Layer Analyzer - ClaRity                      │
├─────────────────────────────────────────────────────────────┤
│ ┌───────────────┐  ┌───────────────────────────────────┐    │
│ │ Original      │  │ Segmentation Result               │    │
│ │ Image         │  │ (Overlay with contours)           │    │
│ │               │  │                                   │    │
│ └───────────────┘  └───────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Mode: [Classical] [CNN] [VLM] [Hybrid] [YOLOv11]           │
├─────────────────────────────────────────────────────────────┤
│ Settings Panel (changes based on selected mode)             │
├─────────────────────────────────────────────────────────────┤
│ [Load Image]  [Analyze]  [Export]                           │
├─────────────────────────────────────────────────────────────┤
│ Results:                                                    │
│ - Detected Layer: Surface Course                            │
│ - Confidence: 94.2%                                         │
│ - Material: Premix asphalt (smooth)                         │
├─────────────────────────────────────────────────────────────┤
│ Legend: [Subgrade] [Subbase] [Base] [Binder] [✓Surface]    │
└─────────────────────────────────────────────────────────────┘
```

### GUI Features

- **Dark Theme** - Professional appearance
- **Drag & Drop** - Easy image loading
- **Mode Selection** - 5 analysis modes
- **Real-time Progress** - Progress bar during analysis
- **Interactive Legend** - Shows detected layers
- **PDF Export** - Save analysis reports

---

## 7. Sample Input Output

### Example 1: Surface Course Detection

**Input:** Aerial image of paved road surface

**Output:**
- Detected Layer: **Surface Course (Layer 5)**
- Confidence: **94.2%**
- Material: Premix asphalt (smooth)
- Visualization: Overlay with dark blue-gray coloring

### Example 2: Base Course Detection

**Input:** Image showing aggregate layer during construction

**Output:**
- Detected Layer: **Base Course (Layer 3)**
- Confidence: **87.5%**
- Material: Crushed aggregate (finer)
- Visualization: Overlay with light pinkish-gray coloring

### Layer Color Coding

| Layer | Color | Description |
|-------|-------|-------------|
| 1 - Subgrade | Brown | Foundation soil |
| 2 - Subbase | Tan/Beige | Coarse aggregate |
| 3 - Base | Light Pink-Gray | Fine aggregate |
| 4 - Binder | Orange-Brown | Asphalt mix |
| 5 - Surface | Dark Blue-Gray | Wearing course |

---

## 8. Source Code

### Project Structure

```
CSC566-Mini-Project-ClaRity/
├── src/                      # Core modules
│   ├── config.py             # Configuration
│   ├── preprocessing.py      # Image preprocessing
│   ├── texture_features.py   # GLCM, LBP extraction
│   ├── segmentation.py       # K-Means, SLIC, Watershed
│   ├── classification.py     # Layer classification
│   ├── deep_learning.py      # DeepLabv3+ model
│   ├── vlm_analyzer.py       # GLM-4.6V integration
│   └── yolo_analyzer.py      # YOLOv11 model
│
├── gui/                      # GUI application
│   ├── main_window.py        # Main application
│   ├── splash_screen.py      # Startup screen
│   └── classical_results.py  # Result dialogs
│
└── run.bat                   # Launch script
```

### Key Code Snippets

#### GLCM Feature Extraction
```python
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return {"contrast": contrast, "energy": energy, "homogeneity": homogeneity}
```

#### K-Means Segmentation
```python
from sklearn.cluster import KMeans

def kmeans_segment(image, n_clusters=5):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image.shape[:2])
```

#### DeepLabv3+ Inference
```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(encoder_name="resnet101", classes=5)
with torch.no_grad():
    output = model(input_tensor)
    labels = torch.argmax(output, dim=1)
```

### How to Run

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Run application
python -m gui.main_window

# Or use the batch file
run.bat
```

---

## 9. Conclusion

This project successfully developed an automated road surface layer analysis system with the following achievements:

### Accomplishments

1. **Multi-Method Analysis** - Implemented 5 different analysis modes to handle various image types
2. **Texture Feature Extraction** - Successfully extracted GLCM and LBP features for layer classification
3. **Deep Learning Integration** - Integrated DeepLabv3+ for semantic segmentation with GPU acceleration
4. **AI-Powered Analysis** - Incorporated GLM-4.6V VLM for intelligent layer identification
5. **Professional GUI** - Created a user-friendly PyQt5 application with dark theme and modern features
6. **Real-time Processing** - Implemented YOLOv11 with live preview capability

### Limitations

- Requires high-resolution images for accurate classification
- VLM mode depends on external API availability
- Deep learning models require significant GPU memory

### Future Improvements

- Train custom models on larger road layer datasets
- Add support for video analysis
- Implement batch processing for multiple images
- Add more detailed texture analysis features

---

*ClaRity Group | CSC566 Image Processing | UiTM Cawangan Perak Kampus Tapah*
