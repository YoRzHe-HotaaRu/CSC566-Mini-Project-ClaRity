# Continue Conversation: Road Surface Layer Analyzer

> **Use this file to provide context to a new AI coding agent session.**
> Copy the contents of this file and paste it as your first message to continue work on this project.

---

## 🎯 Project Overview

**Project Title**: Automated Road Surface Layers Analysis using Texture-Based Image Segmentation

**Course**: CSC566 - Image Processing (Mini Project 30%)

**Group**: ClaRity Group (A4CDCS2306A)

**Lecturer**: Ts. ZAABA BIN AHMAD

### Group Members
1. AMIR HAFIZI BIN MUSA (2024745815)
2. AQIL IMRAN BIN NORHIDZAM (2024779269)
3. MUHAMMAD 'ADLI BIN MOHD ALI (2024974573)
4. NIK MUHAMMAD HAZIQ BIN NIK HASNI (2024741073)

---

## 📋 Project Goal

Build an automated system to classify **5 distinct road construction layers** from **Google Earth Pro aerial satellite images** using:
1. **Classical image processing** (GLCM, LBP, Gabor, K-Means segmentation)
2. **Deep Learning** (DeepLabv3+ semantic segmentation with CUDA)
3. **Vision Language Model** (GLM-4.6V via ZenMux API)
4. **PyQt5 GUI** with 4 analysis modes

### 5 Road Layers (Bottom to Top)
| Layer | Name | Material |
|:-----:|------|----------|
| 1 | Subgrade | In-site soil/backfill |
| 2 | Subbase Course | Crushed aggregate (coarse) |
| 3 | Base Course | Crushed aggregate (fine) |
| 4 | Binder Course | Premix asphalt |
| 5 | Surface Course | Premix asphalt (smooth) |

---

## ✅ What Has Been Completed

### All Code Implementation is DONE

| Category | Files Created |
|----------|---------------|
| **Source Modules** | `src/__init__.py`, `config.py`, `preprocessing.py`, `texture_features.py`, `segmentation.py`, `morphology.py`, `descriptors.py`, `classification.py`, `deep_learning.py`, `vlm_analyzer.py`, `visualization.py` |
| **GUI** | `gui/__init__.py`, `gui/main_window.py` |
| **Tests** | `tests/conftest.py`, `test_preprocessing.py`, `test_texture_features.py`, `test_segmentation.py`, `test_morphology.py`, `test_descriptors.py`, `test_classification.py`, `test_deep_learning.py`, `test_vlm_analyzer.py`, `test_integration.py`, `test_performance.py` |
| **Config** | `.gitignore`, `.env`, `.env.example`, `requirements.txt`, `README.md` |
| **Scripts** | `setup.bat`, `run_gui.bat`, `run_tests.bat`, `install_pytorch_cuda.bat` |
| **Data Folders** | `data/subgrade/`, `data/subbase/`, `data/base_course/`, `data/binder_course/`, `data/surface_course/`, `results/` |

### Key Technical Details

- **DeepLabv3+**: Uses ResNet-101 backbone, CUDA-accelerated
- **VLM API**: GLM-4.6V via ZenMux (`https://zenmux.ai/api/v1`)
- **API Key**: Stored in `.env` as `ZENMUX_API_KEY`
- **GUI**: PyQt5 with dark theme, 4 analysis modes, threaded processing
- **Tests**: 100+ test cases with pytest, mocked API calls

---

## 🚀 What Needs To Be Done Next

### Immediate (On Laptop with RTX 4050)

1. **Install Dependencies**
   ```batch
   setup.bat
   ```

2. **Install PyTorch with CUDA 12.x** (for RTX 4050)
   ```batch
   install_pytorch_cuda.bat
   ```
   Or manually:
   ```batch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify CUDA**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show RTX 4050
   ```

4. **Run Tests**
   ```batch
   run_tests.bat
   ```

5. **Launch GUI**
   ```batch
   run_gui.bat
   ```

### Dataset Collection

- Collect Google Earth Pro aerial images of road construction sites
- Organize into 5 folders: `data/subgrade/`, `data/subbase/`, `data/base_course/`, `data/binder_course/`, `data/surface_course/`
- Recommended: 20-50 images per layer

### Documentation (Later)

- [ ] Methodology report
- [ ] Results tables
- [ ] Discussion and conclusion
- [ ] Academic paper
- [ ] Presentation slides (5-minute demo)

---

## 🔧 Project Structure

```
CSC566-Mini-Project/
├── .env                    # API keys (ZENMUX_API_KEY)
├── .env.example            # Template
├── .gitignore
├── requirements.txt
├── README.md
├── setup.bat               # Full setup script
├── run_gui.bat             # Launch GUI
├── run_tests.bat           # Run pytest
├── install_pytorch_cuda.bat # Install CUDA PyTorch
│
├── src/                    # 11 source modules
│   ├── config.py           # All configuration & road layers
│   ├── preprocessing.py    # Noise, contrast, color
│   ├── texture_features.py # GLCM, LBP, Gabor
│   ├── segmentation.py     # K-Means, SLIC, Watershed
│   ├── morphology.py       # Post-processing
│   ├── descriptors.py      # Chain code, Fourier
│   ├── classification.py   # RandomForest, SVM classifier
│   ├── deep_learning.py    # DeepLabv3+ with CUDA
│   ├── vlm_analyzer.py     # GLM-4.6V ZenMux API
│   └── visualization.py    # Plotting utilities
│
├── gui/
│   └── main_window.py      # PyQt5 GUI (dark theme)
│
├── tests/                  # 11 test files
│   ├── conftest.py         # Fixtures
│   └── test_*.py           # Unit/integration tests
│
└── data/                   # Dataset (5 layers)
    ├── subgrade/
    ├── subbase/
    ├── base_course/
    ├── binder_course/
    └── surface_course/
```

---

## 🔑 Important Configuration

### API Key (in `.env`)
```env
ZENMUX_API_KEY=sk-ai-v1-3460a902f67de8b0e509fa1b0bc6a490bb124c177574faa804d066f7217752bc
ZENMUX_BASE_URL=https://zenmux.ai/api/v1
VLM_MODEL=z-ai/glm-4.6v
```

### CUDA Setup for RTX 4050
- Requires CUDA 12.x compatible PyTorch
- Install command: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---

## 📝 Notes for AI Agent

1. **All code is written** - focus on testing, debugging, and dataset work
2. **Virtual environment**: Use `.venv` in project root
3. **GUI is complete**: Modern dark theme with 4 analysis modes
4. **Deep Learning ready**: Just needs CUDA PyTorch installed
5. **Tests use mocking**: VLM tests don't require actual API calls
6. **Heuristic fallback**: Classification works without trained model

### If Tests Fail
- Check `src/morphology.py` - `fill_holes` function may have naming conflict
- Ensure all imports work after installing dependencies
- VLM tests require `requests` library

### GUI Features
- Load image → Select analysis mode → Click Analyze → View results
- Supports: Classical, DeepLabv3+, VLM, Hybrid modes
- Shows GLCM features and confidence scores

---

## 🎓 Deliverables Breakdown

| Component | Weight | Status |
|-----------|--------|--------|
| Presentation | 5% | 🔲 Pending |
| Report | 15% | 🔲 Pending |
| Paper | 10% | 🔲 Pending |
| **Code** | - | ✅ Complete |

---

*Last updated: January 11, 2026*
*Created on PC, to be continued on laptop with RTX 4050*
