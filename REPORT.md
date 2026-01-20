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

Road infrastructure plays a critical role in transportation and urban development. Proper construction and maintenance of roads require accurate identification of the various layers that make up the road structure. This project addresses the challenge of automating the analysis of road construction layers using image processing techniques applied to aerial satellite imagery.

The road construction process involves laying multiple layers, each with distinct materials and characteristics. From bottom to top, these layers include the Subgrade (foundation soil), Subbase Course (coarse crushed aggregate), Base Course (finer crushed aggregate), Binder Course (premix asphalt with aggregate), and Surface Course (smooth wearing surface). Each layer exhibits unique visual and textural properties that can be identified through image analysis.

Our system, named ClaRity, implements a comprehensive multi-method approach to layer identification. The application combines classical image processing techniques such as Gray-Level Co-occurrence Matrix (GLCM) and Local Binary Patterns (LBP) with modern deep learning models including DeepLabv3+ for semantic segmentation. Additionally, the system integrates Vision Language Model (GLM-4.6V) capabilities for AI-powered analysis and YOLOv11 for real-time instance segmentation with live preview functionality.

The entire application is built using Python 3.10+ with a professional PyQt5 graphical user interface. The system leverages CUDA GPU acceleration for efficient processing of high-resolution satellite imagery, making it practical for real-world applications in civil engineering and road inspection.

---

## 2. Objectives

The primary objective of this project is to develop an automated pipeline capable of accurately classifying the five distinct road construction layers from aerial satellite images. This involves creating a system that can distinguish between layers based on their visual and textural characteristics without requiring manual inspection.

The second objective focuses on implementing robust texture-based feature extraction methods. The system utilizes GLCM to calculate texture properties such as contrast, energy, homogeneity, and correlation. Alongside GLCM, Local Binary Patterns are employed to capture local texture patterns that help differentiate between the various road materials.

Another key objective is to provide users with multiple analysis modes to suit different needs and image types. The system offers five distinct modes: Classical analysis using traditional image processing, CNN-based deep learning using DeepLabv3+, VLM analysis using the GLM-4.6V vision language model, Hybrid mode combining Classical and VLM approaches, and YOLOv11 for real-time instance segmentation.

Finally, the project aims to deliver a professional and user-friendly GUI application. The interface includes features such as drag-and-drop image loading, real-time progress feedback, interactive visualization with overlays and contours, and the ability to export comprehensive PDF reports of the analysis results.

---

## 3. Data Collection

The images used in this project are sourced from Google Earth Pro, which provides high-resolution aerial satellite imagery of various locations. These images capture road construction sites at different stages, showing the various layers as they are being laid or exposed during maintenance work.

The dataset is organized into five categories corresponding to the five road layers. Each category folder contains sample images that predominantly display characteristics of that particular layer. The Subgrade folder contains images showing the foundation soil layer with its characteristic brown earth tones and irregular texture. The Subbase folder includes images of coarse crushed aggregate with visible stones and rough surfaces. The Base Course folder contains images of finer aggregate with more uniform texture. The Binder Course folder shows dark asphalt surfaces with visible aggregate particles. Finally, the Surface Course folder contains images of smooth, uniform asphalt wearing surfaces.

When selecting images for analysis, care was taken to include samples with varying lighting conditions, angles, and resolutions to ensure the system can handle diverse real-world scenarios. The images are preprocessed to reduce noise and enhance contrast before feature extraction and classification.

---

## 4. Flowchart

The system follows a structured processing pipeline that begins with image loading and ends with visualization of results. When a user loads an image into the application, it first undergoes preprocessing which includes noise reduction using filters such as Median or Gaussian, contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization), and optional sharpening to enhance texture details.

After preprocessing, the user selects one of the five available analysis modes. In Classical mode, the system extracts texture features using GLCM and LBP algorithms, then applies segmentation using K-Means clustering, SLIC Superpixels, or Watershed algorithm. The segmented regions undergo morphological operations to clean up noise, and finally the dominant layer is classified based on texture characteristics.

In CNN mode, the preprocessed image is resized and passed through the DeepLabv3+ semantic segmentation model. The model outputs pixel-wise class predictions which are then converted to a colored overlay visualization that is blended with the original image. Instance contours are drawn around detected regions to provide clear visual boundaries.

The VLM mode sends the image to the GLM-4.6V API through ZenMux, which returns a structured analysis including layer identification, confidence scores, material descriptions, and recommendations. The Hybrid mode runs both Classical and VLM analyses, then combines the results using configurable conflict resolution rules.

YOLOv11 mode performs instance segmentation, detecting and masking individual layer regions in the image. This mode also supports live preview functionality where the system can capture frames from any window on the screen and perform real-time analysis.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Load Image  │ ──▶ │ Preprocessng │ ──▶ │ Select Mode │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
        ┌───────────┬───────────┬───────────┬───┴───────┐
        ▼           ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Classical│ │   CNN   │ │   VLM   │ │ Hybrid  │ │  YOLO   │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │           │
        └───────────┴───────────┴───────────┴───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌────────────┐         ┌────────────┐
             │ Classify   │         │ Visualize  │
             │ Layer      │         │ Results    │
             └────────────┘         └────────────┘
```

---

## 5. Results of Prototype

The prototype successfully demonstrates the ability to identify and classify road construction layers using multiple analysis methods. Testing with various satellite images showed that the system can accurately distinguish between different layer types based on their texture and visual characteristics.

The Classical analysis mode performed well on images with clear, distinct textures. GLCM features such as contrast and energy proved particularly useful for differentiating between smooth asphalt surfaces (high energy, low contrast) and rough aggregate layers (low energy, high contrast). The homogeneity measure helped identify uniform surfaces typical of the Surface Course layer.

The CNN mode using DeepLabv3+ demonstrated strong performance on complex images where multiple layers or mixed materials were present. The semantic segmentation provided pixel-level classification that captured subtle transitions between layers. Processing time on GPU was approximately 1-2 seconds per image, making it practical for interactive use.

VLM analysis provided an additional layer of intelligence by offering natural language descriptions of the detected materials. The AI model could explain why a particular layer was identified and provide context about typical materials and construction practices. This feature proved valuable for users who may not be familiar with road construction terminology.

The Hybrid mode showed improved accuracy by cross-validating Classical and VLM results. When both methods agreed on the layer classification, confidence in the result was higher. The configurable conflict resolution rules allowed users to prioritize either method based on their specific needs.

YOLOv11 instance segmentation performed well for real-time applications, achieving inference speeds suitable for live preview mode. The instance-level masks provided clear visualization of individual layer regions within the image.

---

## 6. System Prototype in GUI

The graphical user interface was designed with usability and professionalism in mind. The application uses a dark theme that reduces eye strain during extended use and provides a modern appearance. The main window is divided into several functional areas that work together to provide a seamless user experience.

The top portion of the interface displays two image panels side by side. The left panel shows the original loaded image and supports drag-and-drop functionality for easy image loading. The right panel displays the segmentation result, which includes colored overlays showing the detected layers, instance contours around regions, and an information banner with layer identification details.

Below the image panels, a row of mode selection buttons allows users to quickly switch between the five analysis modes. Each mode has its own settings panel that appears when that mode is selected. For Classical mode, users can configure preprocessing options, texture feature selection, and segmentation method. For CNN mode, users can choose the backbone model and processing resolution. VLM mode allows adjustment of temperature and analysis type. Hybrid mode provides weight sliders and conflict resolution rules. YOLO mode includes confidence thresholds and live preview settings.

The action buttons section includes Load Image, Analyze, and Export functions. A progress bar appears during analysis to provide real-time feedback on processing status. Below this, the Results panel displays detailed information about the classification including the detected layer name, confidence percentage, material description, and processing method used.

At the bottom of the window, an interactive layer legend shows all five road layers with their associated colors. The detected layer is highlighted, making it easy to understand the classification result at a glance.

---

## 7. Sample Input Output

To illustrate the system's capabilities, consider an example where an aerial image of a recently paved road surface is analyzed. The input image shows a section of road with the characteristic smooth, dark appearance of fresh asphalt. The uniform texture with minimal visible aggregate suggests this is the final wearing course layer.

When this image is processed using CNN mode, the system applies DeepLabv3+ semantic segmentation. The output visualization shows the original image with a semi-transparent dark blue-gray overlay indicating Surface Course classification. Contour lines are drawn around the detected region, and an information banner at the top displays "DeepLabv3+: Layer 5 - Surface Course" with a confidence score of approximately 94%.

In the results panel, the system reports the detected layer as Surface Course (Layer 5) with the material identified as "Premix asphalt (smooth)". The confidence percentage indicates how certain the model is about this classification. Processing time and method details are also provided.

For a different example, consider an image showing an exposed aggregate layer during road construction. This image displays visible stones of varying sizes with a rough, granular texture. Processing this image through Classical mode results in identification as Base Course or Subbase Course, depending on the aggregate size. The GLCM features show high contrast values and low energy, characteristic of rough aggregate surfaces.

The layer color coding used in visualization follows a consistent scheme across all modes. Subgrade appears in brown tones representing the earth foundation. Subbase Course uses tan or beige colors for coarse aggregate. Base Course displays as light pinkish-gray for finer aggregate. Binder Course shows as orange-brown for the initial asphalt layer. Surface Course uses dark blue-gray for the smooth wearing surface.

---

## 8. Source Code

The project codebase is organized into two main directories: the `src` folder containing core processing modules and the `gui` folder containing the application interface. This modular structure separates concerns and makes the code maintainable and extensible.

The `src/config.py` file defines configuration constants including road layer definitions with their names, materials, and display colors. The `src/preprocessing.py` module implements image preprocessing functions including noise filtering and contrast enhancement. The `src/texture_features.py` module contains the GLCM and LBP feature extraction algorithms that form the basis of Classical analysis.

Segmentation algorithms are implemented in `src/segmentation.py`, which includes K-Means clustering, SLIC Superpixels, and Watershed segmentation. The `src/classification.py` module contains the layer classification logic that maps extracted features to road layer categories.

Deep learning functionality is provided by `src/deep_learning.py`, which wraps the DeepLabv3+ model from the segmentation-models-pytorch library. This module handles model initialization, preprocessing, inference, and result processing. The `src/yolo_analyzer.py` module similarly wraps YOLOv11 functionality for instance segmentation.

VLM integration is implemented in `src/vlm_analyzer.py`, which handles communication with the ZenMux API to access the GLM-4.6V vision language model. The module sends images to the API and parses the structured responses.

The GUI implementation centers on `gui/main_window.py`, which contains over 2500 lines of code implementing the complete application interface, analysis workflow, and visualization logic. Supporting modules include `gui/splash_screen.py` for the animated startup screen and `gui/classical_results.py` for the detailed result dialogs shown in Classical mode.

To run the application, users first activate the virtual environment, then execute the main module using the command `python -m gui.main_window`. Alternatively, Windows users can simply run the `run.bat` batch file which handles environment activation automatically.

---

## 9. Conclusion

This project successfully achieved its objectives of creating an automated road surface layer analysis system. The multi-method approach combining Classical image processing, deep learning, vision language models, and instance segmentation provides a versatile solution that can handle diverse image types and user requirements.

The texture-based feature extraction using GLCM and LBP proved effective for distinguishing between road layers with different surface characteristics. The integration of DeepLabv3+ semantic segmentation enabled accurate pixel-level classification even in complex images. The addition of GLM-4.6V VLM analysis provided an intelligent layer of interpretation that enhanced user understanding of the results.

The professional PyQt5 GUI successfully delivers an accessible interface for users who may not have technical expertise in image processing. Features such as drag-and-drop image loading, real-time progress feedback, and interactive visualization make the system practical for real-world use. The PDF export functionality allows users to document and share their analysis results.

There are some limitations to acknowledge. The accuracy of classification depends on the quality and resolution of input images. The VLM mode requires internet connectivity and depends on the availability of the external API. Deep learning inference requires significant GPU memory, which may limit use on some systems.

Future improvements could include training custom deep learning models specifically on road layer datasets to improve accuracy. Video analysis capabilities could be added for continuous monitoring applications. Batch processing features would enable analysis of multiple images in sequence. Additional texture analysis methods could further enhance classification accuracy.

In conclusion, the ClaRity Road Surface Layer Analyzer demonstrates the practical application of image processing techniques learned in CSC566 to solve a real-world problem in civil engineering. The system provides a foundation that could be extended for professional use in road construction monitoring and quality control.

---

*ClaRity Group | CSC566 Image Processing | UiTM Cawangan Perak Kampus Tapah*
