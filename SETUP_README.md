# 🚀 Quick Start Guide: Road Surface Layer Analyzer

This guide explains how to set up, install dependencies, and run the **ClaRity Road Surface Layer Analyzer** application.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your system:

1.  **Python 3.10+**: [Download from python.org](https://www.python.org/downloads/)
    *   *Note: Ensure you check "Add Python to PATH" during installation.*
2.  **Git**: [Download Git](https://git-scm.com/downloads) (to clone the repository)

---

## 🛠️ Installation & Setup

We have provided an automated batch script to handle the setup process for you.

### Step 1: Clone the Repository (No need if you already has this complete folder)
Open your terminal or command prompt and run:
```bash
git clone https://github.com/YoRzHe-HotaaRu/CSC566-Mini-Project-ClaRity.git
cd CSC566-Mini-Project-ClaRity
```

### Step 2: Run the Setup Script
Double-click on **`setup.bat`** in the project folder, or run it from the command line:
```cmd
setup.bat
```
This script will automatically:
*   ✅ Create a Python virtual environment (`.venv`)
*   ✅ Upgrade `pip` to the latest version
*   ✅ Install all required dependencies from `requirements.txt`

---

## ⚙️ Configuration (Optional)

If you plan to use the **VLM (Vision Language Model)** analysis mode, you need to configure your API key.

1.  Find the `.env.example` file in the project root.
2.  Rename it to `.env` (or create a new file named `.env`).
3.  Add your ZenMux API key:
    ```env
    ZENMUX_API_KEY=your_api_key_here
    ```

---

## ▶️ How to Run

To launch the application, simply **double-click** the launcher script:

### 👉 **`START_GUI.bat`** 👈

This will:
1.  Activate the virtual environment.
2.  Launch the graphical user interface (GUI).

*Alternatively, you can run it from the command line:*
```cmd
START_GUI.bat
```

---

## 📸 Documentation Screenshots

The screenshots in this directory are used for the project documentation.

*   **main_gui.png**: Main application interface
*   **segmentation.png**: Deep learning segmentation result
*   **vlm_analysis.png**: AI-powered Vision Language Model result
*   **results_panel.png**: Classification findings and statistics
