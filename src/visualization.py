"""
Visualization Module for Road Surface Layer Analyzer
Provides display utilities, plotting, and result visualization.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .config import ROAD_LAYERS, LAYER_COLORS_RGB


def display_image(
    image: np.ndarray,
    title: str = "Image",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = None,
    save_path: Optional[str] = None
) -> None:
    """
    Display a single image using matplotlib.
    
    Args:
        image: Image to display
        title: Figure title
        figsize: Figure size
        cmap: Colormap (auto-detected if None)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=figsize)
    
    if len(image.shape) == 3:
        # Convert BGR to RGB for display
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        else:
            plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap or 'gray')
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.tight_layout()
    plt.show()


def display_comparison(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5),
    cols: int = None,
    save_path: Optional[str] = None
) -> None:
    """
    Display multiple images side by side.
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
        cols: Number of columns (auto-calculated if None)
        save_path: Path to save figure
    """
    n = len(images)
    if cols is None:
        cols = min(n, 4)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        ax.set_title(title)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def create_segmentation_colormap() -> ListedColormap:
    """
    Create colormap for road layer visualization.
    
    Returns:
        Matplotlib colormap
    """
    colors = [LAYER_COLORS_RGB[i] for i in range(1, 6)]
    return ListedColormap(colors)


def visualize_segmentation(
    original: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize segmentation results with overlay.
    
    Args:
        original: Original image
        labels: Segmentation labels (1-indexed for road layers)
        alpha: Overlay transparency
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Overlay image
    """
    # Create colored segmentation
    colored = create_colored_segmentation(labels)
    
    # Create overlay
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        original_rgb = original.copy()
    
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, colored, alpha, 0)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original
    axes[0].imshow(cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Segmentation")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Add legend
    add_layer_legend(fig)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
    
    return overlay


def create_colored_segmentation(
    labels: np.ndarray
) -> np.ndarray:
    """
    Create colored image from segmentation labels.
    
    Args:
        labels: Segmentation labels (1-indexed)
        
    Returns:
        BGR colored image
    """
    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for layer_num in range(1, 6):
        if layer_num in ROAD_LAYERS:
            mask = labels == layer_num
            if mask.any():
                color = ROAD_LAYERS[layer_num]["color"]
                colored[mask] = color
    
    return colored


def add_layer_legend(fig: plt.Figure, loc: str = "right") -> None:
    """
    Add road layer legend to figure.
    
    Args:
        fig: Matplotlib figure
        loc: Legend location
    """
    import matplotlib.patches as mpatches
    
    patches = []
    for layer_num in range(1, 6):
        layer = ROAD_LAYERS[layer_num]
        color = LAYER_COLORS_RGB[layer_num]
        patch = mpatches.Patch(color=color, label=f"{layer_num}. {layer['name']}")
        patches.append(patch)
    
    fig.legend(handles=patches, loc='center right', title="Road Layers")


def plot_histogram(
    image: np.ndarray,
    title: str = "Histogram",
    color: str = 'gray',
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot image histogram.
    
    Args:
        image: Grayscale image
        title: Plot title
        color: Histogram color
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    if len(image.shape) == 3:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title(f"{title} - RGB Channels")
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.fill_between(range(256), hist.flatten(), alpha=0.3)
        plt.title(title)
    
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def plot_feature_heatmap(
    feature_map: np.ndarray,
    title: str = "Feature Heatmap",
    cmap: str = 'jet',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot texture feature heatmap.
    
    Args:
        feature_map: 2D feature map
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    im = plt.imshow(feature_map, cmap=cmap)
    plt.colorbar(im, label="Feature Value")
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = [ROAD_LAYERS[i]["name"] for i in range(1, 6)]
    
    plt.figure(figsize=figsize)
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def plot_layer_distribution(
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Plot distribution of road layers in segmentation.
    
    Args:
        labels: Segmentation labels
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Dictionary with layer percentages
    """
    total_pixels = labels.size
    layer_counts = {}
    layer_percentages = {}
    
    for layer_num in range(1, 6):
        count = (labels == layer_num).sum()
        layer_counts[ROAD_LAYERS[layer_num]["name"]] = count
        layer_percentages[ROAD_LAYERS[layer_num]["name"]] = count / total_pixels * 100
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = [LAYER_COLORS_RGB[i] for i in range(1, 6)]
    names = list(layer_counts.keys())
    counts = list(layer_counts.values())
    
    # Bar chart
    axes[0].barh(names, counts, color=colors)
    axes[0].set_xlabel("Pixel Count")
    axes[0].set_title("Layer Distribution (Pixels)")
    
    # Pie chart
    axes[1].pie(counts, labels=names, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Layer Distribution (%)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
    
    return layer_percentages


def create_result_report(
    original: np.ndarray,
    segmented: np.ndarray,
    classification_result: Dict,
    texture_features: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive result visualization report.
    
    Args:
        original: Original image
        segmented: Segmentation labels
        classification_result: Classification results
        texture_features: Extracted texture features
        save_path: Path to save report
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax1 = fig.add_subplot(2, 3, 1)
    if len(original.shape) == 3:
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(original, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Segmented image
    ax2 = fig.add_subplot(2, 3, 2)
    colored_seg = create_colored_segmentation(segmented)
    ax2.imshow(cv2.cvtColor(colored_seg, cv2.COLOR_BGR2RGB))
    ax2.set_title("Segmentation Result")
    ax2.axis('off')
    
    # Overlay
    ax3 = fig.add_subplot(2, 3, 3)
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        original_rgb = original.copy()
    overlay = cv2.addWeighted(original_rgb, 0.6, colored_seg, 0.4, 0)
    ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax3.set_title("Overlay")
    ax3.axis('off')
    
    # Classification result text
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    result_text = f"""
    Classification Result
    ─────────────────────
    Layer: {classification_result.get('layer_name', 'N/A')}
    Confidence: {classification_result.get('confidence', 0):.1%}
    Material: {classification_result.get('material', 'N/A')}
    """
    ax4.text(0.1, 0.5, result_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    ax4.set_title("Classification")
    
    # GLCM features
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    if "glcm" in texture_features:
        glcm = texture_features["glcm"]
        feature_text = f"""
    GLCM Texture Features
    ─────────────────────
    Contrast:    {glcm.get('contrast', 0):.4f}
    Energy:      {glcm.get('energy', 0):.4f}
    Homogeneity: {glcm.get('homogeneity', 0):.4f}
    Correlation: {glcm.get('correlation', 0):.4f}
    Entropy:     {glcm.get('entropy', 0):.4f}
        """
        ax5.text(0.1, 0.5, feature_text, fontsize=10, family='monospace',
                 verticalalignment='center', transform=ax5.transAxes)
    ax5.set_title("Texture Features")
    
    # Layer distribution
    ax6 = fig.add_subplot(2, 3, 6)
    colors = [LAYER_COLORS_RGB[i] for i in range(1, 6)]
    names = [ROAD_LAYERS[i]["name"] for i in range(1, 6)]
    counts = [(segmented == i).sum() for i in range(1, 6)]
    ax6.pie(counts, labels=None, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.legend(names, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax6.set_title("Layer Distribution")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def save_results_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save result image to file.
    
    Args:
        image: Image to save
        path: Output path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), image)
