import numpy as np
import matplotlib.pyplot as plt

def plot_correlation_circle(pca, features, axis1=1, axis2=2):
    """
    Displays a PCA correlation circle for two selected axes.

    pca: PCA object already fitted
    features: list of variable names (columns)
    axis1: first axis (1-indexed)
    axis2: second axis (1-indexed)
    """
    # Loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Indices Python (0-indexed)
    x_axis = axis1 - 1
    y_axis = axis2 - 1

    plt.figure(figsize=(7,7))
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)

    # Circle with R=1
    circle = plt.Circle((0,0), 1, color='blue', fill=False)
    plt.gca().add_artist(circle)

    # Arrows for each feature
    for i, col in enumerate(features):
        plt.arrow(0, 0, loadings[i, x_axis], loadings[i, y_axis],
                  color='red', alpha=0.7, head_width=0.03, length_includes_head=True)
        plt.text(loadings[i, x_axis]*1.1, loadings[i, y_axis]*1.1, col, color='black')

    plt.xlabel(f"PC{axis1} ({pca.explained_variance_ratio_[x_axis]*100:.1f}%)")
    plt.ylabel(f"PC{axis2} ({pca.explained_variance_ratio_[y_axis]*100:.1f}%)")
    plt.title(f"PCA Correlation Circle (PC{axis1} vs PC{axis2})")
    plt.axis("equal")
    plt.show()
