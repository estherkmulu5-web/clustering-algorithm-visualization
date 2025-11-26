"""
Clustering Algorithm Performance Visualization
----------------------------------------------
Generates:
1. A grouped bar plot comparing all algorithms.
2. Individual plots for each algorithm.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
data = {
    'Algorithm': ['DBSCAN', 'SSA-DBSCAN', 'AMD-DBSCAN', 'k-means'],
    'Silhouette Score': [0.41, 0.54, 0.57, 0.29],
    'Noise Ratio (%)': [22, 16, 14, 0],
    'Execution Time (s)': [1.42, 3.70, 4.10, 0.80],
    'Spatial Cluster Accuracy': ['Good', 'Very Good', 'Excellent', 'Poor'],
    'Density Handling': ['Weak on multi-density', 'Strong', 'Very Strong', 'None']
}

df = pd.DataFrame(data)

metrics = ['Silhouette Score', 'Noise Ratio (%)', 'Execution Time (s)']


# ---------------------------------------------------------
# Grouped Bar Plot for All Algorithms
# ---------------------------------------------------------
def plot_grouped_metrics(df, metrics):
    bar_positions = np.arange(len(df['Algorithm']))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))

    # Numerical metrics
    for idx, metric in enumerate(metrics):
        plt.bar(bar_positions + idx * bar_width, df[metric], bar_width, label=metric)

    # Add annotations (qualitative metrics)
    for i, row in df.iterrows():
        annotation = f"{row['Spatial Cluster Accuracy']}, {row['Density Handling']}"
        max_height = max([row[m] for m in metrics])
        plt.text(bar_positions[i] + bar_width, max_height * 1.05, annotation,
                 ha='center', fontsize=9)

    plt.xlabel('Algorithm')
    plt.ylabel('Metric Value')
    plt.title('Clustering Algorithm Performance Comparison')
    plt.xticks(bar_positions + bar_width, df['Algorithm'])
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Individual Algorithm Plots
# ---------------------------------------------------------
def plot_individual_metrics(df, metrics):
    for _, row in df.iterrows():
        plt.figure(figsize=(6, 4))

        values = [row[m] for m in metrics]
        plt.bar(metrics, values)
        plt.ylim(0, max(df[metrics].max()) * 1.2)

        annotation = (
            f"Spatial Accuracy: {row['Spatial Cluster Accuracy']}\n"
            f"Density Handling: {row['Density Handling']}"
        )
        plt.text(1, max(values) * 1.1, annotation, ha='center', fontsize=10)

        plt.title(f"{row['Algorithm']} Metrics")
        plt.ylabel("Metric Value")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------
# Run Plots
# ---------------------------------------------------------
if __name__ == "__main__":
    plot_grouped_metrics(df, metrics)
    plot_individual_metrics(df, metrics)
