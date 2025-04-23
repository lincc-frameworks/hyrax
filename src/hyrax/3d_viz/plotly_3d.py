import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def plot_umap_3d_interactive(results_dir, fig_size=(800, 600)):
    """Reads UMAP batch results, extracts object IDs,
    and creates an interactive 3D scatter plot with tooltips."""

    results_dir = Path(results_dir)

    # Find batch files matching 'batch_<number>.npy'
    batch_files = sorted([f for f in results_dir.glob("batch_*.npy") if re.match(r"batch_\d+\.npy$", f.name)])

    if not batch_files:
        raise FileNotFoundError(f"No valid batch files found in {results_dir}")

    # Load embeddings and object IDs from all batches
    embeddings_list = []
    object_ids_list = []

    for batch_file in batch_files:
        data = np.load(batch_file)
        embeddings_list.append(data["tensor"])
        object_ids_list.append(data["id"])  # Corrected field name

    # Concatenate all embeddings and object IDs
    embeddings = np.concatenate(embeddings_list, axis=0)
    object_ids = np.concatenate(object_ids_list, axis=0)

    if embeddings.shape[1] != 3:
        raise ValueError(f"Expected 3D embeddings, but got shape {embeddings.shape}")

    hover_texts = [
        f"Object ID: {obj_id}<br>X: {x:.3f}<br>Y: {y:.3f}<br>Z: {z:.3f}"
        for (x, y, z), obj_id in zip(embeddings, object_ids)
    ]

    # Create interactive 3D scatter plot with Plotly
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                z=embeddings[:, 2],
                mode="markers",
                marker=dict(size=3, color=embeddings[:, 2], colorscale="Viridis", opacity=0.8),
                text=hover_texts,
                hoverinfo="text",
            )
        ]
    )

    # Customize layout
    fig.update_layout(
        title="Interactive 3D UMAP Embeddings",
        width=fig_size[0],
        height=fig_size[1],
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3",
            bgcolor="black",
        ),
        template="plotly_dark",
    )

    fig.show()
