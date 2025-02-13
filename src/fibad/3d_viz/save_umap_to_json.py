import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm


def save_umap_json(results_dir, output_json="umap_data.json"):
    """Saves UMAP 3D embeddings and object IDs into a JSON file for use in Three.js"""

    results_dir = Path(results_dir)

    # Find batch files matching 'batch_<number>.npy'
    batch_files = sorted([f for f in results_dir.glob("batch_*.npy") if re.match(r"batch_\d+\.npy$", f.name)])

    if not batch_files:
        raise FileNotFoundError(f"No valid batch files found in {results_dir}")

    embeddings_list = []
    object_ids_list = []

    for batch_file in tqdm(batch_files):
        data = np.load(batch_file)
        embeddings_list.append(data["tensor"])
        object_ids_list.append(data["id"])  # Correct field name

    # Concatenate all embeddings and object IDs
    embeddings = np.concatenate(embeddings_list, axis=0)
    object_ids = np.concatenate(object_ids_list, axis=0)

    # Convert to JSON format
    json_data = {
        "points": [
            {"x": float(x), "y": float(y), "z": float(z), "id": int(obj_id)}
            for (x, y, z), obj_id in zip(embeddings, object_ids)
        ]
    }

    # Save to file
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"UMAP data saved to {output_json}")
