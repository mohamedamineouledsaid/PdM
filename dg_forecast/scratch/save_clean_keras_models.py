"""
Extract trained weights directly from zip into build_model() and save clean .keras model files.
"""
import sys
import os
import zipfile
import tempfile
import h5py
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
from backend.forecasting import PositionalEncoding, build_model

def save_clean_model(src_file, dst_name):
    if not os.path.exists(src_file):
        print(f"Skipping {src_file} (not found)")
        return

    print(f"Extracting weights from '{src_file}'...")
    z = zipfile.ZipFile(src_file, 'r')
    tmp = tempfile.mktemp(suffix='.h5')
    with open(tmp, 'wb') as f:
        f.write(z.read('model.weights.h5'))

    items = []
    with h5py.File(tmp, 'r') as h5f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and name.startswith('layers/'):
                items.append(np.array(obj))
        h5f.visititems(visitor)

    try:
        os.remove(tmp)
    except Exception:
        pass

    m = build_model(360, 79)
    print(f"Model created (expected weights: {len(m.weights)}, extracted weights: {len(items)})")
    m.set_weights(items)
    print("Set weights into build_model cleanly!")

    dst_path = os.path.join("models", dst_name)
    m.save(dst_path)
    print(f"Saved clean model to: {dst_path}")

if __name__ == '__main__':
    save_clean_model("model_h6.keras", "dg_forecast_6h.keras")
    save_clean_model("model_h3.keras", "dg_forecast_3h.keras")
    save_clean_model("model_h3.keras", "dg_forecast_1h.keras")
    print("\nALL CLEAN MODELS SAVED SUCCESSFULLY!")
