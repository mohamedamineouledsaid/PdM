"""
Extract trained Keras model weights to standard numpy .npz files.
This completely bypasses Python bytecode marshal issues across Python versions.
"""
import sys
import os
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
from backend.forecasting import PositionalEncoding, build_model

print("Extracting weights from .keras models to .npz archives using Conda well_predict env...")

custom_objects = {"PositionalEncoding": PositionalEncoding}

models_to_convert = [
    ("6h", "model_h6.keras"),
    ("3h", "model_h3.keras"),
    ("1h", "model_h3.keras")
]

for h_code, src_file in models_to_convert:
    if not os.path.exists(src_file):
        print(f"Skipping {src_file} (file not found)")
        continue

    print(f"\nProcessing '{src_file}'...")
    try:
        # Load model using conda env where it was trained
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(src_file, custom_objects=custom_objects, compile=False, safe_mode=False)
        
        # Extract numpy weight arrays
        weight_arrays = model.get_weights()
        print(f"Extracted {len(weight_arrays)} weight tensors.")

        # Verify against build_model architecture
        test_model = build_model(360, 79)
        test_model.set_weights(weight_arrays)
        print(f"Verified set_weights() compatibility on build_model()!")

        # Save to .npz file
        npz_path = os.path.join("models", f"dg_forecast_{h_code}.npz")
        np.savez_compressed(npz_path, *weight_arrays)
        print(f"Saved: {npz_path}")

    except Exception as e:
        print(f"Error processing {src_file}: {e}")

print("\nDone extracting NPZ weights!")
