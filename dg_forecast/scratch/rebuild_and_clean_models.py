"""
Rebuild models without Lambda layers and export clean .keras files.
This permanently eliminates bad marshal data errors across all Python versions.
"""
import sys
import os
import tensorflow as tf

sys.path.insert(0, '.')
from backend.forecasting import PositionalEncoding, build_model

print("Rebuilding models without Lambda layers using Conda well_predict env...")

custom_objects = {"PositionalEncoding": PositionalEncoding}

targets = [
    ("6h", "model_h6.keras"),
    ("3h", "model_h3.keras"),
    ("1h", "model_h3.keras")
]

for h_code, src_file in targets:
    if not os.path.exists(src_file):
        print(f"Skipping {src_file} (not found)")
        continue

    print(f"\nProcessing '{src_file}'...")
    try:
        # 1. Enable unsafe deserialization to load trained weights in Conda env
        try:
            tf.keras.config.enable_unsafe_deserialization()
        except Exception:
            pass

        orig_model = tf.keras.models.load_model(src_file, custom_objects=custom_objects, compile=False, safe_mode=False)
        weights = orig_model.get_weights()
        print(f"Loaded original model '{src_file}' ({len(weights)} weight arrays).")

        # 2. Build clean architecture without Lambda layer
        clean_model = build_model(360, 79)
        clean_model.set_weights(weights)
        print("Set weights into clean model architecture successfully!")

        # 3. Save clean .keras file
        dst_path = os.path.join("models", f"dg_forecast_{h_code}.keras")
        clean_model.save(dst_path)
        print(f"Saved clean model file: {dst_path}")

    except Exception as e:
        print(f"Error processing {src_file}: {e}")

print("\nFinished cleaning and saving models!")
