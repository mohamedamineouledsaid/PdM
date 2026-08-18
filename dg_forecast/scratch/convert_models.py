import sys
import os
sys.path.insert(0, '.')

import tensorflow as tf
from backend.forecasting import PositionalEncoding, build_model

try:
    tf.keras.config.enable_unsafe_deserialization()
except Exception:
    pass

print("Converting Keras models to .h5 weights using Conda well_predict env...")

custom_objects = {"PositionalEncoding": PositionalEncoding}

for h_code, src_file in [("6h", "model_h6.keras"), ("3h", "model_h3.keras"), ("1h", "model_h3.keras")]:
    if os.path.exists(src_file):
        print(f"Processing {src_file}...")
        try:
            model = tf.keras.models.load_model(src_file, custom_objects=custom_objects, compile=False, safe_mode=False)
            h5_path = os.path.join("models", f"dg_forecast_{h_code}.h5")
            model.save_weights(h5_path)
            print(f"Successfully exported weights: {h5_path}")
        except Exception as e:
            print(f"Error converting {src_file}: {e}")
