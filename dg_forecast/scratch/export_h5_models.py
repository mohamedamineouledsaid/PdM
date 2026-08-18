import sys
import os
sys.path.insert(0, '.')

import tensorflow as tf
from backend.forecasting import PositionalEncoding

print("Exporting Keras models to HDF5 .h5 model format using Conda well_predict env...")

custom_objects = {"PositionalEncoding": PositionalEncoding}

for h_code, src_file in [("6h", "model_h6.keras"), ("3h", "model_h3.keras"), ("1h", "model_h3.keras")]:
    if os.path.exists(src_file):
        print(f"Loading {src_file}...")
        try:
            model = tf.keras.models.load_model(src_file, custom_objects=custom_objects, compile=False)
            out_path = os.path.join("models", f"dg_forecast_{h_code}.h5")
            model.save(out_path)
            print(f"Successfully exported H5 model: {out_path}")
        except Exception as e:
            print(f"Error converting {src_file}: {e}")
