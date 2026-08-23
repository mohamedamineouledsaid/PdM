import tensorflow as tf
import os
import sys

# Ensure path is correct for imports
sys.path.append(r"c:\Users\BADRO INFO\.gemini\antigravity-ide\scratch\predictive_maintenance_app")

from backend.inference import PositionalEncoding, ReduceSumLayer

path = r"c:\Users\BADRO INFO\.gemini\antigravity-ide\scratch\predictive_maintenance_app\models\pm_transformer_6h.keras"

print(f"Loading {path}...")
try:
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            'PositionalEncoding': PositionalEncoding, 
            'ReduceSumLayer': ReduceSumLayer,
            'tf': tf
        },
        safe_mode=False,
        compile=False
    )
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
