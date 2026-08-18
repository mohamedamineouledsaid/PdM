import sys
import os
import tensorflow as tf

sys.path.insert(0, '.')
from backend.forecasting import PositionalEncoding

@tf.keras.utils.register_keras_serializable(package="Custom")
class SumPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        return super().get_config()

custom_objects = {
    "PositionalEncoding": PositionalEncoding,
    "Lambda": SumPoolingLayer
}

print("Testing model load in Python 3.13...")
try:
    m = tf.keras.models.load_model("model_h6.keras", custom_objects=custom_objects, compile=False, safe_mode=False)
    print("✅ SUCCESS! Model model_h6.keras loaded cleanly in Python 3.13!")
    print("Model summary input shape:", m.input_shape, "output shape:", m.output_shape)
except Exception as e:
    print("❌ Failed:", e)
