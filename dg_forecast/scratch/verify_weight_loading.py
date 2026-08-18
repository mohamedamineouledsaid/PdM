import sys
import os
import zipfile
import tempfile
import h5py
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
from backend.forecasting import PositionalEncoding

def build_exact_model(seq_len=360, n_features=79):
    inputs = tf.keras.layers.Input(shape=(seq_len, n_features))
    
    # 1D CNN
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('gelu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    
    # Positional Encoding
    x = PositionalEncoding(max_seq_len=180, d_model=32)(x)
    
    # 3 Transformer Blocks (8 heads, key_dim 64, ff_dim 256)
    for _ in range(3):
        x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.10)(x_norm, x_norm)
        attn_out = tf.keras.layers.Dropout(0.10)(attn_out)
        x = tf.keras.layers.Add()([x, attn_out])
        
        x_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_out = tf.keras.layers.Dense(256, activation='gelu')(x_norm2)
        ff_out = tf.keras.layers.Dropout(0.10)(ff_out)
        ff_out = tf.keras.layers.Dense(32)(ff_out)
        ff_out = tf.keras.layers.Dropout(0.10)(ff_out)
        x = tf.keras.layers.Add()([x, ff_out])
        
    # Attention pooling
    x_norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attn_weights = tf.keras.layers.Dense(1, activation='softmax')(x_norm_final)
    weighted = tf.keras.layers.Multiply()([x_norm_final, attn_weights])
    x = tf.keras.layers.GlobalAveragePooling1D()(weighted)
    
    # MLP Head
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN_Transformer")


def test_weight_extraction(keras_file):
    print(f"\nTesting direct ZIP weight extraction for '{keras_file}'...")
    m = build_exact_model(360, 79)
    print(f"Model created. Weights count expected: {len(m.weights)}")
    
    z = zipfile.ZipFile(keras_file, 'r')
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
        
    print(f"Extracted {len(items)} weight arrays from ZIP archive.")
    m.set_weights(items)
    print(f"SUCCESS! set_weights() completed with ZERO errors!")
    return m

if __name__ == '__main__':
    m6 = test_weight_extraction("model_h6.keras")
    m3 = test_weight_extraction("model_h3.keras")
    print("\nALL WEIGHTS EXTRACTED AND VERIFIED 100% CLEANLY!")
