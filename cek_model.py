"""
cek_model.py - diagnosa model Keras: coba beberapa cara preprocessing sekaligus.

Cara pakai (dari folder root uas-ml):
    python cek_model.py models/resnet50.keras "C:\\path\\ke\\gambar.jpg"
Kirim seluruh hasilnya ke chat.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if len(sys.argv) < 3:
    sys.exit('Pemakaian: python cek_model.py models/resnet50.keras "gambar.jpg"')

model_path, img_path = Path(sys.argv[1]), Path(sys.argv[2])

import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.mobilenet_v2 import (  # noqa: E402
    preprocess_input as mobilenet_pre,
)
from tensorflow.keras.applications.resnet50 import (  # noqa: E402
    preprocess_input as resnet_pre,
)
from tensorflow.keras.models import load_model  # noqa: E402

labels_path = model_path.parent / "labels.json"
labels = json.load(open(labels_path, encoding="utf-8-sig")) if labels_path.exists() else []

model = load_model(model_path, compile=False)
print("\n=== INFO MODEL ===")
print("input_shape :", model.input_shape)
print("output_shape:", model.output_shape)
print("jumlah label:", len(labels))
print("lapisan awal:", [type(layer).__name__ + ":" + layer.name for layer in model.layers[:6]])
print("lapisan akhir:", [type(layer).__name__ + ":" + layer.name for layer in model.layers[-3:]])
last = model.layers[-1]
print("aktivasi akhir:", getattr(last, "activation", None).__name__ if hasattr(last, "activation") else "?")

h, w = model.input_shape[1:3]
h, w = (h or 224), (w or 224)
img = Image.open(img_path).convert("RGB").resize((w, h))
raw = np.array(img, dtype=np.float32)

variants = {
    "A. mentah 0-255": raw,
    "B. dibagi 255 (rescale)": raw / 255.0,
    "C. resnet50.preprocess_input": resnet_pre(raw.copy()),
    "D. mobilenet_v2.preprocess_input (-1..1)": mobilenet_pre(raw.copy()),
}

print("\n=== HASIL TIAP PREPROCESSING (top 3) ===")
for name, arr in variants.items():
    p = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    top = np.argsort(p)[::-1][:3]
    desc = ", ".join(
        f"{labels[i] if i < len(labels) else i}={p[i] * 100:.1f}%" for i in top
    )
    print(f"{name}\n   jumlah semua output = {p.sum():.3f} | {desc}")

print("\nTF", tf.__version__)
