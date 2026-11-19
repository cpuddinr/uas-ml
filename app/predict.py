"""
predict.py - jembatan antara tampilan (app.py) dan model.

Cara kerja:
- Kalau file model + labels.json sudah ada di folder models/, dipakai model ASLI.
- Kalau belum ada, otomatis pakai hasil ACAK (palsu) supaya web tetap bisa dikembangkan.
Tidak perlu ubah kode app.py saat model sudah jadi, cukup taruh file di models/.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = (224, 224)

MODEL_NAMES = ["SVM (GLCM)", "MobileNetV2", "ResNet50"]
# Tiap model boleh punya beberapa kemungkinan nama file (yang pertama ditemukan dipakai).
MODEL_FILES = {
    "SVM (GLCM)": ["svm_glcm.pkl"],
    "MobileNetV2": ["mobilenetv2.keras", "mobilenetv2.h5"],
    "ResNet50": ["resnet50.keras", "resnet50.h5"],
}


def _model_path(model_name):
    """Path file model yang ada di folder models/, atau None kalau belum ada."""
    for fname in MODEL_FILES[model_name]:
        p = MODELS_DIR / fname
        if p.exists():
            return p
    return None

# Preprocessing untuk model Keras. HARUS sama dengan yang dipakai saat training
# di notebook 02 / 03. Pilihan: "none" (piksel mentah 0-255, preprocessing sudah di
# dalam model), "mobilenet_v2", "resnet50", atau "rescale" (bagi 255).
# ResNet50 sudah diuji dengan cek_model.py: "none" memberi hasil benar.
KERAS_PREPROCESS = {
    "MobileNetV2": "none",  # sudah ada lapisan Rescaling di dalam model (diuji cek_model.py)
    "ResNet50": "none",
}

# Dipakai hanya kalau models/labels.json belum ada. Ganti dengan kelas dataset asli.
PLACEHOLDER_LABELS = [
    "Batik Parang",
    "Batik Kawung",
    "Batik Mega Mendung",
    "Batik Truntum",
    "Batik Sido Mukti",
]

_cache = {}


def load_labels():
    """Baca models/labels.json. Mendukung list ["a","b"] atau dict {"0":"a"} / {"a":0}."""
    path = MODELS_DIR / "labels.json"
    if not path.exists():
        return list(PLACEHOLDER_LABELS)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if all(str(k).isdigit() for k in data.keys()):
        return [data[k] for k in sorted(data.keys(), key=int)]
    return [name for name, _ in sorted(data.items(), key=lambda kv: kv[1])]


def is_real(model_name):
    """True kalau file model dan labels.json sudah tersedia."""
    return _model_path(model_name) is not None and (MODELS_DIR / "labels.json").exists()


#model palsu
def _predict_fake(image, model_name, labels):
    # Seed dari isi gambar + nama model: gambar yang sama -> hasil sama (stabil untuk demo)
    seed_src = image.tobytes() + model_name.encode()
    seed = int(hashlib.md5(seed_src).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(len(labels)) * 0.5)
    return probs


#model asli
def _glcm_features(image):
    """Ekstraksi GLCM. HARUS sama dengan notebook 01_glcm_svm (jarak, sudut, properti)."""
    from skimage.feature import graycomatrix, graycoprops

    gray = np.array(image.convert("L").resize(IMG_SIZE), dtype=np.uint8)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats, dtype=np.float64).reshape(1, -1)


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _predict_svm(image):
    if "svm" not in _cache:
        import joblib

        _cache["svm"] = joblib.load(_model_path("SVM (GLCM)"))
    model = _cache["svm"]
    feats = _glcm_features(image)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(feats)[0]
    # SVC tanpa probability=True: pakai softmax dari decision_function
    return _softmax(model.decision_function(feats))[0]


def _predict_keras(image, model_name):
    if model_name not in _cache:
        from tensorflow.keras.models import load_model

        _cache[model_name] = load_model(_model_path(model_name), compile=False)
    model = _cache[model_name]

    arr = np.array(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    mode = KERAS_PREPROCESS[model_name]
    if mode == "mobilenet_v2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    elif mode == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        arr = preprocess_input(arr)
    elif mode == "rescale":
        arr = arr / 255.0
    # mode "none": piksel mentah 0-255 (preprocessing sudah ada di dalam model)
    return model.predict(np.expand_dims(arr, 0), verbose=0)[0]


#fungsi utama
def predict(image, model_name):
    """
    image      : PIL.Image
    model_name : salah satu dari MODEL_NAMES
    return     : dict
        label         -> nama motif prediksi
        confidence    -> probabilitas kelas prediksi (0-1)
        probabilities -> dict {nama_kelas: probabilitas}, urut dari terbesar
        mode          -> "asli" atau "palsu"
        elapsed       -> waktu prediksi (detik)
    """
    labels = load_labels()
    start = time.perf_counter()

    if is_real(model_name):
        if model_name == "SVM (GLCM)":
            probs = _predict_svm(image)
        else:
            probs = _predict_keras(image, model_name)
        mode = "asli"
    else:
        probs = _predict_fake(image, model_name, labels)
        mode = "palsu"

    elapsed = time.perf_counter() - start
    probs = np.asarray(probs, dtype=float)
    if len(probs) != len(labels):
        raise ValueError(
            f"Model menghasilkan {len(probs)} kelas, tetapi labels.json berisi "
            f"{len(labels)} kelas. Pastikan labels.json sama dengan data training."
        )
    top = int(np.argmax(probs))
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return {
        "label": labels[top],
        "confidence": float(probs[top]),
        "probabilities": {name: float(p) for name, p in ranked},
        "mode": mode,
        "elapsed": elapsed,
    }