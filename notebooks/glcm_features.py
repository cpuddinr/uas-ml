"""
Ekstraksi fitur tekstur GLCM (Gray-Level Co-occurrence Matrix).
Dipakai oleh notebooks/01_glcm_svm.ipynb dan app/predict.py (versi asli)
agar fitur saat training dan saat prediksi DIJAMIN sama.

Konfigurasi: GLCM dihitung pada tiap kanal R, G, B (motif batik dibedakan
pola teksturnya pada tiap kanal warna), sehingga menghasilkan
5 properti x 2 jarak x 4 sudut x 3 kanal = 120 fitur per gambar.

Referensi fitur: R.M. Haralick, K. Shanmugam, I. Dinstein (1973),
"Textural Features for Image Classification", IEEE Trans. SMC.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# --------- Konfigurasi fitur (JANGAN diubah tanpa re-training) ------------
IMG_SIZE = 224          # gambar sudah di-resize 224x224 saat preprocessing
LEVELS = 32             # kuantisasi level intensitas per kanal
DISTANCES = [1, 2]      # jarak piksel pasangan
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 derajat
PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
SYMMETRIC = True
NORMED = True
N_FEATURES = len(PROPS) * len(DISTANCES) * len(ANGLES) * 3  # = 120


def _quantize(channel: np.ndarray) -> np.ndarray:
    """Kuantisasi satu kanal [0..255] ke LEVELS level intensitas."""
    return (channel.astype(np.float64) / 256.0 * LEVELS).astype(np.uint8)


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """Hitung 120 fitur GLCM (5 prop x 2 jarak x 4 sudut x 3 kanal RGB)."""
    feats = []
    for ch in range(3):  # R, G, B
        glcm = graycomatrix(
            _quantize(img_rgb[:, :, ch]),
            distances=DISTANCES,
            angles=ANGLES,
            levels=LEVELS,
            symmetric=SYMMETRIC,
            normed=NORMED,
        )
        feats += [graycoprops(glcm, p).ravel() for p in PROPS]
    return np.concatenate(feats).astype(np.float64)


def feature_names() -> list:
    """Nama kolom fitur: kanal -> properti -> jarak -> sudut."""
    names = []
    for ch, ch_name in zip(range(3), "RGB"):
        for p in PROPS:
            for d in DISTANCES:
                for a_deg in [0, 45, 90, 135]:
                    names.append(f"{ch_name}_{p}_d{d}_a{a_deg}")
    return names
