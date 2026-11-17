# fitur GLCM buat model SVM
# notebook 1 dan web app pake file ini biar fitur pas training sama pas prediksi nya sama
# (kalau beda, hasil prediksinya bakal ngaco)

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# jangan diubah kalau gak mau retrain ulang
IMG_SIZE = 224
LEVELS = 32          # level abu2nya
DISTANCES = [1, 2]   # jarak piksel
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 derajat
PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

N_FEATURES = len(PROPS) * len(DISTANCES) * len(ANGLES) * 3  # 120 (3 kanal RGB)


def extract_features(img_rgb):
    # hitung GLCM di tiap kanal R G B, ambil 5 propertinya
    # hasilnya 5 prop x 2 jarak x 4 sudut x 3 kanal = 120 angka
    feats = []
    for ch in range(3):
        q = (img_rgb[:, :, ch].astype(np.float64) / 256.0 * LEVELS).astype(np.uint8)
        glcm = graycomatrix(q, distances=DISTANCES, angles=ANGLES,
                            levels=LEVELS, symmetric=True, normed=True)
        for p in PROPS:
            feats.append(graycoprops(glcm, p).ravel())
    return np.concatenate(feats).astype(np.float64)


def feature_names():
    # nama kolom fitur, urutan: kanal -> properti -> jarak -> sudut
    names = []
    for ch_name in "RGB":
        for p in PROPS:
            for d in DISTANCES:
                for a in [0, 45, 90, 135]:
                    names.append(f"{ch_name}_{p}_d{d}_a{a}")
    return names
