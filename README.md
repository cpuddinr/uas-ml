# UAS ML — Klasifikasi Motif Batik

Klasifikasi 20 motif batik Indonesia dengan 3 model: **GLCM + SVM**, **MobileNetV2**, **ResNet50**.

Dataset: [Indonesian Batik Motifs (Kaggle)](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) — ekstrak ke folder `data/` (**jangan push ke git**, sudah di-gitignore).

## Setup (Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow==2.21.0 pillow numpy scikit-learn scikit-image opencv-python matplotlib seaborn tqdm pandas joblib streamlit
```

## Struktur

```
data/                    # dataset Kaggle (diekstrak mandiri, tidak masuk git)
notebooks/
├── data_utils.py        # buat_split / load_split / load_images (pipeline tim)
├── split.csv            # split 70:30 stratified, seed 42 (WAJIB sama untuk semua)
├── glcm_features.py     # ekstraksi fitur GLCM (dipakai training & web)
├── 01_glcm_svm.ipynb    # GLCM + SVM  -> models/svm_glcm.pkl
├── 02_mobilenetv2.ipynb # MobileNetV2 -> models/mobilenetv2.h5
└── resnet.ipynb         # ResNet50    -> models/resnet50.keras
models/
├── labels.json          # urutan kelas = urutan output semua model (JANGAN diubah)
├── svm_glcm.pkl         # model SVM (ikut git, kecil)
├── mobilenetv2.h5       # unduh dari Drive (besar, di-gitignore)
├── resnet50.keras       # unduh dari Drive (besar, di-gitignore)
└── metrics_*.json       # metrik lengkap tiap model (untuk laporan)
app/                     # aplikasi web (Streamlit)
```

## Catatan

- Split data **wajib** via `notebooks/split.csv` / `load_split()` agar evaluasi semua model konsisten.
- Model besar (`.h5` / `.keras`) tidak masuk git — unduh dari
  [folder Drive tim](https://drive.google.com/drive/folders/1gCFERH8k8E1C0rTzOZePFDvbjyCbeSun?usp=drive_link)
  lalu taruh di `models/`.
- Hasil di data uji (295 gambar): GLCM+SVM 18,6% | MobileNetV2 35,3% | ResNet50 (lihat metrics json).
