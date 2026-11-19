# uas-ml

Klasifikasi 20 motif batik indonesia. 3 model: GLCM+SVM, MobileNetV2, ResNet50.

dataset: https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs
ekstrak ke folder `data/` (jangan di push, udah di gitignore)

## setup (pake python 3.12)

```
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow==2.21.0 pillow numpy scikit-learn scikit-image opencv-python matplotlib seaborn tqdm pandas joblib streamlit
```

## isi folder

```
data/                    dataset kaggle (mandiri, gak masuk git)
notebooks/
  data_utils.py          load_split / load_images (pipeline tim)
  split.csv              split 70:30 stratified seed 42
  glcm_features.py       fitur GLCM (dipake notebook 1 & web)
  01_glcm_svm.ipynb      svm
  02_mobilenetv2.ipynb   mobilenet
  resnet.ipynb           resnet
models/
  labels.json            urutan kelas (JANGAN diubah, harus sama dgn output model)
  svm_glcm.pkl
  mobilenetv2.h5         unduh dari drive
  resnet50.keras         unduh dari drive
  metrics_*.json         hasil evaluasi tiap model
app/                     web streamlit
```

## catatan

- split WAJIB lewat split.csv biar semua model evaluasinya konsisten
- model yang gedenya (.h5 / .keras) gak masuk git, unduh di drive:
  https://drive.google.com/drive/folders/1gCFERH8k8E1C0rTzOZePFDvbjyCbeSun?usp=drive_link
- hasil di data test: svm 18.6% | mobilenet 35.3% (detailnya di metrics_*.json)
