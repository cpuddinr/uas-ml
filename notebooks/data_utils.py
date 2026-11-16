import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
SPLIT_CSV = os.path.join(BASE, "split.csv")
IMG_SIZE = (224, 224)
EXT = (".jpg", ".jpeg", ".png")

# RESNET
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _gambar_valid(path):
    try:
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def buat_split(test_size=0.3, seed=42, timpa=False):
    if os.path.exists(SPLIT_CSV) and not timpa:
        print("split.csv sudah ada, pakai load_split() aja ya")
        return pd.read_csv(SPLIT_CSV)

    rows = []
    for kelas in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, kelas)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(EXT):
                continue
            if _gambar_valid(os.path.join(folder, f)):
                rows.append((f"{kelas}/{f}", kelas))
            else:
                print("DILEWATI (rusak):", f"{kelas}/{f}")

    df = pd.DataFrame(rows, columns=["path", "label"])
    _, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    df["split"] = "train"
    df.loc[test.index, "split"] = "test"
    df.to_csv(SPLIT_CSV, index=False)
    return df


def load_split():
    df = pd.read_csv(SPLIT_CSV)
    kelas = sorted(df["label"].unique())
    label2id = {k: i for i, k in enumerate(kelas)}
    return df, kelas, label2id


def load_images(df, label2id, size=IMG_SIZE):
    X, y = [], []
    for path, label in zip(df["path"], df["label"]):
        img = Image.open(os.path.join(DATA_DIR, path)).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        X.append(np.array(img, dtype=np.uint8))
        y.append(label2id[label])
    return np.array(X), np.array(y)


def normalisasi_resnet(X):
    X = X.astype(np.float32) / 255.0
    return (X - MEAN) / STD