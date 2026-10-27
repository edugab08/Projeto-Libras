import os
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
EPOCHS = 8
VAL_FRACTION = 0.2
SEED = 42

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
print("Classes:", class_names)


train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    files = sorted(os.listdir(os.path.join(DATA_DIR, cls)))
    files = [os.path.join(DATA_DIR, cls, f) for f in files if f.lower().endswith(".jpg")]
    cut = int(len(files) * (1 - VAL_FRACTION))
    train_paths += files[:cut]; train_y += [idx] * cut
    val_paths += files[cut:]; val_y += [idx] * (len(files) - cut)
    print(f"  {cls}: {cut} treino / {len(files) - cut} validacao")


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)

# Data augmentation: so atua durante o treino
augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False  # <- o "congelamento" do transfer learning

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("gesture_model.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Curvas de treino vs validacao (otimo para discutir overfitting em aula)
plt.figure(figsize=(10, 4))
for i, (m, t) in enumerate([("accuracy", "Acuracia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(hist.history[m], label="treino")
    plt.plot(hist.history["val_" + m], label="validacao")
    plt.title(t); plt.xlabel("epoca"); plt.legend()
plt.tight_layout(); plt.savefig("curvas.png")

# Matriz de confusao simples
y_true = np.array(val_y)
y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

print("\nMatriz de confusao (linhas = real, colunas = previsto):")
print(" " * 11 + " ".join(f"{c[:7]:>7}" for c in class_names))
for i, c in enumerate(class_names):
    print(f"{c[:10]:>10} " + " ".join(f"{v:>7}" for v in cm[i]))


def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10, qualidade=70):
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110
    cell = (pag_w - 2 * margem) // cols
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows
    try:
        fonte = ImageFont.load_default(size=30)
    except TypeError:
        fonte = ImageFont.load_default()

    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for classe in classes:
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        n_pag = max(1, -(-len(fotos) // por_pagina))
        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {len(fotos)} imagens | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)
            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                x = margem + (i % cols) * cell + 3
                y = topo + (i // cols) * cell + 3
                pagina.paste(img, (x, y))
            paginas.append(pagina)

    paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                    quality=qualidade, resolution=150)
    print(f"PDF salvo: {saida} ({len(paginas)} paginas)")


gerar_pdf_dataset(DATA_DIR)

print("\nPronto! Modelo salvo em gesture_model.keras. Agora rode: python play.py")

import os

from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "data"


def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10, qualidade=70):
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110  # pagina A4 a 150 dpi
    cell = (pag_w - 2 * margem) // cols  # tamanho de cada miniatura
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows

    try:
        fonte = ImageFont.load_default(size=30)
    except TypeError:  # Pillow mais antigo
        fonte = ImageFont.load_default()

    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))

    for classe in classes:  # uma secao por classe
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        n_pag = max(1, -(-len(fotos) // por_pagina))  # divisao arredondada para cima

        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {len(fotos)} imagens | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)

            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                x = margem + (i % cols) * cell + 3
                y = topo + (i // cols) * cell + 3
                pagina.paste(img, (x, y))

            paginas.append(pagina)

    paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                    quality=qualidade, resolution=150)
    print(f"PDF salvo: {saida} ({len(paginas)} paginas)")


gerar_pdf_dataset(DATA_DIR)