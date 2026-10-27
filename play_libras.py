"""
Passo 3 (Libras) - Mostra o sinal reconhecido pelo modelo em tempo real.

O modelo roda em uma thread separada: o video fica fluido mesmo em PC lento,
e so a letra demora um pouco mais para atualizar.

Uso: python play_libras.py
  q = sair
"""
import json
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
LIMIAR = 0.60   # abaixo disso o modelo "nao tem certeza" e mostramos "?"

model = tf.keras.models.load_model("gesture_model.keras")
class_names = json.load(open("class_names.json"))


def predict(roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model(img[None].astype("float32"), training=False).numpy()[0]
    return class_names[int(np.argmax(probs))], float(np.max(probs)), probs


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam.")

lock = threading.Lock()
estado = {
    "roi": None,                              # ultimo recorte da camera
    "label": "nada", "conf": 0.0,
    "probs": np.zeros(len(class_names)),
    "recent": deque(maxlen=5),                # voto da maioria nas ultimas previsoes
    "parar": False,
}


def trabalhador():
    """Fica prevendo o recorte mais recente, no ritmo que o PC aguentar."""
    while True:
        with lock:
            if estado["parar"]:
                return
            roi = estado["roi"]
            estado["roi"] = None              # so processa recortes novos
        if roi is None:
            time.sleep(0.01)
            continue
        label, conf, probs = predict(roi)
        with lock:
            estado.update(label=label, conf=conf, probs=probs)
            estado["recent"].append(label)


threading.Thread(target=trabalhador, daemon=True).start()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    with lock:
        estado["roi"] = roi.copy()
        conf, probs = estado["conf"], estado["probs"].copy()
        recent = list(estado["recent"])
    stable = Counter(recent).most_common(1)[0][0] if recent else "nada"

    if stable == "nada" or conf < LIMIAR:
        texto, cor = "Sinal: ?", (0, 0, 255)
    else:
        texto, cor = f"Sinal: {stable} ({conf:.0%})", (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), cor, 3)
    cv2.putText(frame, texto, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

    # barras de probabilidade por classe
    for i, c in enumerate(class_names):
        bw = int(probs[i] * 150)
        cv2.rectangle(frame, (10, 15 + i * 22), (10 + bw, 29 + i * 22), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Reconhecimento de Libras", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with lock:
    estado["parar"] = True
cap.release()
cv2.destroyAllWindows()