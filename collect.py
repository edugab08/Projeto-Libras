import os
import time
import cv2

# >>> TROQUE pelos 10 sinais escolhidos (sem acento, sem espaco) <<<
SINAIS = ["A", "B", "C", "D", "E", "I", "O", "U", "V", "W"]

CLASSES = SINAIS + ["nada"]
# teclas 1..9 e 0 para os sinais, n para "nada"
KEYS = {ord(str((i + 1) % 10)): s for i, s in enumerate(SINAIS)}
KEYS[ord("n")] = "nada"
KEY_LABEL = {s: str((i + 1) % 10) for i, s in enumerate(SINAIS)}
KEY_LABEL["nada"] = "n"

DATA_DIR = "data"
IMG_SIZE = 224
SAVE_EVERY_N_FRAMES = 3
MAX_PER_CLASS = 300      # 300 por sinal 
BLOCK = 30               # fotos por "bloco"
COUNTDOWN = 3.0
DEBOUNCE = 0.5           # ignora repeticao de tecla (segurar) por 0.5s

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)

def count(cls):
    return len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.endswith(".jpg")])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam. Feche outros apps que usam a camera "
                     "(Zoom, Meet...) ou tente cv2.VideoCapture(1).")

recording = None       # classe sendo gravada
countdown_start = None
last_key_time = 0
frame_count = 0
block_saved = 0
luz = "escuro"         # iluminacao atual (so um rotulo; troque com a tecla l)
WIN = "Coleta"
print("Janela aberta. Clique nela. 1-9/0 = sinais, n = nada, l = trocar iluminacao, q = sair.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler a webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    status, color = "PARADO - aperte uma tecla na janela", (0, 255, 0)
    if recording and countdown_start is not None:
        left = COUNTDOWN - (time.time() - countdown_start)
        if left > 0:
            status, color = f"{recording}: prepare a mao... {int(left) + 1}", (0, 255, 255)
        else:
            countdown_start = None
            block_saved = 0
    if recording and countdown_start is None:
        status, color = f"GRAVANDO: {recording} ({block_saved}/{BLOCK})", (0, 0, 255)
        frame_count += 1
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if count(recording) < MAX_PER_CLASS:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                nome = f"{int(time.time() * 1000)}_{luz}.jpg"
                cv2.imwrite(os.path.join(DATA_DIR, recording, nome), img)
                block_saved += 1
                if block_saved >= BLOCK:
                    print(f"Bloco de {BLOCK} fotos de {recording} ({luz}) concluido. "
                          f"Troque a iluminacao (tecla l) e grave de novo.")
                    recording = None
            else:
                print(f"{recording}: completo ({MAX_PER_CLASS} fotos).")
                recording = None

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Fundo: {luz.upper()} (tecla l troca)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    for i, c in enumerate(CLASSES):
        cv2.putText(frame, f"[{KEY_LABEL[c]}] {c}: {count(c)}/{MAX_PER_CLASS}", (10, 92 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 255:
        continue
    if key == ord("q"):
        break
    if key == ord("l") and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        luz = "claro" if luz == "escuro" else "escuro"
        recording = None
        print(f"Iluminacao agora: {luz}")
        continue
    if key in KEYS and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        cls = KEYS[key]
        if recording == cls:
            recording = None
            print(f"Pausado: {cls}")
        else:
            recording, countdown_start = cls, time.time()
            print(f"Preparando: {cls} ({luz})")
    elif key not in KEYS and key != ord("l"):
        print(f"(tecla nao reconhecida: codigo {key})")

cap.release()
cv2.destroyAllWindows()
print({c: count(c) for c in CLASSES})
