import sys
import pathlib
import numpy as np
import cv2
import torch

pathlib.PosixPath = pathlib.WindowsPath

YOLO_PATH = r'C:\Users\rahmo\Documents\projetsemestriel\controllers\arm_controller\yolov5'
MODEL_PATH = r'C:\Users\rahmo\Documents\projetsemestriel\controllers\arm_controller\best.pt'

sys.path.insert(0, YOLO_PATH)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

print("[YOLO] Chargement du modèle...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(MODEL_PATH, device=device)
model.warmup()
print("[YOLO] Modèle prêt !")

from controller import Robot, Camera, Display

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ── Caméras ──
side_camera = robot.getDevice("side_camera")
if side_camera is None:
    raise RuntimeError('[ERREUR] Caméra "side_camera" introuvable sur le robot UR5e.')
side_camera.enable(timestep)

top_camera = robot.getDevice("top_camera")
if top_camera is None:
    raise RuntimeError('[ERREUR] Caméra "top_camera" introuvable sur le robot UR5e.')
top_camera.enable(timestep)

# ── Display (optionnel) ──
display = robot.getDevice("side_display")
USE_DISPLAY = display is not None
if USE_DISPLAY:
    disp_w = display.getWidth()
    disp_h = display.getHeight()
    print("[INFO] Display 'side_display' trouvé et activé.")
else:
    print("[AVERT] Display 'side_display' introuvable. Affichage via cv2.imshow.")


# ──────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────

def preprocess(img_rgb):
    """Prépare un tenseur YOLO 1×3×640×640 depuis une image RGB."""
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_chw = img_resized.transpose((2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(img_chw))
    tensor = tensor.to(device).float() / 255.0
    return tensor.unsqueeze(0)


def run_inference(tensor):
    pred = model(tensor)
    pred = non_max_suppression(pred, 0.25, 0.45)

    detections = []

    for det in pred:
        if len(det):
            for *box, conf, cls in det:

                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)

                # 🔥 PROTECTION AGAINST CRASH
                if isinstance(model.names, dict):
                    if cls_id not in model.names:
                        print(f"[WARNING] Invalid class id: {cls_id}")
                        continue
                    label = model.names[cls_id]
                else:
                    if cls_id >= len(model.names):
                        print(f"[WARNING] Invalid class id: {cls_id}")
                        continue
                    label = model.names[cls_id]

                detections.append((label, float(conf), x1, y1, x2, y2))

                print(f"[DETECT] {label} ({conf:.2f})")

    return detections

def draw_detections(frame_rgb, detections, orig_w, orig_h):
    """Dessine les bounding boxes sur l'image (mise à l'échelle depuis 640×640)."""
    img = frame_rgb.copy()
    scale_x = orig_w / 640
    scale_y = orig_h / 640

    for label, conf, x1, y1, x2, y2 in detections:
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def send_to_display(display, img_rgb, width, height):
    """Envoie l'image annotée au Display Webots."""
    img_resized = cv2.resize(img_rgb, (width, height))
    img_bgra = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGRA)
    img_bytes = img_bgra.tobytes()
    ir = display.imageNew(img_bytes, Display.BGRA, width, height)
    display.imagePaste(ir, 0, 0, False)
    display.imageDelete(ir)


def webots_image_to_rgb(camera):
    """Convertit l'image Webots (BGRA bytes) en tableau RGB numpy."""
    raw = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB), w, h


# ──────────────────────────────────────────────
# Boucle principale
# ──────────────────────────────────────────────

while robot.step(timestep) != -1:

    # ── TOP camera ──
    img_rgb, w, h = webots_image_to_rgb(top_camera)

    tensor = preprocess(img_rgb)
    detections = run_inference(tensor)
    annotated = draw_detections(img_rgb, detections, w, h)

    # ── Display ──
    if USE_DISPLAY:
        send_to_display(display, annotated, disp_w, disp_h)
    else:
        cv2.imshow("TOP Camera - YOLO", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()