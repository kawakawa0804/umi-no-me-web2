import os
import io
import csv
import base64
import datetime
import threading

from flask import Flask, render_template, request, jsonify, send_file, Response

# ============== Flask ==============
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

# ---- CORS（必要なら origins を自分のドメインに絞る）----
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
except Exception:
    pass

# ============== Paths & Env ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 既定モデル（従来互換）
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# “best” の別名解決（デプロイ先でパスを環境変数で差し替え可能）
MODEL_MAP = {
    "best": os.environ.get("MODEL_BEST_PATH", DEFAULT_MODEL_PATH)
}

# Ultralytics/BLAS のスレッド抑制
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============== Model (lazy) ==============
model = None
_loaded_path = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()

def _ensure_model_loaded(target_path: str) -> bool:
    """
    指定パスのモデルが未ロード/別物なら読み込む
    """
    global model, _loaded_path
    if model is not None and _loaded_path == target_path:
        return True
    if not os.path.exists(target_path):
        print(f"[WARN] Model not found: {target_path}")
        return False
    with _model_lock:
        if model is not None and _loaded_path == target_path:
            return True
        try:
            from ultralytics import YOLO
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass
            m = YOLO(target_path)
            model = m
            _loaded_path = target_path
            print(f"[INFO] Model loaded: {_loaded_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            model = None
            _loaded_path = None
            return False

# ============== Logs ==============
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "detections.csv")

def _append_rows(rows):
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for d in rows:
            w.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                d["class_id"], d["confidence"], *d["bbox"]
            ])

# ============== Utils ==============
def _read_image_bytes():
    # multipart/form-data: 'image' or 'file'
    if "image" in request.files or "file" in request.files:
        file = request.files.get("image") or request.files.get("file")
        return file.read()
    # JSON: { "frame": "data:image/jpeg;base64,..." }
    if request.is_json:
        data_url = request.json.get("frame")
        if isinstance(data_url, str) and "base64," in data_url:
            b64 = data_url.split("base64,", 1)[1]
            return base64.b64decode(b64)
    return None

def _resolve_model_path() -> str:
    # ?model=best などを解決。なければ既定。
    name = request.args.get("model") or request.form.get("model")
    if name and name in MODEL_MAP:
        return MODEL_MAP[name]
    return MODEL_PATH

# ============== Routes ==============
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "<h1>Umi no Me</h1><p>Server is running.</p>", 200

@app.route("/health")
@app.route("/api/health")
def health():
    return "ok", 200

@app.route("/detect", methods=["POST"])
@app.route("/api/detect", methods=["POST"])  # ← フロント互換のため追加
def detect():
    if not _infer_lock.acquire(blocking=False):
        return jsonify({"error": "busy"}), 429
    try:
        target_model_path = _resolve_model_path()
        if not _ensure_model_loaded(target_model_path):
            return jsonify({"error": f"Model not available: {target_model_path}"}), 503

        img_bytes = _read_image_bytes()
        if not img_bytes:
            return jsonify({"error": "No image provided"}), 400

        import numpy as np
        import cv2
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # 幅480pxへ縮小
        target_w = 480
        h, w = img.shape[:2]
        if w > target_w:
            scale = target_w / float(w)
            img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

        try:
            results = model.predict(
                source=img,
                imgsz=320,
                conf=0.45,
                iou=0.5,
                max_det=3,
                agnostic_nms=True,
                device="cpu",
                half=False,
                retina_masks=False,
                classes=None,
                verbose=False,
                stream=False
            )
        except Exception as e:
            return jsonify({"error": f"inference failed: {e}"}), 500

        detections = []
        try:
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = [float(x) for x in box.xyxy[0].tolist()]
                    detections.append({"class_id": cls_id, "confidence": conf, "bbox": xyxy})
        except Exception as e:
            return jsonify({"error": f"parse failed: {e}"}), 500

        if detections:
            _append_rows(detections)

        return jsonify(detections)
    finally:
        _infer_lock.release()

@app.route("/csv")
def csv_view():
    from collections import deque
    rows = deque(maxlen=200)
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
    html = [
        "<html><head><meta charset='utf-8'><title>detections.csv (tail 200)</title>",
        "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>",
        "</head><body>",
        "<h2>detections.csv (latest 200 rows)</h2>",
        "<p><a href='/logs/detections.csv' download>CSVをダウンロード</a></p>",
        "<table>",
        "<tr><th>time</th><th>class_id</th><th>confidence</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr>"
    ]
    for row in rows:
        html.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html += ["</table>", "</body></html>"]
    return Response("\n".join(html), mimetype="text/html")

@app.route("/logs/detections.csv")
def csv_download():
    if not os.path.exists(CSV_PATH):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["time", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        buf.seek(0)
        return Response(buf.read(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=detections.csv"})
    return send_file(CSV_PATH, mimetype="text/csv", as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
