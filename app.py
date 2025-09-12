import os
import io
import csv
import base64
import datetime
import threading

from flask import Flask, render_template, request, jsonify, send_file, Response

# ================== Flask ==================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MBまでに抑える（負荷軽減）

# ================== Paths & Env ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt"))

# Ultralytics/BLASのスレッドを抑制（CPU負荷・メモリ圧縮）
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ================== Model (lazy) ==================
model = None
_model_lock = threading.Lock()     # モデル読み込みの排他
_infer_lock = threading.Lock()     # 推論を同時に1つに制限（Free環境での安定化）

def _ensure_model_loaded() -> bool:
    """必要時にのみモデルを読み込む（起動を軽く）"""
    global model
    if model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}")
        return False
    with _model_lock:
        if model is not None:
            return True
        try:
            from ultralytics import YOLO
            try:
                import torch
                torch.set_num_threads(1)  # CPUスレッドさらに絞る
            except Exception:
                pass
            m = YOLO(MODEL_PATH)
            model = m
            print(f"[INFO] Model loaded: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            model = None
            return False

# ================== Logs ==================
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "detections.csv")

def _append_rows(rows):
    # rows: list[dict]
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for d in rows:
            w.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                d["class_id"], d["confidence"], *d["bbox"]
            ])

# ================== Utils ==================
def _read_image_bytes():
    """
    - multipart/form-data: file フィールド 'image' または 'file'
    - JSON: { "frame": "data:image/jpeg;base64,..." }
    のどちらでもOK。bytesを返す。
    """
    if "image" in request.files or "file" in request.files:
        file = request.files.get("image") or request.files.get("file")
        return file.read()

    if request.is_json:
        data_url = request.json.get("frame")
        if isinstance(data_url, str) and "base64," in data_url:
            b64 = data_url.split("base64,", 1)[1]
            return base64.b64decode(b64)
    return None

# ================== Routes ==================
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "<h1>Umi no Me</h1><p>Server is running.</p>", 200

@app.route("/health")
def health():
    return "ok", 200

@app.route("/detect", methods=["POST"])
def detect():
    # 同時推論は1件に制限（重複は弾く）
    if not _infer_lock.acquire(blocking=False):
        return jsonify({"error": "busy"}), 429
    try:
        if not _ensure_model_loaded():
            return jsonify({"error": "Model not available"}), 503

        img_bytes = _read_image_bytes()
        if not img_bytes:
            return jsonify({"error": "No image provided"}), 400

        import numpy as np
        import cv2

        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # さらに縮小（幅480px目安）
        target_w = 480
        h, w = img.shape[:2]
        if w > target_w:
            scale = target_w / float(w)
            img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

        # 軽量推論設定（CPU）
        try:
            results = model.predict(
                source=img,
                imgsz=320,         # さらに小さく
                conf=0.45,         # しきい値上げで後段を軽く
                iou=0.5,
                max_det=3,         # 出力を絞る
                agnostic_nms=True, # NMSを単純化
                device="cpu",
                half=False,
                retina_masks=False,
                classes=None,      # 特定クラスだけに絞るなら [0,1,...] を指定
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
    """detections.csv の最後の200行だけHTML表示（軽量）"""
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
