#!/usr/bin/env python3
"""
ALPR Web UI — 车牌识别图形化界面
基于 Flask 的轻量级 Web 应用，支持图片上传和一键识别。
"""
import os
import cv2
import numpy as np
import time
import uuid
import logging

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from paddleocr import PaddleOCR

from src.keras_utils import load_model, detect_lp
from src.utils import im2single, image_files_from_folder
from src.label import Label, Shape, lwrite, writeShapes, readShapes, lread
from src.drawing_utils import draw_label, draw_losangle, write2img
from src.utils import crop_region

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'web_output')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LP_THRESHOLD = 0.5
VEHICLE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Model Cache ─────────────────────────────────────────────────────
_models = {}


def get_models():
    """Lazy-load models on first request."""
    if not _models:
        log.info("Loading YOLOv8 vehicle detector...")
        _models['yolo'] = YOLO('yolov8n.pt')

        log.info("Loading WPOD-Net license plate detector...")
        wpod_path = 'data/lp-detector/wpod-net_update1.h5'
        if os.path.exists(wpod_path):
            _models['wpod'] = load_model(wpod_path)
        else:
            log.warning("WPOD-Net model not found at %s. Run: bash get-networks.sh", wpod_path)
            _models['wpod'] = None

        log.info("Loading PaddleOCR...")
        _models['ocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        log.info("All models loaded.")
    return _models


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path, session_id):
    """Run the full ALPR pipeline on a single image."""
    outdir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(outdir, exist_ok=True)

    models = get_models()
    timing = {}
    vehicles = []
    plates = []

    # ── Read original ──
    Iorig = cv2.imread(image_path)
    if Iorig is None:
        raise ValueError("Cannot read image: %s" % image_path)
    orig_h, orig_w = Iorig.shape[:2]
    WH = np.array([orig_w, orig_h], dtype=float)

    # ── Stage 1: Vehicle Detection (YOLOv8) ──
    t0 = time.time()
    results = models['yolo'](image_path, conf=VEHICLE_THRESHOLD,
                             classes=VEHICLE_CLASSES, verbose=False)
    timing['vehicle_detection'] = time.time() - t0

    bname = os.path.splitext(os.path.basename(image_path))[0]
    Lcars = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tl = np.array([x1 / WH[0], y1 / WH[1]])
            br = np.array([x2 / WH[0], y2 / WH[1]])
            label = Label(0, tl, br)
            Icar = crop_region(Iorig, label)
            if Icar is not None:
                car_path = os.path.join(outdir, '%s_%dcar.png' % (bname, i))
                cv2.imwrite(car_path, Icar)
                Lcars.append(label)
                vehicles.append({
                    'index': i,
                    'tl': [float(tl[0]), float(tl[1])],
                    'br': [float(br[0]), float(br[1])],
                })

        lwrite(os.path.join(outdir, '%s_cars.txt' % bname), Lcars)

    # ── Stage 2: License Plate Detection (WPOD-Net) ──
    t0 = time.time()
    if models['wpod'] is not None and Lcars:
        for i, lcar in enumerate(Lcars):
            car_img_path = os.path.join(outdir, '%s_%dcar.png' % (bname, i))
            Ivehicle = cv2.imread(car_img_path)
            if Ivehicle is None:
                continue

            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2**4)), 608)

            Llp, LlpImgs, _ = detect_lp(
                models['wpod'], im2single(Ivehicle),
                bound_dim, 2**4, (240, 80), LP_THRESHOLD
            )

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                lp_path = os.path.join(outdir, '%s_%dlp.png' % (bname, i))
                cv2.imwrite(lp_path, Ilp * 255.)

                s = Shape(Llp[0].pts)
                writeShapes(os.path.join(outdir, '%s_%dlp.txt' % (bname, i)), [s])

                plates.append({
                    'vehicle_index': i,
                    'lp_image': '%s_%dlp.png' % (bname, i),
                })

    timing['lp_detection'] = time.time() - t0

    # ── Stage 3: OCR (PaddleOCR) ──
    t0 = time.time()
    for p in plates:
        lp_img_path = os.path.join(outdir, p['lp_image'])
        if not os.path.exists(lp_img_path):
            continue

        result = models['ocr'].ocr(lp_img_path, cls=True)
        if result and result[0]:
            lines = result[0]
            lines.sort(key=lambda x: x[0][0][0])
            text = ''.join([line[1][0] for line in lines])
            p['text'] = text
            p['confidence'] = max([line[1][1] for line in lines]) if lines else 0

            with open(os.path.join(outdir, '%s_str.txt' % os.path.splitext(p['lp_image'])[0]), 'w') as f:
                f.write(text + '\n')
        else:
            p['text'] = ''
            p['confidence'] = 0

    timing['ocr'] = time.time() - t0

    # ── Generate annotated output ──
    t0 = time.time()
    annotated = Iorig.copy()
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)

    for i, lcar in enumerate(Lcars):
        draw_label(annotated, lcar, color=YELLOW, thickness=3)

        lp_label_path = os.path.join(outdir, '%s_%dlp.txt' % (bname, i))
        lp_str_path = os.path.join(outdir, '%s_str.txt' % os.path.splitext(
            os.path.join(outdir, '%s_%dlp.png' % (bname, i)))[0])

        if os.path.exists(lp_label_path):
            Llp_shapes = readShapes(lp_label_path)
            if Llp_shapes:
                pts = Llp_shapes[0].pts * lcar.wh().reshape(2, 1) + lcar.tl().reshape(2, 1)
                ptspx = pts * WH.reshape(2, 1)
                draw_losangle(annotated, ptspx, RED, 3)

                if os.path.exists(lp_str_path):
                    with open(lp_str_path, 'r') as f:
                        lp_str = f.read().strip()
                    llp = Label(0, tl=pts.min(1), br=pts.max(1))
                    write2img(annotated, llp, lp_str)

    output_path = os.path.join(outdir, 'annotated.jpg')
    cv2.imwrite(output_path, annotated)
    timing['annotation'] = time.time() - t0

    timing['total'] = sum(timing.values())

    return {
        'session_id': session_id,
        'image_width': orig_w,
        'image_height': orig_h,
        'vehicle_count': len(vehicles),
        'plates': plates,
        'timing': {k: '%.2fs' % v for k, v in timing.items()},
        'annotated_image': 'annotated.jpg',
    }


# ── Routes ──────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='请选择一张图片')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='请选择一张图片')

        if not allowed_file(file.filename):
            return render_template('index.html', error='仅支持 JPG/PNG 格式')

        try:
            session_id = uuid.uuid4().hex[:12]
            filename = secure_filename('%s_%s' % (session_id, file.filename))
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            result = process_image(save_path, session_id)

        except Exception as e:
            log.exception("Processing failed")
            return render_template('index.html', error='处理失败: %s' % str(e))

    return render_template('index.html', result=result)


@app.route('/output/<session_id>/<filename>')
def serve_output(session_id, filename):
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], session_id), filename
    )


@app.route('/upload/<session_id>/<filename>')
def serve_upload(session_id, filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    log.info("Starting ALPR Web UI...")
    log.info("Make sure models are downloaded: bash get-networks.sh")
    log.info("Open http://127.0.0.1:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
