import atexit
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
photo_folder = os.path.join(BASE_DIR, "static", "photos")
os.makedirs(photo_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera handling (lazy init + lock so the live stream and captures never
# fight over the same device, and a missing/disconnected camera no longer
# crashes the whole app).
# ---------------------------------------------------------------------------
camera = None
camera_lock = threading.Lock()


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def read_frame():
    with camera_lock:
        cam = get_camera()
        if cam is None or not cam.isOpened():
            return False, None
        return cam.read()


def release_camera():
    global camera
    if camera is not None:
        camera.release()


atexit.register(release_camera)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
FILTERS = {
    "none": "No Filter",
    "grayscale": "Grayscale",
    "sepia": "Sepia",
    "vintage": "Vintage",
    "cool": "Cool",
    "warm": "Warm",
    "invert": "Invert",
    "cartoon": "Cartoon",
    "soft": "Soft Glow",
    "vivid": "Vivid",
    "noir": "Noir",
}


def _sepia(frame):
    kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    sep = cv2.transform(frame, kernel)
    return np.clip(sep, 0, 255).astype(np.uint8)


def _vignette(frame, strength=0.75):
    h, w = frame.shape[:2]
    kx = cv2.getGaussianKernel(w, w * 0.5)
    ky = cv2.getGaussianKernel(h, h * 0.5)
    kernel = ky * kx.T
    mask = kernel / kernel.max()
    mask = (1 - strength) + strength * mask
    out = frame.astype(np.float64)
    for i in range(3):
        out[:, :, i] *= mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


def apply_filter(frame, filter_name):
    if filter_name not in FILTERS:
        filter_name = "none"

    if filter_name == "none":
        return frame

    if filter_name == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if filter_name == "sepia":
        return _sepia(frame)

    if filter_name == "vintage":
        return _vignette(_sepia(frame), strength=0.55)

    if filter_name == "cool":
        b, g, r = cv2.split(frame)
        b = cv2.add(b, 25)
        r = cv2.subtract(r, 15)
        return cv2.merge((b, g, r))

    if filter_name == "warm":
        b, g, r = cv2.split(frame)
        r = cv2.add(r, 25)
        b = cv2.subtract(b, 15)
        return cv2.merge((b, g, r))

    if filter_name == "invert":
        return cv2.bitwise_not(frame)

    if filter_name == "cartoon":
        return _cartoon(frame)

    if filter_name == "soft":
        return cv2.bilateralFilter(frame, 9, 75, 75)

    if filter_name == "vivid":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if filter_name == "noir":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return _vignette(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), strength=0.6)

    return frame


def clean_filter_name(value):
    value = (value or "none").strip().lower()
    return value if value in FILTERS else "none"


# ---------------------------------------------------------------------------
# Photo helpers
# ---------------------------------------------------------------------------
def get_photo_path(filename):
    return os.path.join(photo_folder, filename)


def is_safe_filename(filename):
    return bool(filename) and os.path.basename(filename) == filename and filename in os.listdir(photo_folder)


def list_photos():
    files = [
        f for f in os.listdir(photo_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(files, key=lambda name: os.path.getmtime(get_photo_path(name)), reverse=True)


def list_strips():
    return [f for f in os.listdir(photo_folder) if f.lower().endswith("_strip.jpg")]


def save_capture(filter_name):
    """Grabs one frame from the camera, applies a filter, and saves it.
    Returns the saved filename, or None if the camera isn't available."""
    success, frame = read_frame()

    if not success or frame is None:
        return None

    frame = apply_filter(frame, filter_name)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    filepath = get_photo_path(filename)
    cv2.imwrite(filepath, frame)
    return filename


def build_strip(selected, add_border, custom_message):
    """Stitches 4 named photos into a single strip image. Returns the
    generated strip filename, or None if something about the input was
    invalid."""
    selected = [name for name in selected if name and is_safe_filename(name)]

    if len(selected) != 4:
        return None

    image_paths = [get_photo_path(name) for name in selected[:4]]
    images = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            return None
        images.append(image)

    target_width = max(img.shape[1] for img in images)
    resized_images = []

    for image in images:
        ratio = target_width / image.shape[1]
        new_height = int(image.shape[0] * ratio)
        resized = cv2.resize(image, (target_width, new_height))
        resized_images.append(resized)

    total_height = sum(img.shape[0] for img in resized_images)
    message_band_height = 70 if custom_message else 0
    strip_height = total_height + message_band_height + (40 if add_border else 0)
    padding = 40 if add_border else 0
    strip = np.full((strip_height, target_width + padding, 3), 255, dtype=np.uint8)
    y_offset = 20 if add_border else 0

    for image in resized_images:
        x_start = 20 if add_border else 0
        strip[y_offset:y_offset + image.shape[0], x_start:x_start + image.shape[1]] = image
        y_offset += image.shape[0]

    if custom_message:
        message_y = strip.shape[0] - message_band_height + 20
        cv2.rectangle(
            strip,
            (0, strip.shape[0] - message_band_height),
            (strip.shape[1] - 1, strip.shape[0] - 1),
            (240, 240, 240),
            -1,
        )
        cv2.rectangle(
            strip,
            (0, strip.shape[0] - message_band_height),
            (strip.shape[1] - 1, strip.shape[0] - 1),
            (0, 0, 0),
            2,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(custom_message, font, 1.0, 2)
        text_x = max(12, (strip.shape[1] - text_size[0]) // 2)
        cv2.putText(strip, custom_message, (text_x, message_y), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    if add_border:
        cv2.rectangle(strip, (0, 0), (strip.shape[1] - 1, strip.shape[0] - 1), (0, 0, 0), 4)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_strip.jpg"
    output_path = os.path.join(photo_folder, filename)
    cv2.imwrite(output_path, strip)

    return filename


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def generate_frames(filter_name):
    misses = 0

    while True:
        success, frame = read_frame()

        if not success or frame is None:
            misses += 1
            # A camera that's temporarily busy/reconnecting shouldn't kill the
            # whole stream forever like it used to - just back off and retry.
            time.sleep(0.3)
            if misses > 50:
                break
            continue

        misses = 0
        frame = apply_filter(frame, filter_name)
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )
        time.sleep(0.03)


@app.route("/video")
def video():
    filter_name = clean_filter_name(request.args.get("filter"))
    return Response(
        generate_frames(filter_name),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture")
def capture():
    filter_name = clean_filter_name(request.args.get("filter"))
    ajax = request.args.get("ajax") == "1"
    filename = save_capture(filter_name)

    if ajax:
        if filename:
            return jsonify({"ok": True, "filename": filename})
        return jsonify({"ok": False, "error": "Camera not available"}), 503

    if filename is None:
        return redirect(url_for("home", error="camera"))

    return redirect(url_for("home"))


@app.route("/")
def home():
    photos = list_photos()
    latest_photo = photos[0] if photos else None

    return render_template(
        "index.html",
        photo=latest_photo,
        photos=photos,
        filters=FILTERS,
        camera_error=request.args.get("error") == "camera",
    )


@app.route("/gallery")
def gallery():
    photos = list_photos()
    return render_template("gallery.html", photos=photos)


@app.route("/download/<filename>")
def download_photo(filename):
    if not is_safe_filename(filename):
        abort(404)

    return send_from_directory(photo_folder, filename, as_attachment=True)


@app.route("/delete/<filename>")
def delete_photo(filename):
    if is_safe_filename(filename):
        os.remove(get_photo_path(filename))

    return redirect(url_for("gallery"))


@app.route("/strip")
def strip_page():
    photos = list_photos()
    return render_template("strip.html", photos=photos)


@app.route("/print_strip")
def print_strip_page():
    strip_files = list_strips()
    latest_strip = None

    if strip_files:
        latest_strip = max(
            strip_files,
            key=lambda name: os.path.getmtime(get_photo_path(name))
        )

    return render_template("print_strip.html", latest_strip=latest_strip)


@app.route("/create_strip", methods=["POST"])
def create_strip():
    selected = request.form.getlist("photos")
    add_border = request.form.get("add_border") == "1"
    custom_message = (request.form.get("message") or "").strip()

    filename = build_strip(selected, add_border, custom_message)

    if filename is None:
        return redirect(url_for("strip_page"))

    return redirect(url_for("gallery"))


@app.route("/create_strip_ajax", methods=["POST"])
def create_strip_ajax():
    """Used by the instant 4-shot burst mode: takes the 4 filenames the
    front end just captured and stitches them into one strip right away."""
    data = request.get_json(silent=True) or {}
    selected = data.get("photos") or []
    add_border = bool(data.get("add_border", True))
    custom_message = (data.get("message") or "").strip()

    filename = build_strip(selected, add_border, custom_message)

    if filename is None:
        return jsonify({"ok": False, "error": "Could not build strip"}), 400

    return jsonify({"ok": True, "strip": filename})


if __name__ == "__main__":
    app.run(debug=True)
