import os
import threading
import time
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
from ultralytics import YOLO
from utils.stream import frame_generator_for_source, stats
from werkzeug.utils import secure_filename

# ==========================
# CONFIGURATION
# ==========================
UPLOAD_DIR = "static/uploads/input"
OUTPUT_DIR = "static/uploads/output"
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# MODEL LOADING
# ==========================
# !!! ACTION REQUIRED: CHANGE THIS PATH to your action-detection trained model's weights (e.g., best.pt) !!!
CUSTOM_MODEL_PATH = r"C:\path\to\your\action_detection_model\weights\best.pt"

if os.path.exists(CUSTOM_MODEL_PATH):
    print(f"✅ Loading custom trained action-detection model: {CUSTOM_MODEL_PATH}")
    MODEL_PATH = CUSTOM_MODEL_PATH
else:
    # This fallback model will LIKELY ONLY detect 'person' and 'bicycle', NOT actions.
    print("⚠️ Custom action-detection model not found! Using pretrained YOLOv8n instead.")
    MODEL_PATH = "yolov8n.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

# ==========================
# ROUTES
# ==========================

@app.route("/")
def index():
    """Home page displaying project details"""
    project = {
        "title": "Intelligent Surveillance for Metro Safety: A Deep Learning Approach to Preventing Accidents and Enhancing Security",
        "guide": "Dr. Satrughan Kumar",
        "team": [
            {"roll": "2200032922", "name": "B. Sriya Avyakta"},
            {"roll": "2200030941", "name": "P. Jyosnasri"},
            {"roll": "2200031294", "name": "Lahari Chowdary"},
            {"roll": "2200031489", "name": "T. Sai Vinay"}
        ],
        "problem_statement": "Metro systems are critical public infrastructures where human safety is paramount. However, accidents such as passengers falling onto tracks or entering restricted zones remain a concern.",
        "solution": "Deep learning-based real-time video analytics using YOLO and transformer-based architectures to detect risky human behaviors.",
        "research_gap": "Current surveillance systems lack automation and real-time intelligence for effective accident prevention."
    }
    return render_template("index.html", project=project)


@app.route("/live")
def live():
    """Live webcam feed page"""
    return render_template("live.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload video and redirect to stream page"""
    if request.method == "POST":
        if "video" not in request.files:
            return "No file part", 400

        file = request.files["video"]
        if file.filename == "":
            return "No selected file", 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXT:
            return "Unsupported file type", 400

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        print(f"✅ Uploaded file saved at: {save_path}")
        return redirect(url_for("stream_video", filename=filename))
    return render_template("upload.html")


@app.route("/stream/<source>")
def stream_choice(source):
    """Choose stream source (cam or uploaded)"""
    if source == "cam":
        return redirect(url_for("live"))
    return "Invalid source", 404


@app.route("/stream_video/<filename>")
def stream_video(filename):
    """Page that streams the uploaded video"""
    return render_template("stream.html", filename=filename)


@app.route("/video_feed")
def video_feed():
    """Streams MJPEG frames from webcam or uploaded file"""
    src = request.args.get("source", "cam")
    file = request.args.get("file", None)

    if src == "uploaded" and file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file))
        if not os.path.exists(file_path):
            return "File not found", 404
        gen = frame_generator_for_source(model, source="file", file_path=file_path)
    else:
        gen = frame_generator_for_source(model, source="cam", file_path=None)

    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_stats")
def video_stats():
    """Return JSON stats for dashboard updates"""
    with stats["lock"]:
        data = {
            "frames_processed": stats.get("frames_processed", 0),
            "detections_total": stats.get("detections_total", 0),
            "avg_confidence": round(stats.get("avg_confidence", 0.0), 3),
            "last_frame_detections": stats.get("last_frame_detections", 0),
            "fps": round(stats.get("fps", 0.0), 2),
            "action_counts": stats.get("action_counts", {}) # ADDED: Send action/object counts
        }
    return jsonify(data)


@app.route("/dashboard")
def dashboard():
    """Dashboard with charts and performance stats"""
    return render_template("dashboard.html")


@app.route("/download/<filename>")
def download(filename):
    """Download uploaded video file"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


# ==========================
# MAIN ENTRY POINT
# ==========================
if __name__ == "__main__":
    print("\n==============================")
    print("🚀 Starting Flask app")
    print(f"📁 Model Path: {MODEL_PATH}")
    print("==============================\n")

    # The existing detect.py (process_video function) is not run here,
    # but is available for offline processing.
    app.run(debug=True)