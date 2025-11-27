from flask import Flask, request, jsonify
import torch
from yolov5 import YOLOv5
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load YOLOv5 model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
yolo = YOLOv5(MODEL_PATH, device="cpu")

@app.route("/")
def home():
    return "YOLOv5 API is running."

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Convert to OpenCV image
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # YOLOv5 inference
    results = yolo.predict(img, size=640)

    # Convert results
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({"detections": detections})

app.run(host="0.0.0.0", port=10000)
