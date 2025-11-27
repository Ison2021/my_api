from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLOv5 using Torch Hub (THIS WORKS ON RENDER)
MODEL_PATH = "best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, source="github")

@app.route("/")
def home():
    return "YOLOv5 API running on Render!"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Run YOLOv5 prediction
    results = model(img)

    # Convert output to dictionary
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({"detections": detections})

app.run(host="0.0.0.0", port=10000)
