from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO("best.pt")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img)
    detections = results[0].boxes.data.tolist()

    return jsonify({"detections": detections})

app.run(host="0.0.0.0", port=10000)
