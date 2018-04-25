from flask import Flask, render_template, request, jsonify, json
from detecion.detection import detection_micro
import cv2
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/test', methods=['POST'])
def detection():
    file = request.files.get("file")
    path = "static/result_img/" + file.filename
    image, result = detection_micro(file)
    print(result)
    im = Image.fromarray(image)
    im.save(path)
    return jsonify({"path": path, "result": result})


if __name__ == '__main__':
    app.run(debug=True)
