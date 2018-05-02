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
    files = request.files.getlist("file")
    result = detection_micro(files)
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
