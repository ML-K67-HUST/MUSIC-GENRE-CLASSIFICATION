from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from machine_learning.predict import predict_
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'music' not in request.files:
        return redirect(request.url)
    file = request.files['music']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('predict', filename=filename))
    return redirect(request.url)

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predictions, predictions_stack, max_confidence, confidence_message = predict_(file_path)
    all_predictions = predictions + predictions_stack
    return render_template('results.html', results=all_predictions,confidence_message=confidence_message, confidence=max_confidence)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
