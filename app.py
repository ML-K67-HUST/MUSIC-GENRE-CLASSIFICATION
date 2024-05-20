from flask import Flask, render_template, request, jsonify, session
import os
from werkzeug.utils import secure_filename
import warnings

from machine_learning.predict import predict_

warnings.filterwarnings('ignore')  # Suppress warnings

app = Flask(__name__)
# Set the upload folder path (replace with your actual path)
app.config['UPLOAD_FOLDER'] = '/home/khangpt/MUSIC-GEN-PROJ/user_song'
app.config['SECRET_KEY'] = '123'  # Required for using sessions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Store filename in session for prediction route (alternative approaches possible)
        session['uploaded_filename'] = filename
        return jsonify({'message': 'Song uploaded successfully!'})

    return jsonify({'error': 'Failed to upload file'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve uploaded filename (assuming it's stored in session)
    filename = session.get('uploaded_filename')

    # Alternative: retrieve filename from request object (if not using session)
    # if not filename:
    #     filename = request.args.get('filename')  # Assuming filename passed as query param

    if not filename:
        return jsonify({'error': 'Missing uploaded filename'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predictions = predict_(file_path)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
