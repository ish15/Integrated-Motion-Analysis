import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf 
import cv2
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'D:\Academics 5 sem\BTP-2\Model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Configure the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess video and make predictions
def process_video_with_model(video_path):
    # Specify the height and width. \\\
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    SEQUENCE_LENGTH = 20

    # Initialize the VideoCapture object to read frames from the video file.
    video_reader = cv2.VideoCapture(video_path)

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Get the number of frames in the video using the cv2.CAP_PROP_FRAME_COUNT property from the video_reader object.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
 
    # Iterating the number of times equal to the fixed length of the sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    # Perform prediction using the model
    frames_array = np.array(frames_list)
    frames_array = np.expand_dims(frames_array, axis=0)  # Add batch dimension
    predicted_labels_probabilities = model.predict(frames_array)[0]
    predicted_label = np.argmax(predicted_labels_probabilities)

    # class_names = ["Class1", "Class2", "Class3"] 
    CLASSES_LIST = ["BenchPress", "CleanAndJerk", "Fencing", "HulaHoop","JumpRope","JumpingJack"]
    predicted_class = CLASSES_LIST[predicted_label]
    confidence = predicted_labels_probabilities[predicted_label]
    
    return predicted_class, confidence

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    video = request.files['video']

    if video.filename == '':
        return redirect(request.url)

    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        result_class, confidence = process_video_with_model(video_path)
        return render_template('result.html', result_class=result_class, confidence=confidence)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
