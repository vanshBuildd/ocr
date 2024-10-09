# app.py

import os
import cv2
import base64
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from celery import Celery
from celery.result import AsyncResult
import redis
from datetime import datetime

# Import the task from tasks.py
from tasks import process_images_task, preprocess_image

# Initialize Flask app
app = Flask(__name__)

# Configure Celery
app.config.update(
    CELERY_BROKER_URL=os.getenv(
        'CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.getenv(
        'CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)


# Initialize Celery
celery = Celery(
    app.import_name,
    broker=app.config['CELERY_BROKER_URL'],
    backend=app.config['CELERY_RESULT_BACKEND']
)
celery.conf.update(app.config)

# Initialize Redis client for task metadata storage
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0))
)

# Path for storing temporary images
TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

# Ensure the temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)


def save_base64_image(base64_str, filename):
    image_data = base64.b64decode(base64_str.split(',')[1])
    image_path = os.path.join(TEMP_FOLDER, filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    # Read the image to add timestamp
    image = cv2.imread(image_path)

    # Append timestamp to the uploaded image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_with_timestamp = append_timestamp(image, f"Uploaded: {timestamp}")

    # Save the timestamped image
    cv2.imwrite(image_path, image_with_timestamp)

    return image_path


def append_timestamp(image, text):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Add timestamp text
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    text_position = (10, 10)  # Position at the top-left corner
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    # Convert PIL image back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Serve the index.html from the frontend folder


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/task-list')
def tasks():
    return send_from_directory('frontend', 'tasks.html')


@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    image_base64 = data.get('image')
    should_preprocess = data.get('preprocess', False)

    if image_base64:
        # Save the base64 image to a file
        image_filename = f"uploaded_image_{datetime.now().timestamp()}.png"
        image_path = save_base64_image(image_base64, image_filename)

        if should_preprocess:
            # Preprocess the image if requested
            processed_image = preprocess_image(image_path)
        else:
            # If no preprocessing is requested, just read the original image
            processed_image = cv2.imread(image_path)

        # Encode the processed image to base64
        _, buffer = cv2.imencode('.png', processed_image)
        processed_image_base64 = 'data:image/png;base64,' + \
            base64.b64encode(buffer).decode('utf-8')

        # Return the processed image to the frontend
        return jsonify({'processed_image': processed_image_base64}), 200

    return jsonify({'error': 'No image provided'}), 400
# Endpoint to check the status of a task


@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    task_info = redis_client.hgetall(f"task:{task_id}")
    response = {
        'task_id': task_id,
        'state': task.state,
        'info': {
            'progress': task_info.get(b'progress', b'').decode('utf-8'),
            'result': task_info.get(b'result', b'').decode('utf-8'),
            'error': task_info.get(b'error', b'').decode('utf-8'),
        }
    }
    return jsonify(response)

# Serve static files (CSS, JS, images, etc.)


# Helper function to get and increment Bill No.
def get_next_bill_no():
    current_date = datetime.now().strftime('%Y-%m-%d')
    bill_no_key = f"bill_no:{current_date}"
    # Check if the date has changed
    if not redis_client.exists(bill_no_key):
        # Reset Bill No. for the new day
        redis_client.set(bill_no_key, 1)
    else:
        # Increment Bill No.
        redis_client.incr(bill_no_key)
    # Get the current Bill No.
    bill_no = int(redis_client.get(bill_no_key))
    return bill_no

# Modify the /submit-task endpoint


@app.route('/submit-task', methods=['POST'])
def submit_task():
    data = request.json
    images_base64 = data.get('images')
    series = data.get('series')

    if images_base64 and isinstance(images_base64, list):
        image_paths = []
        time_str = datetime.now().strftime('%H:%M:%S')
        date_str = datetime.now().strftime('%Y-%m-%d')
        # Save each image and collect their paths
        for idx, image_base64 in enumerate(images_base64):
            image_filename = f"""processed_image_{
                datetime.now().timestamp()}_{idx}.png"""
            image_path = os.path.join(TEMP_FOLDER, image_filename)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_base64.split(',')[1]))
            image_paths.append(image_path)

        # Get the Bill No., Time, and Date
        bill_no = get_next_bill_no()
        # time_str = datetime.now().strftime('%H:%M:%S')
        # date_str = datetime.now().strftime('%Y-%m-%d')

        # Enqueue the task with the list of image paths and metadata
        task = process_images_task.delay(
            image_paths, bill_no, time_str, date_str, series)

        # Return the task ID to the frontend
        return jsonify({'task_id': task.id}), 202  # 202 Accepted

    return jsonify({'error': 'No images provided'}), 400


@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('frontend', filename)


@app.route('/tasks', methods=['GET'])
def get_tasks():
    show_today_only = request.args.get('today', 'false').lower() == 'true'
    current_date = datetime.now().strftime('%Y-%m-%d')

    task_keys = redis_client.keys('task:*')
    tasks = []

    for task_key in task_keys:
        task_data = redis_client.hgetall(task_key)
        task_info = {k.decode('utf-8'): v.decode('utf-8')
                     for k, v in task_data.items()}

        task_date = task_info.get('date', '')

        # If filtering for today's tasks, skip tasks not from today
        if show_today_only and task_date != current_date:
            continue

        # Include only the required fields
        tasks.append({
            'task_id': task_key.decode('utf-8').split(':')[1],
            'bill_no': int(task_info.get('bill_no', 0)),
            'time': task_info.get('time', ''),
            'date': task_info.get('date', ''),
            'state': task_info.get('state', 'UNKNOWN'),
            'progress': task_info.get('progress', ''),
            'result': task_info.get('result', ''),
            'error': task_info.get('error', '')
        })

    # Sort tasks by Bill No.
    tasks.sort(key=lambda x: x['bill_no'])

    return jsonify(tasks)


if __name__ == '__main__':
    app.run(debug=True)
