import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from openai import OpenAI
from celery_app import make_celery
import requests

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)
celery = make_celery(app)
# Set the API key and model name
MODEL = "gpt-4o"
client = OpenAI(api_key=OPEN_API_KEY)


# Path for storing temporary images
TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

# Ensure the temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# Function to append timestamp on an image


def extract_table_from_ocr(ocr_text):
    prompt = f"""

You are an AI assistant specialized in extracting structured information from OCR-processed text. Your task is to analyze the following OCR-extracted text and create a html table with the following columns: Product Details, Rate, GST, NOS, Quantity, and Amount.

Instructions:
1. Carefully read the OCR-extracted text provided.
2. Identify information related to products, Rates, GST, NOS (if applicable), quantities, and amounts.
3. Organize this information into an HTML table format.
4. If any information is missing or unclear, mark it as "N/A" in the table.
5. PLEASE DO NOT WRITE ANYTHING ELSE APART FROM THE HTML TABLE

Please provide the extracted information in the following HTML table format:

<table border="1">
    <tr>
        <th>Product Details</th>
        <th>HSN Code </th>
        <th>Rate</th>
        <th>GST</th>
        <th>NOS</th>
        <th>Quantity</th>
        <th>Amount</th>
    </tr>
    <tr>
        <td>[Product 1]</td>
        <th>[HSN Code 1] </th>
        <td>[Rate 1]</td>
        <td>[GST 1]</td>
        <td>[NOS 1]</td>
        <td>[Quantity 1]</td>
        <td>[Amount 1]</td>
    </tr>
    <!-- More rows as needed -->
</table>
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ocr_text}
        ],

    )

    return response.choices[0].message.content


def append_timestamp(image, text):
    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Add timestamp text
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    text_position = (10, 10)  # Position at the top-left corner
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    # Convert PIL image back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Preprocess image (convert to grayscale, threshold, blur)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(
        gray, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply median blur
    blurred = cv2.medianBlur(thresh, 3)

    # Append timestamp to the processed image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    processed_with_timestamp = append_timestamp(
        blurred, f"Processed: {timestamp}")

    return processed_with_timestamp

# Decode base64 image to OpenCV format and save it


@celery.task(bind=True)
def process_image_task(self, image_path):
    # Update task state to 'Processing'
    self.update_state(state='PROCESSING', meta={
                      'progress': 'Starting preprocessing'})

    # Preprocess the saved image
    processed_image = preprocess_image(image_path)

    # Perform OCR
    self.update_state(state='PROCESSING', meta={'progress': 'Performing OCR'})
    ocr_text = perform_ocr(processed_image)

    # Process OCR text with GPT
    self.update_state(state='PROCESSING', meta={
                      'progress': 'Processing with GPT'})
    gpt_output = extract_table_from_ocr(ocr_text)

    # Save or store the result as needed
    result = {'gpt_output': gpt_output}

    return result


def perform_ocr(image):
    # Convert OpenCV image to base64
    print("Performing OCR")
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Retrieve the OCR API key from environment variables
    OCR_API_KEY = "K85136675288957"
    if not OCR_API_KEY:
        raise ValueError("OCR_API_KEY environment variable not set.")

    # Prepare the payload for the OCR.space API
    payload = {
        'base64Image': 'data:image/png;base64,' + img_base64,
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': '2',
        'isTable': 'true'
    }

    headers = {
        'apikey': OCR_API_KEY,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # Send the request to the OCR.space API
    response = requests.post('https://api.ocr.space/parse/image',
                             data=payload,
                             headers=headers)
    result = response.json()

    # Check if OCR was successful
    if result.get('IsErroredOnProcessing'):
        error_message = result.get('ErrorMessage', ['Unknown error'])[0]
        raise Exception(f"OCR failed: {error_message}")

    # Extract the OCR text
    parsed_results = result.get('ParsedResults')
    if parsed_results and len(parsed_results) > 0:
        ocr_text = parsed_results[0].get('ParsedText', '')
    else:
        ocr_text = ''
    print(ocr_text)
    return ocr_text


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

# Encode OpenCV image to base64 without changing its size or quality


def encode_image_to_base64(image):
    # Save the processed image to a temporary location to ensure quality
    temp_image_path = os.path.join(TEMP_FOLDER, "processed_image.png")
    cv2.imwrite(temp_image_path, image)

    # Read the image as bytes and encode to base64
    with open(temp_image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    return 'data:image/png;base64,' + img_base64

# Serve the index.html from the frontend folder


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

# Preprocess the image and return the processed image


@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    image_base64 = data.get('image')
    should_preprocess = data.get('preprocess', True)
    print("preprocess", should_preprocess)
    if image_base64:
        # Save the base64 image to a file
        image_filename = "uploaded_image.png"
        image_path = save_base64_image(image_base64, image_filename)

        if should_preprocess:
            # Preprocess the saved image (grayscale, thresholding, blurring)
            processed_image = preprocess_image(image_path)
        else:
            # If preprocessing is not requested, just read the image
            processed_image = cv2.imread(image_path)

        # Encode the processed image back to base64 without altering dimensions or quality
        processed_image_base64 = encode_image_to_base64(processed_image)

        return jsonify({'processed_image': processed_image_base64})

    return jsonify({'error': 'No image provided'}), 400


@app.route('/ocr-api', methods=['POST'])
def ocr_api():
    data = request.json
    image_base64 = data.get('image')

    # Perform OCR here (call your OCR service or library)
    ocr_text = "Simulated OCR output from the image."

    return jsonify({'ocr_text': ocr_text})


@app.route('/gpt-api', methods=['POST'])
def gpt_api():
    data = request.json
    ocr_text = data.get('ocr_text')
    print(ocr_text)
    gpt_output = extract_table_from_ocr(ocr_text)

    return jsonify({'html_output': gpt_output})


@app.route('/result')
def result():
    gpt_output = request.args.get('output', '')
    return f"<h1>GPT Output:</h1><p>{gpt_output}</p>"


@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('frontend', filename)


if __name__ == '__main__':
    app.run(debug=True)
