# tasks.py

from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime
from celery import Celery
from celery.utils.log import get_task_logger
import redis
import requests
from openai import OpenAI
from celery_app import make_celery
from typing import List
import json
from dotenv import load_dotenv

INVOICE_DB_PATH = "invoice.json"
load_dotenv()


def load_invoices():
    if not os.path.exists(INVOICE_DB_PATH):
        return []
    with open(INVOICE_DB_PATH, 'r') as f:
        try:
            invoices = json.load(f)
        except json.JSONDecodeError:
            invoices = []
    return invoices


def save_invoices(invoices):
    with open(INVOICE_DB_PATH, 'w') as f:
        json.dump(invoices, f, indent=4)


OPEN_API_KEY = os.getenv('OPENAI_API_KEY')
# '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'
client = OpenAI(api_key=OPEN_API_KEY)

# Initialize logger
logger = get_task_logger(__name__)

# Configure Celery
celery = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Initialize Redis client for task metadata storage using environment variables
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0))
)

# OpenAI API Configuration

MODEL = "gpt-4o"

# Path for storing temporary images
TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

# Ensure the temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)


def extract_table_from_ocr(ocr_text):
    prompt = """
    You are an AI assistant specialized in extracting structured information from OCR-processed text. Your task is to analyze the following OCR-extracted text and create a JSON object with the following structure:
    {
"name": "",
"invoice_no": "",
"GSTIN": "",
"date":
"total_amount": "",
"transporter_name": "",
"vehicle_no": "",
"eway_bill_no": "",
"lr_no": "",
"igst": "",
"cgst": "",
"sgst": "",
"taxable_amount": "",
"rounded_off_-": "",
"rounded_off_+": "",
"products": [
{
"product_details": "",
"hsn_code": "",
"rate": "",
"gst": "",
"nos": "",
"quantity": "",
"unit": "",
"amount": ""
}
]
}
Instructions:

Carefully read the OCR-extracted text provided.
Identify information related to the company name, date, invoice number, GSTIN, total amount, TRANSPORTER NAME,	VEHICLE NO,	EWAY BILL NO, LR NO, IGST, CGST, SGST, TAXABLE AMT.
For each product mentioned, extract details including product description, HSN Code, Rate, GST, NOS (if applicable), quantity, amount, .
Organize this information into the JSON format specified above.
Quantity should always be in "value unit" format. For example - 325.5 Kgs or  32.5 T 
If any information is missing or unclear, use "" as the value.
Ensure that the "products" array contains an object for each product mentioned in the text.
PLEASE DO NOT WRITE ANYTHING ELSE APART FROM THE JSON OBJECT


Please provide the extracted information in the JSON format specified above.
    """

    print(ocr_text)
    response = client.chat.completions.create(model=MODEL,
                                              messages=[
                                                  {"role": "system",
                                                      "content": prompt},
                                                  {"role": "user",
                                                      "content": ocr_text}
                                              ])

    return response.choices[0].message.content


def perform_ocr(image_path):
    # Convert OpenCV image to base64
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Retrieve the OCR API key from environment variables
    OCR_API_KEY = os.getenv('OCR_API_KEY')
    # OCR_API_KEY = "K85136675288957"
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
    OCR_API_URL = os.getenv('OCR_API_URL', 'https://api.ocr.space/parse/image')

    # Send the request to the OCR.space API
    response = requests.post(OCR_API_URL,
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

    return ocr_text


def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('preprocess1')

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


@celery.task(bind=True)
def process_images_task(self, image_paths, bill_no, time_str, date_str, series):
    task_id = self.request.id
    try:
        self.update_state(state='PROCESSING', meta={
                          'progress': 'Starting GPT processing on images'})

        # Combine image paths into a single string
        combined_image_paths: str = ','.join(image_paths)

        self.update_state(state='PROCESSING', meta={
                          'progress': 'Processing images with GPT'})

        # Process images directly with GPT
        gpt_output = extract_table_from_image(combined_image_paths)
        print("GPT OUTPUT "+gpt_output)
        # Update progress
        self.update_state(state='PROCESSING', meta={
                          'progress': 'GPT processing completed'})

        # Save or store the result as needed
        result = {'gpt_output': gpt_output}

        invoice_data = json.loads(gpt_output)
        invoice_data['series'] = series
        # Load existing invoices
        invoices = load_invoices()

        # Check if invoice_no already exists
        existing_invoice_nos = {invoice['invoice_no'] for invoice in invoices}
        if invoice_data['invoice_no'] in existing_invoice_nos:
            # Invoice already exists, discard new data
            self.update_state(state='REJECTED', meta={
                'progress': 'Invoice already present'})
            logger.info(
                f"Invoice {invoice_data['invoice_no']} already exists. Discarding new data.")
        else:
            # Save the new invoice
            invoices.append(invoice_data)
            save_invoices(invoices)
            logger.info(f"Invoice {invoice_data['invoice_no']} saved.")
            sheets_data = prepare_sheets_data(invoice_data)

            self.update_state(state='PROCESSING', meta={
                'progress': 'Data pushed to Google Sheets'})

        redis_client.hset(f"task:{task_id}", mapping={
            'bill_no': bill_no,
            'time': time_str,
            'date': date_str,
            'state': 'Completed',
            'progress': 'Completed',
            'result': invoice_data['invoice_no'],
            'error': '',
            'image_paths': json.dumps(image_paths)
        })

        # Update task state to SUCCESS
        self.update_state(state='SUCCESS', meta={'result': result})

        return result

    except Exception as e:
        # Error handling
        self.update_state(state='FAILURE', meta={'error': str(e)})
        redis_client.hset(f"task:{task_id}", mapping={
            'bill_no': bill_no,
            'time': time_str,
            'date': date_str,
            'state': 'Failure',
            'progress': 'Failed',
            'result': '',
            'error': str(e),
            'image_paths': json.dumps(image_paths)
        })
        logger.error(f"Task {task_id} failed: {str(e)}")
        raise


def process_images(image_paths: List[str]) -> List[dict]:
    print("IDHAR")
    print(image_paths)
    image_contents = []
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            print(f"Warning: Skipping '{image_path}' as it's not a file.")
            continue

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode('utf-8')
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        except IOError as e:
            print(f"Error reading file '{image_path}': {e}")

    return image_contents


def extract_table_from_image(image_paths):
    prompt = """
    You are an AI assistant specialized in extracting structured information from images. Your task is to analyze the following image(s) and create a JSON object with the following structure:
    {
        "name": "",
        "invoice_no": "",
        "GSTIN": "",
        "date": "",
        "total_amount": "",
        "transporter_name": "",
        "vehicle_no": "",
        "eway_bill_no": "",
        "lr_no": "",
        "igst": "",
        "cgst": "",
        "sgst": "",
        "taxable_amount": "",
        "rounded_off_-": "",
        "rounded_off_+": "",
        "products": [
            {
                "product_details": "",
                "hsn_code": "",
                "rate": "",
                "gst": "",
                "nos": "",
                "quantity": "",
                "unit": "",
                "amount": ""
            }
        ]
    }
    Instructions:
    - Carefully analyze the image(s) provided.
    - Identify information related to the company name, date, invoice number, GSTIN, total amount, TRANSPORTER NAME, VEHICLE NO, EWAY BILL NO, LR NO, IGST, CGST, SGST, TAXABLE AMT.
    - For each product mentioned, extract details including product description, HSN Code, Rate, GST, NOS (if applicable), quantity, amount.
    - Organize this information into the JSON format specified above.
    - Quantity should always be in "value unit" format. For example - 325.5 Kgs or 32.5 T 
    - If any information is missing or unclear, use "" as the value.
    - do not do any mathematical calculations, just extract the information.
    - Ensure that the "products" array contains an object for each product mentioned in the image(s).
    PLEASE DO NOT WRITE ANYTHING ELSE APART FROM THE JSON OBJECT

    Please provide the extracted information in the JSON format specified above BUT DONT PROVIDE JSON CODE, I WANT JSON TEXT.
    """
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze these images and extract the information as per the instructions."}
            ]
        }
    ]

    # Add each image to the messages
    for image_path in image_paths.split(','):
        with open(image_path.strip(), "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2500
    )

    return response.choices[0].message.content


def prepare_sheets_data(invoice_data):
    sheets_data = []

    for product in invoice_data.get("products", []):

        try:
            quantity = float(product.get("quantity", "0").split()[
                             0].replace(',', ''))
            rate = float(product.get("rate", "0").replace(',', ''))
            amount = float(product.get("amount", "0").replace(',', ''))
        except ValueError:
            # If conversion fails, set to 0 and mark as wrong
            quantity, rate, amount = 0, 0, 0

        # Check if quantity * rate = amount
        calculated_amount = quantity * rate
        amount_str = product.get("amount", "0")
        if abs(calculated_amount - amount) > 0.01:  # Allow for small floating-point discrepancies
            amount_str += "_wrong"

        product_data = {
            # You may need to determine how to set this
            "Series": invoice_data.get("series", "A1"),
            "date": invoice_data.get("date", ""),
            "INVOICE NO": invoice_data.get("invoice_no", ""),
            "Purchase Type": "L/GST-18%",  # You may need to determine how to set this
            "party name": invoice_data.get("name", "") if len(sheets_data) == 0 else "",
            "mat.centre": "",  # You may need to determine how to set this
            "item name": product.get("product_details", ""),
            # Extracting numeric quantity
            "qty": product.get("quantity", "").split()[0] if product.get("quantity") else "0",
            # Extracting unit
            "unit": product.get("quantity", "").split()[1] if product.get("quantity") else "",
            "price": product.get("rate", 0),
            "amount": amount_str,
            "NARRATION": "",
            "TRANSACTION TYPE": "Purchase",
            "REASON": "",
            "TRANSPORTER NAME": invoice_data.get("transporter_name", ""),
            "VEHICLE NO": invoice_data.get("vehicle_no", ""),
            "EWAY BILL NO": invoice_data.get("eway_bill_no", ""),
            "LR NO": invoice_data.get("lr_no", ""),
            "IGST": invoice_data.get("igst", ""),
            "CGST": invoice_data.get("cgst", ""),
            "SGST": invoice_data.get("sgst", ""),
            "Discount": 0,
            "CUTTING CHARGES.+ New": 0,
            "PF": 0,
            "TRANSPORT": "",
            "TAXABLE AMT": invoice_data.get("taxable_amount", ""),
            "Rounded Off (-)": invoice_data.get("rounded_off_-", ""),
            "Rounded Off (+)": invoice_data.get("rounded_off_+", ""),
            "FINAL AMOUNT": product.get("amount", 0)
        }

        # Calculate GST amounts if available
        # gst_percentage = product.get("gst", "0").strip('%') / 100
        # if gst_percentage > 0:
        #     taxable_amount = product.get("amount", 0).replace(',', '')
        #     gst_amount = taxable_amount * gst_percentage
        #     product_data["TAXABLE AMT"] = taxable_amount
        #     product_data["CGST"] = gst_amount / 2
        #     product_data["SGST"] = gst_amount / 2
        # Convert gst_percentage to float and handle potential ValueError
        try:
            gst_percentage = float(product.get("gst", "0").strip('%')) / 100

        except ValueError:
            gst_percentage = 0

        if gst_percentage > 0:
            taxable_amount = float(
                str(product.get("amount", "0")).replace(',', ''))
            gst_amount = taxable_amount * gst_percentage
            product_data["TAXABLE AMT"] = taxable_amount
            product_data["CGST"] = gst_amount / 2
            product_data["SGST"] = gst_amount / 2

        push_to_sheets(product_data)
        sheets_data.append(product_data)

    return sheets_data


def push_to_sheets(data):
    # Replace with your actual Google Sheets Web App URL os.getenv('SHEETS_API_URL')
    SHEETS_API_URL = os.getenv('SHEETS_API_URL')
    # SHEETS_API_URL = "https://script.google.com/macros/s/AKfycbzDQ_g_jjtDp5mGGH-LcG0oT5jlNqYFo59N046lEBZ-f5OEn2qqHGGngVpz57fa_kUqww/exec"
    headers = {'Content-Type': 'application/json'}
    print("Pushing to Gsheets")
    response = requests.post(SHEETS_API_URL, json=data, headers=headers)
    print("data" + str(data))
    if response.status_code != 200:
        raise Exception(
            f"Failed to push data to Google Sheets: {response.text}")
    print("Successfully Pushed")


def ensure_temp_folder():
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)


ensure_temp_folder()

# celery -A tasks worker --loglevel=info
