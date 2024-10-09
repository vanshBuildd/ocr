from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

import torch
from IPython.display import display
import time

# Define model ID
model_id = "microsoft/Phi-3-vision-128k-instruct"
# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# Define BitsAndBytes configuration for 4-bit quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# Load model with 4-bit quantization and map to CUDA
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cuda",
#     trust_remote_code=True,
#     torch_dtype="auto",
#     quantization_config=nf4_config,
# )


def model_inference(messages, image, max_token):
    start_time = time.time()

    # Prepare prompt with image token
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process prompt and image for model input
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    # Generate text response using model
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=max_token,
        do_sample=False,
    )
    # Remove input tokens from generated response
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    # Decode generated IDs to text
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    display(image)
    end_time = time.time()
    print("Inference time: {}".format(end_time - start_time))
    # Print the generated response
    return response


# prompt = [{"role": "user", "content": "\nOCR the text of the image."}]

# # Load image from local path
# path_image = "test.jpg"  # Replace with your actual image path
# image = Image.open(path_image)
# # Perform inference
# model_inference(prompt, image, 500)


def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found
# Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def model_inference_gpt4o(prompt_text, image_data_url):
    # Prepare the request payload
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]}
    ]
# Perform inference
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=2000
    )
    # Extract and return the generated response
    return response.choices[0].message['content']


# Example usage
# Print the generated response
# print(response_text)
# prompt = [{"role": "user", "content": "\nOCR the text of the image: <image>."}]
# Example usage
image_path = 'test.jpg'
data_url = local_image_to_data_url(image_path)
prompt_text = "Extract IRN number from this picture:"
response_text = model_inference_gpt4o(prompt_text, data_url)
print(response_text)
