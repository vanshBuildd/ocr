import base64
from azure.ai.openai import AzureOpenAI
from mimetypes import guess_type
import requests
# Define your Azure OpenAI resource endpoint and key
api_base = 'https://test-jkcompany.openai.azure.com/'
api_key = '5d6d3ee2695943e2b4cf0e6e6e3a49f0'
deployment_name = 'test'
api_version = '2023-03-15-preview'  # This might change in the future
# Create a client object
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}openai/deployments/{deployment_name}"
)


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


# Example usage
image_path = 'test.jpg'
data_url = local_image_to_data_url(image_path)

headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
}

ENDPOINT = "https://test-jkcompany.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
prompt_text = "What is the IRN number?"
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful assistant."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url
                }
            }
        ]
    }
]

# Create the JSON payload for the API request
payload = {
    "messages": messages,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
}
# Send request
try:
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    response.raise_for_status()
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")

# Handle the response as needed (e.g., print or process)
print(response.json())
