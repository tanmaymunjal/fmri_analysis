import clip
import numpy as np
import torch
import os
from PIL import Image
import configparser
import requests
import base64
import io

# Load API key from a configuration file
config = configparser.ConfigParser()
config.read("config.ini")
openai_api_key = config.get("OpenAI", "API_KEY")

# load clip model
clip_model = config.get("CLIP", "MODEL")
model, preprocess = clip.load("ViT-B/32")
model.eval()


# Helper function to recursively find all .bmp images in a directory
def find_images(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".bmp"):
                image_files.append(os.path.join(root, file))
    return image_files


# Function to preprocess and encode all images in a folder
def encode_all_images(folder_path):
    image_files = find_images(folder_path)
    images = [preprocess(Image.open(file)).unsqueeze(0) for file in image_files]
    images_tensor = torch.cat(images)
    with torch.no_grad():
        image_features = model.encode_image(images_tensor).float()
    return image_features, image_files


# Function to compare a specific image with a specific text
def compare_image_with_text(all_image_features, all_image_names, image_name, text):
    try:
        index = all_image_names.index(image_name)
    except ValueError:
        print(f"Error: '{image_name}' not found in the list of images.")
        return

    specific_feature = all_image_features[index].unsqueeze(0)

    # Tokenize and encode text
    text = clip.tokenize([text]).to(specific_feature.device)
    with torch.no_grad():
        text_features = model.encode_text(text).float()

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        text_features, specific_feature, dim=1
    )
    return similarity


# Function to encode the image
def encode_image(image_path):
    with Image.open(image_path) as image:
        with io.BytesIO() as buffer:
            image.convert('RGB').save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_caption(image_path: str, api_key: str = openai_api_key, history=None, attempt=1) -> str:
    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    messages = history if history else []
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please generate a small and informative caption for this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ],
    })
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 25,
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], messages
        else:
            raise Exception(f"API error: {response.status_code} {response.text}")
    except Exception as e:
        if attempt < 5:
            return generate_caption(image_path, api_key, history=messages, attempt=attempt + 1)
        else:
            return f"Failed to generate caption after multiple attempts due to error: {e}", messages

def process_images(folder_path):
    all_image_features, all_image_names = encode_all_images(folder_path)
    results = []
    i = 0
    for image_path in all_image_names:
        i += 1
        print(i)
        try:
            best_caption = None
            best_similarity = -1
            history = []
            for attempt in range(5):
                caption, history = generate_caption(image_path, history=history, attempt=attempt + 1)
                similarity = compare_image_with_text(all_image_features, all_image_names, image_path, caption)
                if similarity is not None and similarity.item() > best_similarity:
                    best_caption = caption
                    best_similarity = similarity.item()
                if best_similarity >= 0.25:
                    break
            results.append((image_path, best_caption))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return results

# Save results
def save_results(results, file_path="captions.txt"):
    with open(file_path, "w") as file:
        for image_path, caption in results:
            file.write(f"{image_path}: {caption}\n")

# Example usage
results = process_images("Stimuli")
save_results(results)
