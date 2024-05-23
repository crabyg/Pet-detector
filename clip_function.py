import os
from CLIP_main import clip
import torch
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# Prepare the inputs
def CLIP_function(image):
    # image = Image.open(image_path)
    image = Image.fromarray(image)
    label = ["sit in the driver's seat alone", "sit in the driver's seat with dog"]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(c) for c in label]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # print("Similarity Values:", similarity)

    try:
        values, indices = similarity[0].topk(2)
        # Print the result
        for value, index in zip(values, indices):
            print(f"{label[index]:>16s}: {100 * value.item():.2f}%")
    except RuntimeError as e:
        print("Error:", e)


# image_path = 'CLAHE/105.jpg'