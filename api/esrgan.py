import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import base64
import io
import os
import time
import requests

# -------------------------------
# Ensure Model Directory Exists
# -------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = "models/realesr-general-x4v3.pth"

# -------------------------------
# Auto Download Model
# -------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading Real-ESRGAN model...")

        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        response = requests.get(url, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        print("Model downloaded:", MODEL_PATH)

download_model()

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = True if torch.cuda.is_available() else False

print("Using device:", device)

# -------------------------------
# Create RealESRGANer
# -------------------------------
# Create the model architecture for realesr-general-x4v3
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=use_half,
    device=device
)
# -------------------------------
# Base64 Helpers
# -------------------------------
def b64_to_image(b64_string):
    decoded = base64.b64decode(b64_string)
    buffer = io.BytesIO(decoded)
    return Image.open(buffer).convert("RGB")

def image_to_b64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# -------------------------------
# Upscale Function
# -------------------------------
def upscale_b64(b64_image):
    img = b64_to_image(b64_image)
    img_np = np.array(img, dtype=np.uint8)

    output_np, _ = upsampler.enhance(img_np, outscale=4)
    out_img = Image.fromarray(output_np)

    os.makedirs("uploads", exist_ok=True)
    output_path = f"uploads/upscaled_{int(time.time())}.png"
    out_img.save(output_path)

    return {
        "file_path": output_path,
        "base64": image_to_b64(out_img)
    }

# -------------------------------
# Test
# -------------------------------
if __name__ == "__main__":
    with open("input.png", "rb") as f:
        b64_input = base64.b64encode(f.read()).decode()

    result = upscale_b64(b64_input)
    print("Saved:", result["file_path"])
    print(result["base64"][:200])
