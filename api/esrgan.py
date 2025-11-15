import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import base64
import io
import os
import time
from config import MODEL_DIR, MODEL_PATH_x2, MODEL_PATH_x4

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = torch.cuda.is_available()

print("Using device:", device)


# ---------------------------
# Base64 <-> Image Converters
# ---------------------------

def b64_to_image(b64_string):
    decoded = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(decoded))
    if img.mode not in ['RGB', 'RGBA']:
        img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
    return img


def image_to_b64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------
# Upscaling Function
# ---------------------------

def upscale_b64(b64_image, scale: int = 2):

    # Select model & weight - use RRDBNet for RealESRGAN models
    if scale == 2:
        model_path = MODEL_PATH_x2
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    elif scale == 4:
        model_path = MODEL_PATH_x4
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        raise ValueError("Unsupported scale. Use scale=2 or scale=4.")

    # Load model
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device
    )

    img = b64_to_image(b64_image)

    # --------- RGBA handling ----------
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        rgb_img = Image.merge('RGB', (r, g, b))

        # RGB upscale
        rgb_np = np.array(rgb_img)
        upscaled_rgb, _ = upsampler.enhance(rgb_np, outscale=scale)

        # Alpha = upscale separately as grayscale
        alpha_np = np.array(a)
        alpha_3ch = np.repeat(alpha_np[:, :, None], 3, axis=2)
        upscaled_alpha_3ch, _ = upsampler.enhance(alpha_3ch, outscale=scale)
        upscaled_alpha = upscaled_alpha_3ch[:, :, 0]

        # Merge RGBA back
        out_img = Image.fromarray(
            np.dstack([upscaled_rgb[:, :, 0],
                       upscaled_rgb[:, :, 1],
                       upscaled_rgb[:, :, 2],
                       upscaled_alpha]).astype(np.uint8),
            mode='RGBA'
        )

    # --------- RGB handling ----------
    else:
        img_np = np.array(img)
        output_np, _ = upsampler.enhance(img_np, outscale=scale)
        out_img = Image.fromarray(output_np)

    # Save file
    os.makedirs("uploads", exist_ok=True)
    output_path = f"uploads/upscaled_{int(time.time())}.png"
    out_img.save(output_path)

    return {
        "file_path": output_path,
        "base64": image_to_b64(out_img)
    }


# ---------------------------
# Test Run
# ---------------------------

if __name__ == "__main__":
    with open("input.png", "rb") as f:
        b64_input = base64.b64encode(f.read()).decode()

    result = upscale_b64(b64_input, scale=2)  # choose 2 or 4
    print("Saved:", result["file_path"])
    print(result["base64"][:200])
