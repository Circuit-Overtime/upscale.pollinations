import torch
import numpy as np
from PIL import Image
import base64
import io
import os
import time
from multiprocessing.managers import BaseManager
from loguru import logger
import threading
from config import MODEL_DIR, MODEL_PATH_x2, MODEL_PATH_x4

os.makedirs(MODEL_DIR, exist_ok=True)

print("Connecting to model servers...")

# ---------------------------
# IPC Model Server Setup
# ---------------------------

class ModelManager(BaseManager): 
    pass

ModelManager.register("ipcService")

# Create connections to multiple model server instances
MODEL_SERVERS = []
NUM_SERVERS = 5  # Total 5 instances
current_server_index = 0
server_lock = threading.Lock()

def initialize_model_servers():
    global MODEL_SERVERS
    base_port = 5002
    
    for i in range(NUM_SERVERS):
        try:
            manager = ModelManager(address=("localhost", base_port + i), authkey=b"ipcService")
            manager.connect()
            MODEL_SERVERS.append(manager.ipcService())
            logger.info(f"Connected to model server on port {base_port + i}")
        except Exception as e:
            logger.error(f"Failed to connect to model server on port {base_port + i}: {e}")

def get_next_server():
    """Round-robin server selection"""
    global current_server_index
    with server_lock:
        if not MODEL_SERVERS:
            raise RuntimeError("No model servers available")
        
        server = MODEL_SERVERS[current_server_index]
        current_server_index = (current_server_index + 1) % len(MODEL_SERVERS)
        return server

# Initialize connections on import
initialize_model_servers()

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
# Upscaling Function with IPC
# ---------------------------

def upscale_b64(b64_image, scale: int = 2):
    if scale not in [2, 4]:
        raise ValueError("Unsupported scale. Use scale=2 or scale=4.")
    
    # Get a model server instance using round-robin
    server = get_next_server()
    
    # Select the appropriate upsampler from the server
    upsampler = server.upsampler_x2 if scale == 2 else server.upsampler_x4
    
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
