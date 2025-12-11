import numpy as np
import cv2
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

# Model paths
UPSCALER_MODEL_x2 = "model_cache/RealESRGAN_x2plus.pth"
FACE_ENHANCER_MODEL = "model_cache/GFPGANv1.4.pth"
UPSCALE_FACTOR = 2

# Load RealESRGAN x2 upscaler
model_x2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2,
    model_path=UPSCALER_MODEL_x2,
    model=model_x2,
    tile=768,
    tile_pad=0,
    pre_pad=0,
    half=False,
    device="cuda"
)

# Load GFPGAN face enhancer
face_enhancer = GFPGANer(
    model_path=FACE_ENHANCER_MODEL,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler,
    device="cuda"
)

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def upscale_face_region(face_img_np):
    _, _, face_restored = face_enhancer.enhance(
        face_img_np,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    return face_restored

def upscale_background(image_np, outscale=UPSCALE_FACTOR):
    upscaled_np, _ = upsampler.enhance(image_np, outscale=outscale)
    return upscaled_np

def blend_faces(base_img, face_img, x, y, w, h, blend_margin=10):
    x_up, y_up = x * UPSCALE_FACTOR, y * UPSCALE_FACTOR
    w_up, h_up = w * UPSCALE_FACTOR, h * UPSCALE_FACTOR
    mask = np.zeros((h_up, w_up), dtype=np.float32)
    margin = blend_margin * UPSCALE_FACTOR
    cv2.rectangle(mask, (margin, margin), (w_up - margin, h_up - margin), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (margin * 2 + 1, margin * 2 + 1), 0)
    mask = np.stack([mask] * 3, axis=2)
    y1, y2 = y_up, y_up + h_up
    x1, x2 = x_up, x_up + w_up
    
    base_img[y1:y2, x1:x2] = (
        face_img * mask + base_img[y1:y2, x1:x2] * (1 - mask)
    ).astype(np.uint8)
    return base_img

def upscale_image_pipeline(image_path, output_path):
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)
    faces = detect_faces(image_np)
    
    if len(faces) > 0:
        print(f"Detected {len(faces)} face(s). Using face-aware upscaling...")
        base_upscaled = upscale_background(image_np)
        for idx, (x, y, w, h) in enumerate(faces):
            print(f"Processing face {idx + 1}/{len(faces)}...")
            padding = int(max(w, h) * 0.3)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_np.shape[1], x + w + padding)
            y2 = min(image_np.shape[0], y + h + padding)
            
            face_region = image_np[y1:y2, x1:x2]
            face_upscaled = upscale_face_region(face_region)
            x1_up = x1 * UPSCALE_FACTOR
            y1_up = y1 * UPSCALE_FACTOR
            h_face, w_face = face_upscaled.shape[:2]
            base_upscaled[y1_up:y1_up + h_face, x1_up:x1_up + w_face] = face_upscaled
        
        result = base_upscaled
        
    else:
        print("No faces detected. Using standard RealESRGAN upscaling...")
        result = upscale_background(image_np)
    
    # Save result
    result_pil = Image.fromarray(result)
    result_pil.save(output_path, format="JPEG", quality=100)
    print(f"Upscaled image saved to {output_path}")
    
    return result_pil

# Example usage
if __name__ == "__main__":
    input_image = "original_image.jpg"
    output_image = "upscaled_output.jpg"
    
    upscaled = upscale_image_pipeline(input_image, output_image)
    print("Processing complete!")