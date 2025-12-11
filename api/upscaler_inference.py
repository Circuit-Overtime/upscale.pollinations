import numpy as np
import cv2
from PIL import Image
from multiprocessing.managers import BaseManager
import torch
from loguru import logger
from config import MAX_8K_DIMENSION, MAX_4K_DIMENSION, MAX_2K_DIMENSION

class modelManager(BaseManager):
    pass

modelManager.register("ipcService")

IPC_PORTS = [6002, 6003]
current_port_index = 0

def get_model_server():
    global current_port_index
    port = IPC_PORTS[current_port_index % len(IPC_PORTS)]
    current_port_index += 1
    try:
        manager = modelManager(address=("localhost", port), authkey=b"ipcService")
        manager.connect()
        return manager.ipcService()
    except Exception as e:
        logger.error(f"Failed to connect to model server on port {port}: {e}")
        raise

def calculate_optimal_scale(image_width: int, image_height: int, user_scale: int) -> tuple:
    max_dimension = MAX_8K_DIMENSION
    direct_width = image_width * user_scale
    direct_height = image_height * user_scale
    if max(direct_width, direct_height) <= max_dimension:
        logger.info(f"Using direct {user_scale}x scale: {direct_width}x{direct_height}")
        return user_scale, direct_width, direct_height, True
    max_scale_width = max_dimension / image_width
    max_scale_height = max_dimension / image_height
    optimal_scale = min(max_scale_width, max_scale_height)
    optimal_scale = int(optimal_scale * 2) / 2
    if optimal_scale < 2:
        logger.warning(f"Image too large for 2x upscaling within 8K limit")
        optimal_scale = 1
        can_upscale = False
    else:
        can_upscale = True
    final_width = int(image_width * optimal_scale)
    final_height = int(image_height * optimal_scale)
    logger.info(f"Adjusted scale from {user_scale}x to {optimal_scale}x to fit 8K: {final_width}x{final_height}")
    return optimal_scale, final_width, final_height, can_upscale

def detect_faces(image_np):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return np.array([])

def upscale_with_model_server(img_array: np.ndarray, scale: int, enhance_faces: bool = True) -> np.ndarray:
    try:
        model_service = get_model_server()
        if scale == 2:
            if enhance_faces:
                logger.info("Upscaling with 2x face enhancement")
                upscaled_img = model_service.enhance_face_x2(img_array.tobytes())
                if isinstance(upscaled_img, bytes):
                    height, width = int(img_array.shape[0] * 2), int(img_array.shape[1] * 2)
                    upscaled_img = np.frombuffer(upscaled_img, dtype=np.uint8).reshape((height, width, 3))
            else:
                logger.info("Upscaling with 2x standard enhancement")
                upscaled_img = model_service.enhance_x2(img_array.tobytes())
                if isinstance(upscaled_img, bytes):
                    height, width = int(img_array.shape[0] * 2), int(img_array.shape[1] * 2)
                    upscaled_img = np.frombuffer(upscaled_img, dtype=np.uint8).reshape((height, width, 3))
        elif scale == 4:
            if enhance_faces:
                logger.info("Upscaling with 4x face enhancement")
                upscaled_img = model_service.enhance_face_x4(img_array.tobytes())
                if isinstance(upscaled_img, bytes):
                    height, width = int(img_array.shape[0] * 4), int(img_array.shape[1] * 4)
                    upscaled_img = np.frombuffer(upscaled_img, dtype=np.uint8).reshape((height, width, 3))
            else:
                logger.info("Upscaling with 4x standard enhancement")
                upscaled_img = model_service.enhance_x4(img_array.tobytes())
                if isinstance(upscaled_img, bytes):
                    height, width = int(img_array.shape[0] * 4), int(img_array.shape[1] * 4)
                    upscaled_img = np.frombuffer(upscaled_img, dtype=np.uint8).reshape((height, width, 3))
        else:
            raise ValueError(f"Invalid scale: {scale}. Must be 2 or 4")
        return upscaled_img
    except Exception as e:
        logger.error(f"Error during model server upscaling: {e}")
        raise

def upscale_image_pipeline(image_path: str, output_path: str, scale: int = 2, enhance_faces: bool = True) -> dict:
    try:
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)
        original_height, original_width = image_np.shape[:2]
        logger.info(f"Original image size: {original_width}x{original_height}")
        if scale not in [2, 4]:
            logger.warning(f"Invalid scale {scale}, defaulting to 2")
            scale = 2
        optimal_scale, final_width, final_height, can_upscale = calculate_optimal_scale(
            original_width, original_height, scale
        )
        if not can_upscale:
            logger.error("Image too large to upscale while maintaining 8K limit")
            return {
                "success": False,
                "error": "Image too large for upscaling",
                "original_size": {"width": original_width, "height": original_height}
            }
        faces = []
        if enhance_faces:
            faces = detect_faces(image_np)
            if len(faces) > 0:
                logger.info(f"Detected {len(faces)} face(s)")
        upscaled_img = upscale_with_model_server(image_np, int(optimal_scale), enhance_faces=(len(faces) > 0))
        upscaled_pil = Image.fromarray(upscaled_img.astype(np.uint8))
        upscaled_pil.save(output_path, format="JPEG", quality=95)
        logger.info(f"Upscaled image saved to {output_path}")
        return {
            "success": True,
            "file_path": output_path,
            "original_size": {"width": original_width, "height": original_height},
            "upscaled_size": {"width": final_width, "height": final_height},
            "requested_scale": scale,
            "applied_scale": optimal_scale,
            "faces_detected": len(faces),
            "faces_enhanced": enhance_faces and len(faces) > 0
        }
    except Exception as e:
        logger.error(f"Upscaling pipeline error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    input_image = "original_image.jpg"
    output_image = "upscaled_output.jpg"
    result = upscale_image_pipeline(input_image, output_image, scale=4, enhance_faces=True)
    print(f"Processing complete: {result}")
