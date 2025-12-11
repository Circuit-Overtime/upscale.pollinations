import numpy as np
import cv2
from PIL import Image
from multiprocessing.managers import BaseManager
import torch
from loguru import logger
from config import MAX_8K_DIMENSION, MAX_4K_DIMENSION, MAX_2K_DIMENSION, RESOLUTION_TARGETS, UPSCALING_THRESHOLDS

class modelManager(BaseManager):
    pass

modelManager.register("ipcService")

IPC_PORTS = [6002]
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

def parse_target_resolution(target: str) -> int:
    try:
        target_lower = str(target).lower().strip()
        if target_lower in RESOLUTION_TARGETS:
            return RESOLUTION_TARGETS[target_lower]
        pixel_value = int(target)
        if pixel_value > 0:
            return pixel_value
        
        logger.warning(f"Invalid target resolution '{target}', defaulting to 4K")
        return RESOLUTION_TARGETS['4k']
    except (ValueError, TypeError):
        logger.warning(f"Could not parse target resolution '{target}', defaulting to 4K")
        return RESOLUTION_TARGETS['4k']

def validate_upscaling_request(image_width: int, image_height: int, target_resolution: str) -> dict:
    target_lower = str(target_resolution).lower().strip()
    
    if target_lower not in UPSCALING_THRESHOLDS:
        return {
            'allowed': False,
            'reason': f"Unknown target resolution: {target_resolution}",
            'threshold_config': None,
            'max_input_dimension': 0,
            'max_scale_factor': 0,
            'image_max_dimension': max(image_width, image_height)
        }
    
    threshold_config = UPSCALING_THRESHOLDS[target_lower]
    max_input_dimension = threshold_config['max_input_dimension']
    max_scale_factor = threshold_config['max_scale_factor']
    image_max_dimension = max(image_width, image_height)
    if image_max_dimension > max_input_dimension:
        return {
            'allowed': False,
            'reason': f"Image too large for {target_resolution.upper()} upscaling. "
                     f"Max input dimension: {max_input_dimension}px, "
                     f"your image: {image_max_dimension}px. "
                     f"({threshold_config['description']})",
            'threshold_config': threshold_config,
            'max_input_dimension': max_input_dimension,
            'max_scale_factor': max_scale_factor,
            'image_max_dimension': image_max_dimension
        }
    
    return {
        'allowed': True,
        'reason': f"Request meets thresholds for {target_resolution.upper()} upscaling",
        'threshold_config': threshold_config,
        'max_input_dimension': max_input_dimension,
        'max_scale_factor': max_scale_factor,
        'image_max_dimension': image_max_dimension
    }

def calculate_upscale_strategy(image_width: int, image_height: int, target_max_dimension: int) -> dict:
    max_dimension = max(target_max_dimension, MAX_8K_DIMENSION)
    max_current_dimension = max(image_width, image_height)
    if max_current_dimension >= max_dimension:
        logger.info(f"Image already at or exceeds target dimension {max_current_dimension}px >= {max_dimension}px")
        return {
            'can_upscale': False,
            'strategy': [],
            'final_width': image_width,
            'final_height': image_height,
            'total_scale': 1.0,
            'reason': 'Image already meets or exceeds target resolution'
        }
    strategy = []
    current_width = image_width
    current_height = image_height
    total_scale = 1.0
    for pass_num in range(2):
        next_width = current_width * 2
        next_height = current_height * 2
        next_max_dimension = max(next_width, next_height)
        if next_max_dimension <= max_dimension:
            strategy.append(2)
            current_width = next_width
            current_height = next_height
            total_scale *= 2
            logger.info(f"Pass {pass_num + 1}: Can apply 2x upscaling -> {current_width}x{current_height}")
        else:
            logger.info(f"Pass {pass_num + 1}: Cannot apply 2x (would be {next_width}x{next_height}, exceeds {max_dimension}px)")
            break
    if len(strategy) == 0:
        max_scale_width = max_dimension / image_width
        max_scale_height = max_dimension / image_height
        optimal_scale = min(max_scale_width, max_scale_height)
        if optimal_scale >= 2:
            strategy.append(2)
            current_width = image_width * 2
            current_height = image_height * 2
            total_scale = 2.0
            logger.info(f"Using 2x upscaling: {current_width}x{current_height}")
        else:
            logger.warning(f"Image too large for any upscaling (max possible scale: {optimal_scale:.2f}x)")
            return {
                'can_upscale': False,
                'strategy': [],
                'final_width': image_width,
                'final_height': image_height,
                'total_scale': 1.0,
                'reason': f'Image too large for upscaling within limits (max scale: {optimal_scale:.2f}x)'
            }
    return {
        'can_upscale': len(strategy) > 0,
        'strategy': strategy,
        'final_width': current_width,
        'final_height': current_height,
        'total_scale': total_scale,
        'reason': f'Applied {len(strategy)} upscaling pass(es) with total scale {total_scale}x'
    }

def detect_faces(image_np):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return np.array([])

def apply_sequential_upscaling(img_array: np.ndarray, strategy: list, enhance_faces: bool = True) -> np.ndarray:
    current_img = img_array
    for pass_num, scale in enumerate(strategy, 1):
        logger.info(f"Applying upscaling pass {pass_num}/{len(strategy)} with scale {scale}x")
        current_img = upscale_with_model_server(current_img, scale, enhance_faces=enhance_faces)
    return current_img

def upscale_with_model_server(img_array: np.ndarray, scale: int, enhance_faces: bool = True) -> np.ndarray:
    try:
        model_service = get_model_server()
        img_data = {
            'data': img_array.tobytes(),
            'shape': img_array.shape,
            'dtype': str(img_array.dtype)
        }
        if scale == 2:
            if enhance_faces:
                logger.info("Upscaling with 2x face enhancement")
                result = model_service.enhance_face_x2(img_data)
            else:
                logger.info("Upscaling with 2x standard enhancement")
                result = model_service.enhance_x2(img_data)
        elif scale == 4:
            if enhance_faces:
                logger.info("Upscaling with 4x face enhancement")
                result = model_service.enhance_face_x4(img_data)
            else:
                logger.info("Upscaling with 4x standard enhancement")
                result = model_service.enhance_x4(img_data)
        else:
            raise ValueError(f"Invalid scale: {scale}. Must be 2 or 4")
        upscaled_img = np.frombuffer(result['data'], dtype=np.uint8).reshape(result['shape'])
        return upscaled_img
    except Exception as e:
        logger.error(f"Error during model server upscaling: {e}")
        raise

def upscale_image_pipeline(image_path: str, output_path: str, target_resolution: str = '4k', enhance_faces: bool = True) -> dict:
    try:
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)
        original_height, original_width = image_np.shape[:2]
        
        logger.info(f"Original image size: {original_width}x{original_height}")
        
        target_max_dimension = parse_target_resolution(target_resolution)
        logger.info(f"Target resolution: {target_resolution} ({target_max_dimension}px max dimension)")
        
        validation_result = validate_upscaling_request(original_width, original_height, target_resolution)
        
        if not validation_result['allowed']:
            logger.warning(f"Request rejected: {validation_result['reason']}")
            return {
                "success": False,
                "error": validation_result['reason'],
                "original_size": {"width": original_width, "height": original_height},
                "target_resolution": target_resolution,
                "target_max_dimension": target_max_dimension,
                "validation_failed": True,
                "threshold_info": {
                    "max_input_dimension": validation_result['max_input_dimension'],
                    "max_scale_factor": validation_result['max_scale_factor'],
                    "description": validation_result['threshold_config']['description'] if validation_result['threshold_config'] else None
                }
            }
        
        strategy_result = calculate_upscale_strategy(
            original_width, original_height, target_max_dimension
        )
        
        if not strategy_result['can_upscale']:
            logger.warning(f"Cannot upscale: {strategy_result['reason']}")
            return {
                "success": False,
                "error": strategy_result['reason'],
                "original_size": {"width": original_width, "height": original_height},
                "target_resolution": target_resolution,
                "target_max_dimension": target_max_dimension
            }
        
        faces = []
        if enhance_faces:
            faces = detect_faces(image_np)
            if len(faces) > 0:
                logger.info(f"Detected {len(faces)} face(s)")
        
        logger.info(f"Applying upscaling strategy: {strategy_result['strategy']}")
        upscaled_img = apply_sequential_upscaling(image_np, strategy_result['strategy'], enhance_faces=(len(faces) > 0))
        
        upscaled_pil = Image.fromarray(upscaled_img.astype(np.uint8))
        upscaled_pil.save(output_path, format="JPEG", quality=95)
        logger.info(f"Upscaled image saved to {output_path}")
        
        return {
            "success": True,
            "file_path": output_path,
            "original_size": {"width": original_width, "height": original_height},
            "upscaled_size": {"width": strategy_result['final_width'], "height": strategy_result['final_height']},
            "target_resolution": target_resolution,
            "target_max_dimension": target_max_dimension,
            "upscaling_strategy": strategy_result['strategy'],
            "total_scale": strategy_result['total_scale'],
            "faces_detected": len(faces),
            "faces_enhanced": enhance_faces and len(faces) > 0,
            "strategy_reason": strategy_result['reason'],
            "threshold_validation": {
                "passed": True,
                "max_input_allowed": validation_result['max_input_dimension'],
                "description": validation_result['threshold_config']['description']
            }
        }
    
    except Exception as e:
        logger.error(f"Upscaling pipeline error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    input_image = "output.jpg"
    output_image = "upscaled_output.jpg"
    result = upscale_image_pipeline(input_image, output_image, target_resolution='8k', enhance_faces=True)
    print(f"Processing complete: {result}")
