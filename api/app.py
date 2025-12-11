from quart import Quart, request, jsonify
import asyncio
import aiohttp
import base64
import io
import os
import glob
from PIL import Image
from loguru import logger
import time
from api.upscaler_inference import upscale_image_pipeline
from concurrent.futures import ThreadPoolExecutor
from quart_cors import cors
from config import UPLOAD_FOLDER, MAX_FILE_SIZE, MAX_IMAGE_DIMENSION, ALLOWED_EXTENSIONS, CLEANUP_INTERVAL, FILE_MAX_AGE
app = Quart(__name__)
cors(app, allow_origin="*")




executor = ThreadPoolExecutor(max_workers=10)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
cleanup_task = None
cleanup_running = True

async def cleanup_old_files():
    global cleanup_running
    logger.info("Starting cleanup background task...")
    
    while cleanup_running:
        try:
            current_time = time.time()
            files_pattern = os.path.join(UPLOAD_FOLDER, "*")
            files_to_check = glob.glob(files_pattern)
            
            deleted_count = 0
            for file_path in files_to_check:
                try:
                    if os.path.isdir(file_path):
                        continue
                    
                    
                    file_mtime = os.path.getmtime(file_path)
                    file_age = current_time - file_mtime
                    
                    if file_age > FILE_MAX_AGE:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path} (age: {file_age:.1f}s)")
                        
                except OSError as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
                    
            if deleted_count > 0:
                logger.info(f"Cleanup completed: deleted {deleted_count} old files")
                
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
        
        # Wait for next cleanup cycle
        await asyncio.sleep(CLEANUP_INTERVAL)

@app.before_serving
async def startup():
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_old_files())
    logger.info("Background cleanup task started")

@app.after_serving
async def shutdown():
    global cleanup_running, cleanup_task
    cleanup_running = False
    
    if cleanup_task:
        try:
            cleanup_task.cancel()
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error stopping cleanup task: {e}")
    
    logger.info("Background tasks stopped")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_size_from_base64(b64_string):
    try:
        decoded = base64.b64decode(b64_string)
        return len(decoded)
    except Exception:
        return 0

def get_image_dimensions(img):
    return img.size  

async def download_image(url: str) -> bytes:
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: Failed to download image")
                
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise Exception(f"Image too large: {content_length} bytes (max: {MAX_FILE_SIZE})")
                
                data = b""
                async for chunk in response.content.iter_chunked(8192):
                    data += chunk
                    if len(data) > MAX_FILE_SIZE:
                        raise Exception(f"Image too large: {len(data)} bytes (max: {MAX_FILE_SIZE})")
                
                return data
    except asyncio.TimeoutError:
        raise Exception("Timeout downloading image")
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}")

def validate_and_prepare_image(image_data: bytes):
    try:
        if len(image_data) > MAX_FILE_SIZE:
            raise Exception(f"Image too large: {len(image_data)} bytes (max: {MAX_FILE_SIZE})")
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise Exception(f"Image dimensions too large: {width}x{height} (max: {MAX_IMAGE_DIMENSION})")
        
        # Save to temp file for processing
        temp_file = os.path.join(UPLOAD_FOLDER, f"temp_{int(time.time() * 1000)}.jpg")
        img.save(temp_file, format="JPEG", quality=95)
        
        return temp_file, width, height, img.format
    except Exception as e:
        raise Exception(f"Invalid image: {str(e)}")

async def process_upscale(image_path: str, target_resolution: str, enhance_faces: bool = True):
    loop = asyncio.get_event_loop()
    try:
        # Generate output path
        output_file = os.path.join(UPLOAD_FOLDER, f"upscaled_{int(time.time() * 1000)}.jpg")
        result = await loop.run_in_executor(
            executor, 
            upscale_image_pipeline, 
            image_path, 
            output_file,
            target_resolution,
            enhance_faces
        )
        
        # Read upscaled image and convert to base64
        if result["success"]:
            with open(result["file_path"], "rb") as f:
                img_bytes = f.read()
                result["base64"] = base64.b64encode(img_bytes).decode()
        
        return result
    except Exception as e:
        logger.error(f"Upscaling error: {e}")
        raise

async def validate_image_url(url: str) -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url, allow_redirects=True) as response:
                if response.status != 200:
                    return False
                
                content_type = response.headers.get('content-type', '').lower()
                valid_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/x-icon']
                
                if not any(vtype in content_type for vtype in valid_types):
                    return False
                
                return True
    except Exception:
        return False

@app.route('/upscale', methods=['POST', 'GET'])
async def upscale_endpoint():
    img_url = ""
    target_resolution = '4k'
    enhance_faces = True
    start_time = time.time()
    
    try:
        if request.method == 'GET':
            img_url = request.args.get('img_url')
            target_resolution = request.args.get('target_resolution', '4k').lower()
            enhance_faces = request.args.get('enhance_faces', 'true').lower() == 'true'
            if not img_url:
                return jsonify({"error": "img_url is required"}), 400
            if target_resolution not in ['2k', '4k', '8k']:
                return jsonify({"error": "Target resolution must be 2k, 4k, or 8k"}), 400
            
            # Validate URL format
            if not isinstance(img_url, str) or not (img_url.startswith('http://') or img_url.startswith('https://')):
                return jsonify({"error": "Invalid img_url format"}), 400
            
            # Validate image URL
            logger.info(f"Validating image URL: {img_url}")
            if not await validate_image_url(img_url):
                return jsonify({"error": "URL does not point to a valid image file"}), 400
            
            # Download image
            try:
                logger.info(f"Downloading image from: {img_url}")
                image_data = await download_image(img_url)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return jsonify({"error": str(e)}), 400

            # Validate and prepare image
            try:
                temp_image_path, width, height, img_format = validate_and_prepare_image(image_data)
                logger.info(f"Image validated: {width}x{height}, format: {img_format}, size: {len(image_data)} bytes")
            except Exception as e:
                logger.error(f"Image validation failed: {e}")
                return jsonify({"error": str(e)}), 400
            
            # Process upscale
            try:
                logger.info(f"Starting upscaling: {width}x{height} -> target: {target_resolution}")
                result = await process_upscale(temp_image_path, target_resolution, enhance_faces)
                
                if not result["success"]:
                    return jsonify({"error": result["error"]}), 400
                
                processing_time = time.time() - start_time
                logger.info(f"Upscaling completed in {processing_time:.2f}s")
                
                return jsonify({
                    "success": True,
                    "file_path": result["file_path"],
                    "base64": result["base64"],
                    "original_size": result["original_size"],
                    "upscaled_size": result["upscaled_size"],
                    "target_resolution": result["target_resolution"],
                    "total_scale": result["total_scale"],
                    "upscaling_strategy": result["upscaling_strategy"],
                    "faces_detected": result["faces_detected"],
                    "faces_enhanced": result["faces_enhanced"],
                    "processing_time": round(processing_time, 2)
                })
                
            except Exception as e:
                logger.error(f"Upscaling failed: {e}")
                return jsonify({"error": f"Upscaling failed: {str(e)}"}), 500
        
        elif request.method == 'POST':
            data = await request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            img_url = data.get('img_url')
            target_resolution = data.get('target_resolution', '4k').lower()
            enhance_faces = data.get('enhance_faces', True)
            if target_resolution not in ['2k', '4k', '8k']:
                return jsonify({"error": "Target resolution must be 2k, 4k, or 8k"}), 400
            if not img_url:
                return jsonify({"error": "img_url is required"}), 400
            if not isinstance(img_url, str) or not (img_url.startswith('http://') or img_url.startswith('https://')):
                return jsonify({"error": "Invalid img_url format"}), 400
                  
            logger.info(f"Validating image URL: {img_url}")
            if not await validate_image_url(img_url):
                return jsonify({"error": "URL does not point to a valid image file"}), 400
            
            try:
                logger.info(f"Downloading image from: {img_url}")
                image_data = await download_image(img_url)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return jsonify({"error": str(e)}), 400

            try:
                temp_image_path, width, height, img_format = validate_and_prepare_image(image_data)
                logger.info(f"Image validated: {width}x{height}, format: {img_format}, size: {len(image_data)} bytes")
            except Exception as e:
                logger.error(f"Image validation failed: {e}")
                return jsonify({"error": str(e)}), 400
            
            try:
                logger.info(f"Starting upscaling: {width}x{height} -> target: {target_resolution}")
                result = await process_upscale(temp_image_path, target_resolution, enhance_faces)
                
                if not result["success"]:
                    return jsonify({"error": result["error"]}), 400
                
                processing_time = time.time() - start_time
                logger.info(f"Upscaling completed in {processing_time:.2f}s")
                
                return jsonify({
                    "success": True,
                    "file_path": result["file_path"],
                    "base64": result["base64"],
                    "original_size": result["original_size"],
                    "upscaled_size": result["upscaled_size"],
                    "target_resolution": result["target_resolution"],
                    "total_scale": result["total_scale"],
                    "upscaling_strategy": result["upscaling_strategy"],
                    "faces_detected": result["faces_detected"],
                    "faces_enhanced": result["faces_enhanced"],
                    "processing_time": round(processing_time, 2)
                })
                
            except Exception as e:
                logger.error(f"Upscaling failed: {e}")
                return jsonify({"error": f"Upscaling failed: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in upscale endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
async def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "max_dimension": MAX_IMAGE_DIMENSION,
        "supported_resolutions": ["2k", "4k", "8k"],
        "cleanup_interval_minutes": CLEANUP_INTERVAL / 60,
        "file_max_age_minutes": FILE_MAX_AGE / 60
    })

@app.route('/status', methods=['GET'])
async def status():
    upload_stats = {"total_files": 0, "total_size_mb": 0}
    try:
        files_pattern = os.path.join(UPLOAD_FOLDER, "*")
        files = glob.glob(files_pattern)
        upload_stats["total_files"] = len([f for f in files if os.path.isfile(f)])
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        upload_stats["total_size_mb"] = round(total_size / 1024 / 1024, 2)
    except Exception as e:
        logger.warning(f"Error getting upload stats: {e}")
    
    return jsonify({
        "status": "running",
        "thread_pool_workers": executor._max_workers,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "supported_resolutions": ["2k", "4k", "8k"],
        "cleanup_task_running": cleanup_running,
        "upload_folder_stats": upload_stats,
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
            "max_dimension": MAX_IMAGE_DIMENSION,
            "cleanup_interval_minutes": CLEANUP_INTERVAL / 60,
            "file_max_age_minutes": FILE_MAX_AGE / 60
        }
    })

@app.route('/cleanup', methods=['POST'])
async def manual_cleanup():
    try:
        current_time = time.time()
        files_pattern = os.path.join(UPLOAD_FOLDER, "*")
        files_to_check = glob.glob(files_pattern)
        
        deleted_count = 0
        errors = []
        
        for file_path in files_to_check:
            try:
                if os.path.isdir(file_path):
                    continue
                
                file_mtime = os.path.getmtime(file_path)
                file_age = current_time - file_mtime
                
                if file_age > FILE_MAX_AGE:
                    os.remove(file_path)
                    deleted_count += 1
                    
            except OSError as e:
                errors.append(f"Failed to delete {file_path}: {e}")
        
        return jsonify({
            "success": True,
            "deleted_files": deleted_count,
            "errors": errors
        })
        
    except Exception as e:
        return jsonify({"error": f"Manual cleanup failed: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting Quart application...")
    app.run(host='0.0.0.0', port=8000, debug=False, workers=4)