MODEL_DIR = "models"
MODEL_PATH_x4 = "models/RealESRGAN_x4plus.pth"
MODEL_PATH_x2 = "models/RealESRGAN_x2plus.pth"
FACE_ENHANCER_MODEL = "model_cache/GFPGANv1.4.pth"
NUM_SERVERS = 2
MAX_FILE_SIZE = 7 * 1024 * 1024  
MAX_IMAGE_DIMENSION = 2048 
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
CLEANUP_INTERVAL = 300  #in seconds has been counted
FILE_MAX_AGE = 300  #in seconds has been counted
# Resolution limits
MAX_8K_DIMENSION = 7680  # 8K resolution
MAX_4K_DIMENSION = 3840  # 4K resolution
MAX_2K_DIMENSION = 2048  # 2K resolution
MODEL_URLS = {
    "RealESRGAN_x2plus.pth": "https://github.com/Circuit-Overtime/upscale.pollinations/releases/download/1.0.0/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/Circuit-Overtime/upscale.pollinations/releases/download/1.0.0/RealESRGAN_x4plus.pth"
}