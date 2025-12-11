from multiprocessing.managers import BaseManager
import threading
import os
import signal
from basicsr.archs.rrdbnet_arch import RRDBNet
from config import MODEL_PATH_x2, MODEL_PATH_x4, FACE_ENHANCER_MODEL
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import torch
import numpy as np
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = torch.cuda.is_available()

class ipcModules:
    def __init__(self):
        logger.info("Loading upscaler and face enhancement models...")
        
        # Initialize RealESRGAN models
        model_x2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_x4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        self.upsampler_x4 = RealESRGANer(
            scale=4,
            model_path=MODEL_PATH_x4,
            model=model_x4,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        self.upsampler_x2 = RealESRGANer(
            scale=2,
            model_path=MODEL_PATH_x2,
            model=model_x2,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        
        self.face_enhancer_x2 = GFPGANer(
            model_path=FACE_ENHANCER_MODEL,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler_x2,
            device=device
        )
        
        self.face_enhancer_x4 = GFPGANer(
            model_path=FACE_ENHANCER_MODEL,
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler_x4,
            device=device
        )
        
        logger.info("All models loaded successfully")

    def enhance_x2(self, img_data: dict, outscale=2):
        try:
            img_array = np.frombuffer(img_data['data'], dtype=np.uint8).reshape(img_data['shape'])
            upscaled_img, _ = self.upsampler_x2.enhance(img_array, outscale=outscale)
            return {
                'data': upscaled_img.tobytes(),
                'shape': upscaled_img.shape,
                'dtype': str(upscaled_img.dtype)
            }
        except Exception as e:
            logger.error(f"Error in x2 enhancement: {e}")
            raise

    def enhance_x4(self, img_data: dict, outscale=4):
        try:
            img_array = np.frombuffer(img_data['data'], dtype=np.uint8).reshape(img_data['shape'])
            upscaled_img, _ = self.upsampler_x4.enhance(img_array, outscale=outscale)
            return {
                'data': upscaled_img.tobytes(),
                'shape': upscaled_img.shape,
                'dtype': str(upscaled_img.dtype)
            }
        except Exception as e:
            logger.error(f"Error in x4 enhancement: {e}")
            raise

    def enhance_face_x2(self, img_data: dict):
        try:
            img_array = np.frombuffer(img_data['data'], dtype=np.uint8).reshape(img_data['shape'])
            _, _, face_restored = self.face_enhancer_x2.enhance(
                img_array,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            return {
                'data': face_restored.tobytes(),
                'shape': face_restored.shape,
                'dtype': str(face_restored.dtype)
            }
        except Exception as e:
            logger.error(f"Error in x2 face enhancement: {e}")
            raise

    def enhance_face_x4(self, img_data: dict):
        try:
            img_array = np.frombuffer(img_data['data'], dtype=np.uint8).reshape(img_data['shape'])
            _, _, face_restored = self.face_enhancer_x4.enhance(
                img_array,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            # Serialize back
            return {
                'data': face_restored.tobytes(),
                'shape': face_restored.shape,
                'dtype': str(face_restored.dtype)
            }
        except Exception as e:
            logger.error(f"Error in x4 face enhancement: {e}")
            raise

    def get_upsampler_x2(self):
        return self.upsampler_x2
    
    def get_upsampler_x4(self):
        return self.upsampler_x4
    
    def get_face_enhancer_x2(self):
        return self.face_enhancer_x2
    
    def get_face_enhancer_x4(self):
        return self.face_enhancer_x4

def shutdown_graceful():
    logger.info("Shutting down model server gracefully...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_graceful()
    os._exit(0)

import threading

def start_server_on_port(port):
    try:
        class modelManager(BaseManager): 
            pass
        
        modelManager.register("ipcService", ipcModules)
        manager = modelManager(address=("localhost", port), authkey=b"ipcService")
        server = manager.get_server()
        logger.info(f"Starting model server on port {port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Error on port {port}: {e}")

if __name__ == "__main__":
    ports = [6002]
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    threads = []
    for port in ports:
        thread = threading.Thread(target=start_server_on_port, args=(port,), daemon=False)
        thread.start()
        threads.append(thread)
        logger.info(f"Spawned thread for port {port}")
    
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        shutdown_graceful()
        logger.info("Shutdown complete.")
