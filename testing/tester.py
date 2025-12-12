import asyncio
import base64
import os
import requests
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

API_URL = "http://localhost:8000/upscale"
IMAGE_PATH = "output.jpg"

def test_upscale_endpoint():
    if not os.path.exists(IMAGE_PATH):
        print(f"Test image '{IMAGE_PATH}' not found.")
        return

    # Start a simple HTTP server to serve the image locally

    class SilentHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    def run_server():
        httpd = HTTPServer(('localhost', 9000), SilentHandler)
        httpd.serve_forever()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    img_url = "https://nzptsfd.telangana.gov.in/newResources/css/img/DownMenu/1.%20Plain%20Tiger.jpg"
    payload = {
        "img_url": img_url,
        "target_resolution": "4k",
    }

    print("Sending request to /upscale endpoint...")
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        print(f"Status code: {response.status_code}")
        data = response.json()
        print("Response JSON:")
        for k, v in data.items():
            if k == "base64":
                print(f"{k}: [base64 string, length={len(v)}]")
            else:
                print(f"{k}: {v}")
        # Optionally save the upscaled image
        if data.get("base64"):
            with open("upscaled_from_api.jpg", "wb") as f:
                f.write(base64.b64decode(data["base64"]))
            print("Upscaled image saved as upscaled_from_api.jpg")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upscale_endpoint()