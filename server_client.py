import socket
import os
from dotenv import load_dotenv

load_dotenv()
pi_ip = os.getenv('ip')  # Raspberry Pi IP
IMAGE_RECEIVE_PORT = 5001  # Port to receive image from Pi
AUDIO_SEND_PORT = 5002     # Port to send audio back to Pi

def receive_image_from_pi(save_path="received_image.jpg", pi_ip=pi_ip, port=IMAGE_RECEIVE_PORT):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            s.listen(1)
            print("Waiting for image from Pi...")
            conn, addr = s.accept()
            print("Connected by", addr)

            with open(save_path, "wb") as f:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    f.write(data)
            conn.close()
        print("Image received and saved at", save_path)
        return save_path
    except Exception as e:
        print("Failed to receive image:", e)
        return None

def send_audio_to_pi(file_path, pi_ip=pi_ip, port=AUDIO_SEND_PORT):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((pi_ip, port))
            # Send file content
            with open(file_path, "rb") as f:
                chunk = f.read(4096)
                while chunk:
                    s.sendall(chunk)
                    chunk = f.read(4096)
        print("Audio sent successfully to Raspberry Pi!")
    except Exception as e:
        print("Failed to send audio:", e)


