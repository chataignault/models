#!/usr/bin/env python3
"""
Raspberry Pi Camera Streaming Server
Captures video from the Raspberry Pi camera and streams it over a socket connection.
"""

import socket
import struct
import time
from picamera2 import Picamera2
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 8888       # Port to listen on

    # Initialize camera
    logger.info("Initializing camera...")
    picam2 = Picamera2()

    # Configure camera for video streaming
    # Using smaller resolution for Raspberry Pi Zero 2 W performance
    video_config = picam2.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"},
        # main={"size": (640, 480), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(video_config)
    picam2.start()

    # Give camera time to initialize
    time.sleep(2)
    logger.info("Camera initialized")

    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    logger.info(f"Server listening on {HOST}:{PORT}")
    logger.info("Waiting for connection...")

    frame_count = 0
    try:
        while True:
            client_socket, addr = server_socket.accept()
            logger.info(f"Connection from {addr}")
            frame_count = 0

            try:
                while True:
                    # Capture frame
                    frame = picam2.capture_array()
                    frame_count += 1

                    # Log frame info periodically
                    if frame_count % 30 == 0:
                        logger.info(f"Sent {frame_count} frames - Frame shape: {frame.shape}, dtype: {frame.dtype}")

                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Encode frame as JPEG (quality: 80 is a good balance)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    result, encoded_frame = cv2.imencode('.jpg', frame_bgr, encode_param)

                    if not result:
                        logger.error("Failed to encode frame")
                        continue

                    data = encoded_frame.tobytes()

                    # Send frame size first, then frame data
                    message_size = struct.pack("L", len(data))
                    client_socket.sendall(message_size + data)

            except (ConnectionResetError, BrokenPipeError):
                logger.info(f"Client {addr} disconnected after {frame_count} frames")
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                client_socket.close()
                logger.info("Waiting for new connection...")

    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
    finally:
        picam2.stop()
        server_socket.close()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
