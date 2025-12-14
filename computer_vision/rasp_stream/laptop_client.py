#!/usr/bin/env python3
"""
Laptop Client for Raspberry Pi Camera Stream
Receives video stream from Raspberry Pi and displays it.
"""

import socket
import struct
import cv2
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def receive_frame(client_socket, payload_size, data=b""):
    """Receive a complete frame from the socket."""
    # Use leftover data from previous frame

    # Receive message size
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None
        data += packet

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    logger.debug(f"Expecting frame of size: {msg_size} bytes")

    # Receive frame data
    while len(data) < msg_size:
        packet = client_socket.recv(4096)
        if not packet:
            return None
        data += packet

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Decode JPEG frame
    try:
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode JPEG frame")
            return None

        logger.debug(f"Frame received: shape={frame.shape}, dtype={frame.dtype}")
        return frame, data
    except Exception as e:
        logger.error(f"Failed to decode frame: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Camera Stream Client')
    parser.add_argument('--host', default='localhost', help='Host to connect to (default: localhost for SSH tunnel)')
    parser.add_argument('--port', type=int, default=8888, help='Port to connect to (default: 8888)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    HOST = args.host
    PORT = args.port

    logger.info(f"Connecting to {HOST}:{PORT}...")

    # Create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client_socket.connect((HOST, PORT))
        logger.info("Connected successfully!")
        logger.info("Press 'q' to quit")

        payload_size = struct.calcsize("L")
        data = b""
        frame_count = 0

        while True:
            # Receive frame (pass leftover data from previous frame)
            result = receive_frame(client_socket, payload_size, data)
            if result is None:
                logger.warning("Connection lost or failed to receive frame")
                break

            frame, data = result
            frame_count += 1

            # Log frame count periodically
            if frame_count % 30 == 0:
                logger.info(f"Frames received: {frame_count}")

            # Display frame (already in BGR format from JPEG decode)
            cv2.imshow('Raspberry Pi Camera Stream', frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit key pressed")
                break

    except ConnectionRefusedError:
        logger.error(f"Could not connect to {HOST}:{PORT}")
        logger.error("Make sure the SSH tunnel is set up and the server is running on the RPi")
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        logger.info(f"Total frames received: {frame_count}")
        logger.info("Client stopped")


if __name__ == "__main__":
    main()
