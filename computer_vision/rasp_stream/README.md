# Raspberry Pi Camera Streaming over SSH

Stream live video from a Raspberry Pi Zero 2 W camera module to your laptop through an SSH tunnel.

## Architecture

- **RPi Server** (`rpi_server.py`): Captures frames from the camera module and streams them over a socket
- **Laptop Client** (`laptop_client.py`): Receives frames through SSH tunnel and displays them using OpenCV
- **SSH Tunnel**: Securely forwards the video stream from RPi to laptop

## Requirements

### Raspberry Pi Zero 2 W

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-picamera2 python3-opencv

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
# Reboot after enabling
```

### Laptop

```bash
# Install OpenCV and other dependencies
pip install opencv-python numpy

# Or using conda
conda install opencv numpy
```

## Setup Instructions

### 1. Transfer Server Script to Raspberry Pi

From your laptop, copy the server script to your RPi:

```bash
scp rpi_server.py pi@raspberrypi.local:~/
```

Replace `raspberrypi.local` with your RPi's IP address if hostname doesn't work.

### 2. Start the Server on Raspberry Pi

SSH into your RPi and run:

```bash
ssh pi@raspberrypi.local
python3 rpi_server.py
```

The server will start listening on port 8888.

### 3. Create SSH Tunnel

From your laptop, create an SSH tunnel to forward the video stream:

```bash
ssh -L 8888:localhost:8888 pi@raspberrypi.local
```

This command:
- Forwards local port 8888 to RPi's port 8888
- Keeps the SSH connection open
- Leave this terminal window open

### 4. Run the Client on Your Laptop

In a new terminal window on your laptop:

```bash
python3 laptop_client.py
```

You should now see the live video stream from your Raspberry Pi camera!

## Usage

### Basic Usage

1. Start server on RPi: `python3 rpi_server.py`
2. Create SSH tunnel: `ssh -L 8888:localhost:8888 pi@raspberrypi.local`
3. Run client on laptop: `python3 laptop_client.py`

### Advanced Options

#### Custom Port

Server:
```bash
# Edit PORT variable in rpi_server.py
PORT = 9999
```

Client:
```bash
python3 laptop_client.py --port 9999
```

SSH Tunnel:
```bash
ssh -L 9999:localhost:9999 pi@raspberrypi.local
```

#### Custom Resolution

Edit `rpi_server.py` and modify the video configuration:

```python
video_config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},  # Change resolution here
    buffer_count=2
)
```

Note: Raspberry Pi Zero 2 W has limited processing power. Higher resolutions will reduce frame rate.

### Controls

- Press `q` in the video window to quit the client
- Press `Ctrl+C` to stop the server

## One-Line Startup (Advanced)

You can run the server and create the SSH tunnel in one command:

```bash
ssh pi@raspberrypi.local "python3 rpi_server.py" &
sleep 2
ssh -L 8888:localhost:8888 -N pi@raspberrypi.local &
python3 laptop_client.py
```

## Troubleshooting

### "Connection refused" error

- Make sure the server is running on the RPi
- Verify the SSH tunnel is active
- Check firewall settings on both devices

### Camera not detected

```bash
# Test camera on RPi
libcamera-hello

# If error, check camera connection
vcgencmd get_camera

# Should show: supported=1 detected=1
```

### Poor performance / low frame rate

- Reduce resolution in `rpi_server.py`
- Close other applications on RPi Zero 2 W
- Ensure good WiFi connection
- Consider using ethernet if available

### "picamera2" module not found

```bash
# On RPi, install picamera2
sudo apt install -y python3-picamera2
```

### SSH tunnel disconnects

Add keep-alive options:
```bash
ssh -L 8888:localhost:8888 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 pi@raspberrypi.local
```

## Performance Tips

1. **Resolution**: Use 640x480 for smooth streaming on RPi Zero 2 W
2. **Network**: Wired ethernet is better than WiFi if available
3. **Buffer**: Adjust `buffer_count` in server config for your needs
4. **Background processes**: Stop unnecessary services on RPi during streaming

## Alternative: Run Server as Background Service

Create a systemd service on RPi:

```bash
sudo nano /etc/systemd/system/camera-stream.service
```

Add:
```ini
[Unit]
Description=Camera Streaming Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi
ExecStart=/usr/bin/python3 /home/pi/rpi_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable camera-stream
sudo systemctl start camera-stream
```

## Security Notes

- This setup uses SSH for encryption and authentication
- The video stream itself is not encrypted (but travels through SSH tunnel)
- Only accessible through SSH authentication
- Server listens on all interfaces (0.0.0.0) but requires SSH tunnel to access from laptop


## On RPi :
```bash
rpicam
```
command for taking photos.
