import cv2
import sys
from argparse import ArgumentParser

def view_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.2f} seconds")
    print("\nPress 'q' to quit, SPACE to pause/resume")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
        
        cv2.imshow('Video Player', frame)
        
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)

    args = parser.parse_args()

    video_file = args.path
    view_video(video_file)

