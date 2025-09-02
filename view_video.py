#!/usr/bin/env python3
"""
Simple video viewer utility.
Updated to use consistent configuration with the refactored system.
"""

import cv2
import sys
from argparse import ArgumentParser
from config import CONFIG

def view_video(video_path: str) -> None:
    """
    View video file with playback controls.
    
    Args:
        video_path: Path to video file
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.default_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video File: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.2f} seconds")
    print("\nControls:")
    print("  SPACE - pause/resume")
    print("  'q' - quit")
    print("  'r' - restart from beginning")
    print()
    
    # Create window with consistent naming
    window_name = 'Video Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CONFIG.system.default_window_width, CONFIG.system.default_window_height)
    
    paused = False
    current_frame = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                current_frame += 1
            
            # Add frame information overlay
            if 'frame' in locals():
                info_text = f"Frame: {current_frame}/{frame_count} | Time: {current_frame/fps:.1f}s"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if paused:
                    cv2.putText(frame, "PAUSED", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord(' '):
                paused = not paused
                status = "paused" if paused else "resumed"
                print(f"Playback {status}")
            elif key == ord('r'):
                print("Restarting video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                paused = False
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video viewer closed")

def main():
    """Main function with improved argument parsing."""
    parser = ArgumentParser(description="Simple video file viewer")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True,
        help="Path to video file to view"
    )
    parser.add_argument(
        "--info-only", 
        action="store_true",
        help="Show video information only (don't play)"
    )
    
    args = parser.parse_args()
    
    if not args.path:
        print("Error: Please specify a video file path with --path")
        sys.exit(1)
    
    if args.info_only:
        # Just show video info without playing
        cap = cv2.VideoCapture(args.path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.default_fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video Information:")
            print(f"  File: {args.path}")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps:.1f}")
            print(f"  Total frames: {frame_count}")
            print(f"  Duration: {frame_count/fps:.2f} seconds")
            cap.release()
        else:
            print(f"Error: Could not open video file {args.path}")
            sys.exit(1)
    else:
        view_video(args.path)


if __name__ == "__main__":
    main()

