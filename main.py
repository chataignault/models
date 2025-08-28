import cv2
import numpy as np
import argparse
import sys
from typing import Optional, Tuple


class VideoProcessor:
    """Video processing pipeline for object detection and tracking."""
    
    def __init__(self, source: str = "0", output_path: Optional[str] = None):
        """
        Initialize video processor.
        
        Args:
            source: Video source - "0" for camera, path for video file
            output_path: Optional path to save processed video
        """
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.frame_count = 0
        
        # Initialize background subtractor (MOG2)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        
        # Detection parameters
        self.min_area = 500  # Minimum blob area
        self.max_area = 10000  # Maximum blob area
        
    def initialize_capture(self) -> bool:
        """Initialize video capture source."""
        try:
            # Try to convert to integer for camera index
            source = int(self.source) if self.source.isdigit() else self.source
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return False
            
            self.cap.set( cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set( cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video source: {self.source}")
            print(f"Resolution: {width}x{height}, FPS: {fps}")
            
            # Initialize video writer if output path specified
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(
                    self.output_path, fourcc, fps, (width, height)
                )
                
            return True
            
        except Exception as e:
            print(f"Error initializing capture: {e}")
            return False
    
    def detect_motion_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Detect moving objects using background subtraction and blob detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detections)
            detections: List of (x, y, w, h, area) tuples
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        processed_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic shape filtering - aspect ratio check for drone-like objects
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for drones
                    detections.append((x, y, w, h, area))
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        processed_frame, 
                        f"Area: {int(area)}", 
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        1
                    )
        
        # Create debug view showing foreground mask
        debug_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        return processed_frame, detections, debug_mask
    
    def apply_horizon_filter(self, frame: np.ndarray, detections: list) -> list:
        """
        Filter detections to keep only objects above horizon line.
        Simplified implementation - assumes horizon at 60% of frame height.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Filtered detections above horizon
        """
        height = frame.shape[0]
        horizon_y = int(height * 0.6)  # Assume horizon at 60% down
        
        # Draw horizon line for visualization
        cv2.line(frame, (0, horizon_y), (frame.shape[1], horizon_y), (255, 0, 0), 2)
        cv2.putText(
            frame, "Horizon", (10, horizon_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        
        # Filter detections above horizon
        filtered_detections = []
        for x, y, w, h, area in detections:
            if y < horizon_y:  # Object center above horizon
                filtered_detections.append((x, y, w, h, area))
                
        return filtered_detections
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the detection pipeline."""
        self.frame_count += 1
        
        # Detect moving objects
        processed_frame, detections, debug_mask = self.detect_motion_objects(frame)
        
        # Apply horizon filtering
        filtered_detections = self.apply_horizon_filter(processed_frame, detections)
        
        # Add frame info
        cv2.putText(
            processed_frame, 
            f"Frame: {self.frame_count} | Detections: {len(filtered_detections)}", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Create composite view for debugging
        h, w = frame.shape[:2]
        debug_resized = cv2.resize(debug_mask, (w//3, h//3))
        
        # Place debug view in corner
        processed_frame[10:10+debug_resized.shape[0], 
                      w-debug_resized.shape[1]-10:w-10] = debug_resized
        
        return processed_frame
    
    def run(self) -> None:
        """Run the video processing pipeline."""
        if not self.initialize_capture():
            return
            
        print("Starting video processing... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Save frame if writer initialized
                if self.writer:
                    self.writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Drone Detection Pipeline', processed_frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Classical Computer Vision Pipeline for Drone Detection"
    )
    parser.add_argument(
        "--source", 
        default="0", 
        help="Video source: '0' for camera, path for video file"
    )
    parser.add_argument(
        "--output", 
        help="Path to save processed video"
    )
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Run without display (for headless operation)"
    )
    
    args = parser.parse_args()
    
    # Check if running in headless mode
    if args.no_display:
        print("Headless mode not implemented yet")
        return
    
    # Initialize and run processor
    processor = VideoProcessor(source=args.source, output_path=args.output)
    processor.run()


if __name__ == "__main__":
    main()
