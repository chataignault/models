#!/usr/bin/env python3
"""
Demo script for classical computer vision drone detection pipeline.
Creates synthetic video with moving objects to demonstrate detection capabilities.
"""

import cv2
import numpy as np
import time
from main import VideoProcessor


class SyntheticVideoDemo:
    """Generate synthetic video with moving objects for testing detection pipeline."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
        
        # Moving objects parameters
        self.objects = [
            {
                'type': 'circle',
                'pos': [50, 100],
                'velocity': [2, 1],
                'size': 25,
                'color': (255, 255, 255)
            },
            {
                'type': 'rectangle', 
                'pos': [200, 150],
                'velocity': [1, -0.5],
                'size': 30,
                'color': (200, 200, 200)
            },
            {
                'type': 'circle',
                'pos': [400, 350], 
                'velocity': [-1.5, 0.5],
                'size': 20,
                'color': (180, 180, 180)
            }
        ]
    
    def generate_frame(self) -> np.ndarray:
        """Generate a frame with moving objects."""
        # Create background (sky gradient)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(self.height):
            intensity = int(100 + (y / self.height) * 100)
            frame[y, :] = (intensity, intensity + 20, intensity + 50)
        
        # Add some noise for realistic background variation
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Update and draw moving objects
        for obj in self.objects:
            # Update position
            obj['pos'][0] += obj['velocity'][0]
            obj['pos'][1] += obj['velocity'][1]
            
            # Bounce off walls
            if obj['pos'][0] <= 0 or obj['pos'][0] >= self.width:
                obj['velocity'][0] *= -1
            if obj['pos'][1] <= 0 or obj['pos'][1] >= self.height:
                obj['velocity'][1] *= -1
                
            # Keep in bounds
            obj['pos'][0] = max(0, min(self.width - 1, obj['pos'][0]))
            obj['pos'][1] = max(0, min(self.height - 1, obj['pos'][1]))
            
            # Draw object
            x, y = int(obj['pos'][0]), int(obj['pos'][1])
            size = obj['size']
            
            if obj['type'] == 'circle':
                cv2.circle(frame, (x, y), size, obj['color'], -1)
            elif obj['type'] == 'rectangle':
                cv2.rectangle(frame, 
                            (x - size//2, y - size//2), 
                            (x + size//2, y + size//2), 
                            obj['color'], -1)
        
        self.frame_count += 1
        return frame


def run_synthetic_demo():
    """Run the synthetic video demo."""
    print("=" * 60)
    print("CLASSICAL COMPUTER VISION DEMO")
    print("Drone Detection using MOG2 Background Subtraction")
    print("=" * 60)
    
    demo = SyntheticVideoDemo()
    processor = VideoProcessor(source="synthetic")
    
    # Don't try to initialize capture for synthetic demo
    processor.cap = None
    processor.frame_count = 0
    
    print("Generating synthetic video with moving objects...")
    print("- White objects represent potential drone targets")
    print("- Blue horizon line filters ground-based objects")
    print("- Green boxes show detections above horizon")
    print("- Press 'q' to quit")
    print()
    
    try:
        while True:
            # Generate synthetic frame
            frame = demo.generate_frame()
            
            # Process through detection pipeline
            processed_frame = processor.process_frame(frame)
            
            # Add demo info
            cv2.putText(
                processed_frame,
                "SYNTHETIC DEMO - Moving Objects Detection",
                (10, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Display
            cv2.imshow('Classical CV Demo - Drone Detection', processed_frame)
            
            # Control frame rate
            key = cv2.waitKey(50) & 0xFF  # ~20 FPS
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print("Demo completed")


def run_performance_test():
    """Run performance benchmarking."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTING")
    print("=" * 60)
    
    demo = SyntheticVideoDemo()
    processor = VideoProcessor()
    processor.cap = None
    
    # Warm up
    print("Warming up...")
    for i in range(10):
        frame = demo.generate_frame()
        _ = processor.process_frame(frame)
    
    # Benchmark
    print("Running benchmark...")
    num_frames = 100
    start_time = time.time()
    
    for i in range(num_frames):
        frame = demo.generate_frame()
        _ = processor.process_frame(frame)
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    
    print(f"Processed {num_frames} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.1f}")
    print(f"Average processing time per frame: {1000/fps:.1f} ms")
    
    # Performance analysis
    if fps >= 30:
        print("✓ Real-time performance achieved (30+ FPS)")
    elif fps >= 15:
        print("⚠ Moderate performance (15-30 FPS)")
    else:
        print("⚠ Low performance (<15 FPS) - optimization needed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo classical CV drone detection")
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run performance benchmark instead of interactive demo"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_performance_test()
    else:
        run_synthetic_demo()