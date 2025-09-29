#!/usr/bin/env python3
"""
Refactored demo script for classical computer vision drone detection pipeline.
Creates synthetic video with moving objects to demonstrate detection capabilities.
Uses new modular architecture for improved performance and maintainability.
"""

import cv2
import numpy as np
import time
import logging

from video_processor import VideoProcessor
from config import CONFIG


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
    """Run the synthetic video demo with refactored components."""
    print("=" * 60)
    print("REFACTORED CLASSICAL COMPUTER VISION DEMO")
    print("Drone Detection with Modular Architecture")
    print("=" * 60)
    
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    demo = SyntheticVideoDemo()
    processor = VideoProcessor(source="synthetic")
    
    # Don't initialize capture for synthetic demo
    processor.cap = None
    processor.frame_count = 0
    
    logger.info("Generating synthetic video with moving objects...")
    logger.info("Features:")
    logger.info("- Modular motion detection with configurable parameters")
    logger.info("- Advanced horizon detection with temporal stability")
    logger.info("- Three-phase tracking system (Pre-Lock-On → Ground → Flight)")
    logger.info("- Real-time performance monitoring")
    logger.info("- Comprehensive debug visualizations")
    logger.info("Press 'q' to quit")
    print()
    
    try:
        while True:
            # Generate synthetic frame
            frame = demo.generate_frame()
            
            # Process through refactored detection pipeline
            processed_frame = processor.process_frame(frame)
            
            # Add demo-specific information
            cv2.putText(
                processed_frame,
                "REFACTORED DEMO - Modular Architecture",
                (10, processed_frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                processed_frame,
                f"Target FPS: {CONFIG.system.target_fps} | Synthetic Objects: {len(demo.objects)}",
                (10, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
            
            # Display with configured window name
            cv2.imshow(CONFIG.system.window_name + ' - Demo', processed_frame)
            
            # Control frame rate (faster for demo)
            key = cv2.waitKey(30) & 0xFF  # ~33 FPS
            if key == ord('q'):
                logger.info("Demo quit requested by user")
                break
                
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        cv2.destroyAllWindows()
        logger.info("Demo completed successfully")
        
        # Print performance summary
        processor._print_performance_summary()


def run_performance_test(num_frames: int = None):
    """Run performance benchmarking with refactored components."""
    if num_frames is None:
        num_frames = CONFIG.system.benchmark_frames
    
    print("\n" + "=" * 60)
    print("REFACTORED PERFORMANCE TESTING")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    demo = SyntheticVideoDemo()
    processor = VideoProcessor()
    processor.cap = None
    
    logger.info(f"Running benchmark with {num_frames} frames...")
    logger.info(f"Target performance: {CONFIG.system.target_fps} FPS")
    
    # Use processor's built-in benchmark method for synthetic data
    class SyntheticCapture:
        def __init__(self, demo):
            self.demo = demo
            self.frame_count = 0
            
        def read(self):
            if self.frame_count < num_frames:
                self.frame_count += 1
                return True, self.demo.generate_frame()
            return False, None
    
    # Replace processor's cap with synthetic capture
    processor.cap = SyntheticCapture(demo)
    
    # Run benchmark
    results = processor.benchmark_performance(num_frames)
    
    # Enhanced performance analysis
    if results:
        target_fps = CONFIG.system.target_fps
        actual_fps = results.get('fps', 0)
        
        print("\nPerformance Analysis:")
        if actual_fps >= target_fps:
            print(f"✓ Excellent performance: {actual_fps:.1f} FPS (target: {target_fps} FPS)")
        elif actual_fps >= target_fps * 0.8:
            print(f"⚠ Good performance: {actual_fps:.1f} FPS (target: {target_fps} FPS)")
        elif actual_fps >= target_fps * 0.5:
            print(f"⚠ Moderate performance: {actual_fps:.1f} FPS (target: {target_fps} FPS)")
        else:
            print(f"❌ Poor performance: {actual_fps:.1f} FPS (target: {target_fps} FPS)")
        
        print(f"\nOptimization Status:")
        print(f"- Motion Detection: Optimized with pre-computed kernels")
        print(f"- Horizon Detection: Adaptive parameters with temporal stability")
        print(f"- Tracking System: Efficient multi-target association")
        print(f"- Visualization: Configurable debug panels")
    else:
        logger.error("Benchmark failed to produce results")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refactored demo for classical CV drone detection")
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run performance benchmark instead of interactive demo"
    )
    parser.add_argument(
        "--benchmark-frames", 
        type=int,
        default=CONFIG.system.benchmark_frames,
        help=f"Number of frames for benchmark (default: {CONFIG.system.benchmark_frames})"
    )
    parser.add_argument(
        "--show-config", 
        action="store_true", 
        help="Show current configuration parameters"
    )
    
    args = parser.parse_args()
    
    if args.show_config:
        print("Current Configuration:")
        print("=" * 40)
        config_dict = CONFIG.get_all_params()
        for section, params in config_dict.items():
            print(f"\n[{section.upper()}]")
            for key, value in params.items():
                print(f"  {key}: {value}")
        return
    
    if args.benchmark:
        run_performance_test(args.benchmark_frames)
    else:
        run_synthetic_demo()