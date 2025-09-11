"""
Refactored video processing pipeline for drone detection system.
Clean, modular implementation using separated components.
"""
import cv2
import numpy as np
import time
import argparse
from typing import Optional, Tuple, Dict, Any
import logging

from config import CONFIG
from motion_detection import MotionDetector
from horizon_detection import HorizonDetector
from tracking_system import PhaseManager, TrackingVisualizer
from visualization import (DebugVisualizer, HorizonVisualizer, FilterVisualizer, 
                         PerformanceMonitor, CompositeDebugView)


class VideoProcessor:
    """
    Main video processing pipeline for drone detection.
    Coordinates all detection and tracking components.
    """
    
    def __init__(self, source: str = "0", output_path: Optional[str] = None):
        """
        Initialize video processor with modular components.
        
        Args:
            source: Video source - "0" for camera, path for video file
            output_path: Optional path to save processed video
        """
        self.source = source
        self.output_path = output_path
        
        # Video capture components
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        
        # Detection and tracking components
        self.motion_detector = MotionDetector()
        self.horizon_detector = HorizonDetector()
        self.phase_manager = PhaseManager()
        
        # Visualization and monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_capture(self) -> bool:
        """
        Initialize video capture source with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to convert to integer for camera index
            source = int(self.source) if self.source.isdigit() else self.source
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Could not open video source: {self.source}")
                return False
            
            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.video.default_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.video.default_height)
            
            # Get actual video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS) or CONFIG.video.default_fps
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video source: {self.source}")
            self.logger.info(f"Resolution: {width}x{height}, FPS: {fps}")
            
            # Initialize video writer if output path specified
            if self.output_path:
                if not self._initialize_writer(width, height, fps):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing capture: {e}")
            return False
    
    def _initialize_writer(self, width: int, height: int, fps: float) -> bool:
        """Initialize video writer for output recording."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*CONFIG.video.output_codec)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            if not self.writer.isOpened():
                self.logger.error(f"Could not initialize video writer for: {self.output_path}")
                return False
            
            self.logger.info(f"Video writer initialized: {self.output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing video writer: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the complete detection pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with all visualizations
        """
        frame_start_time = time.time()
        self.frame_count += 1
        
        try:
            # Step 1: Motion detection
            fg_mask, motion_detections = self.motion_detector.detect_motion_objects(frame)
            
            # Step 2: Horizon detection and filtering
            horizon_line, horizon_confidence = self.horizon_detector.detect_horizon(frame, self.frame_count)
            filtered_detections = self._apply_horizon_filter(frame, motion_detections)
            
            # Step 3: Tracking and phase management
            tracking_results = self.phase_manager.update(filtered_detections, self.frame_count)
            
            # Step 4: Visualization
            processed_frame = self._create_visualization(
                frame.copy(), fg_mask, motion_detections, filtered_detections, 
                horizon_line, horizon_confidence, tracking_results
            )
            
            # Step 5: Performance monitoring
            frame_time = time.time() - frame_start_time
            self.performance_monitor.update(frame_time)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame {self.frame_count}: {e}")
            return frame  # Return original frame on error
    
    def _apply_horizon_filter(self, frame: np.ndarray, detections: list) -> list:
        """Apply horizon filtering to detections."""
        if not self.horizon_detector.horizon_line:
            return detections  # No horizon detected, return all detections
        
        horizon_coords = self.horizon_detector.get_horizon_line_coords(frame.shape[:2])
        if not horizon_coords:
            return detections
        
        # Determine sky side
        sky_side_sign = self.horizon_detector.determine_sky_side(frame, horizon_coords)
        
        # Filter detections
        filtered_detections = []
        (x1, y1), (x2, y2) = horizon_coords
        
        for x, y, w, h, area in detections:
            # Calculate detection center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check which side of horizon the detection is on
            if x2 != x1:
                expected_y = y1 + (y2 - y1) * (center_x - x1) / (x2 - x1)
                side_sign = 1 if center_y > expected_y else -1
            else:
                side_sign = 1 if center_x > x1 else -1
            
            # Keep detection if it's on the sky side (with margin)
            if (expected_y - center_y + CONFIG.horizon.horizon_margin) * sky_side_sign > 0:
                filtered_detections.append((x, y, w, h, area))
        
        return filtered_detections
    
    def _create_visualization(self, frame: np.ndarray, fg_mask: np.ndarray, 
                            motion_detections: list, filtered_detections: list,
                            horizon_line: Optional[Tuple[float, float]], horizon_confidence: float,
                            tracking_results: Dict[str, Any]) -> np.ndarray:
        """Create comprehensive visualization of all detection components."""
        
        # Add crosshair
        TrackingVisualizer.add_crosshair(frame)
        
        # Draw horizon line if detected
        if horizon_line:
            horizon_coords = self.horizon_detector.get_horizon_line_coords(frame.shape[:2])
            if horizon_coords:
                HorizonVisualizer.draw_horizon_line(frame, horizon_coords, horizon_confidence)
        
        # Draw detection filtering results
        FilterVisualizer.draw_filtered_detections(frame, motion_detections, filtered_detections)
        
        # Draw tracking results
        active_targets = tracking_results.get('active_targets', [])
        primary_target = tracking_results.get('primary_target')
        TrackingVisualizer.draw_targets(frame, active_targets, primary_target)
        
        # Draw phase information
        phase_info = tracking_results.get('phase_info', {})
        TrackingVisualizer.draw_phase_info(frame, self.phase_manager, phase_info)
        
        # Add system information
        system_info = {
            'detection_count': len(filtered_detections),
            'frame_count': self.frame_count
        }
        CompositeDebugView.add_system_info(frame, system_info)
        
        # Add performance information
        self.performance_monitor.draw_performance_info(frame)
        
        # Add debug panels
        detected_segments = getattr(self.horizon_detector, 'detected_segments', [])
        frame = CompositeDebugView.create_debug_panel(frame, fg_mask, detected_segments)
        
        return frame
    
    def run(self) -> None:
        """Run the video processing pipeline with proper error handling."""
        if not self.initialize_capture():
            return
        
        self.logger.info("Starting video processing... Press 'q' to quit")
        
        # Create resizable window
        cv2.namedWindow(CONFIG.system.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONFIG.system.window_name, 
                        CONFIG.system.default_window_width, 
                        CONFIG.system.default_window_height)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.info("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Save frame if writer initialized
                if self.writer:
                    self.writer.write(processed_frame)
                
                # Display frame
                cv2.imshow(CONFIG.system.window_name, processed_frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit requested by user")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during video processing: {e}")
        finally:
            self._cleanup()
            self._print_performance_summary()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.cap:
                self.cap.release()
            if self.writer:
                self.writer.release()
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _print_performance_summary(self) -> None:
        """Print performance summary statistics."""
        perf_summary = self.performance_monitor.get_performance_summary()
        
        self.logger.info("Performance Summary:")
        self.logger.info(f"  Average FPS: {perf_summary['fps']:.1f}")
        self.logger.info(f"  Average frame time: {perf_summary['avg_frame_time']:.1f} ms")
        self.logger.info(f"  Min frame time: {perf_summary['min_frame_time']:.1f} ms")
        self.logger.info(f"  Max frame time: {perf_summary['max_frame_time']:.1f} ms")
        self.logger.info(f"  Total frames processed: {self.frame_count}")
        
        if perf_summary['fps'] >= CONFIG.system.target_fps:
            self.logger.info("✓ Real-time performance achieved")
        else:
            self.logger.warning("⚠ Performance below real-time target")
    
    def benchmark_performance(self, num_frames: int = 100) -> Dict[str, float]:
        """
        Run performance benchmark without display.
        
        Args:
            num_frames: Number of frames to process for benchmark
            
        Returns:
            Performance statistics dictionary
        """
        if not self.initialize_capture():
            return {}
        
        self.logger.info(f"Running performance benchmark for {num_frames} frames...")
        
        # Warmup
        warmup_frames = CONFIG.system.warmup_frames
        self.logger.info(f"Warming up with {warmup_frames} frames...")
        for _ in range(warmup_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            _ = self.process_frame(frame)
        
        # Reset performance monitor for benchmark
        self.performance_monitor = PerformanceMonitor()
        
        # Benchmark
        benchmark_start = time.time()
        frames_processed = 0
        
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(f"Video ended at frame {i}")
                break
            
            _ = self.process_frame(frame)
            frames_processed += 1
        
        benchmark_time = time.time() - benchmark_start
        
        # Calculate results
        results = self.performance_monitor.get_performance_summary()
        results['total_benchmark_time'] = benchmark_time
        results['frames_processed'] = frames_processed
        
        self.logger.info("Benchmark Results:")
        self.logger.info(f"  Processed {frames_processed} frames in {benchmark_time:.2f} seconds")
        self.logger.info(f"  Overall FPS: {frames_processed / benchmark_time:.1f}")
        self.logger.info(f"  Average FPS: {results['fps']:.1f}")
        
        self._cleanup()
        return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Refactored Classical Computer Vision Pipeline for Drone Detection"
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
        "--benchmark", 
        action="store_true",
        help="Run performance benchmark instead of interactive mode"
    )
    parser.add_argument(
        "--benchmark-frames", 
        type=int,
        default=100,
        help="Number of frames for benchmark (default: 100)"
    )
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Run without display (for headless operation)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(source=args.source, output_path=args.output)
    
    if args.benchmark:
        # Run benchmark
        results = processor.benchmark_performance(args.benchmark_frames)
        return
    
    if args.no_display:
        print("Headless mode not yet implemented")
        return
    
    # Run interactive mode
    processor.run()


if __name__ == "__main__":
    main()