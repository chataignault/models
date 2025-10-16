"""
Visualization utilities for drone detection system.
Provides debug views, overlays, and performance monitoring displays.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from config import CONFIG


class DebugVisualizer:
    """Creates debug visualizations for various components of the detection pipeline."""
    
    @staticmethod
    def create_edge_debug_view(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create debug view showing Canny edge detection results.
        
        Args:
            frame: Input BGR frame
            target_size: (width, height) for debug view
            
        Returns:
            Resized edge visualization as BGR image
        """
        # Convert to grayscale and apply same preprocessing as horizon detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (CONFIG.horizon.blur_kernel_size, CONFIG.horizon.blur_kernel_size), 0)
        
        # Apply Canny edge detection with configured thresholds
        edges = cv2.Canny(blurred, CONFIG.horizon.canny_low_threshold, CONFIG.horizon.canny_high_threshold)
        
        # Convert to BGR and resize
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_resized = cv2.resize(edges_bgr, target_size)
        
        # Add label
        cv2.putText(edges_resized, "Canny Edges", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return edges_resized
    
    @staticmethod
    def create_hough_debug_view(frame: np.ndarray, detected_segments: List, 
                               target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create debug view showing Hough line detection results.
        
        Args:
            frame: Input BGR frame
            detected_segments: List of detected line segments
            target_size: (width, height) for debug view
            
        Returns:
            Hough lines visualization as BGR image
        """
        h, w = frame.shape[:2]
        debug_w, debug_h = target_size
        
        # Create black canvas
        debug_canvas = np.zeros((debug_h, debug_w, 3), dtype=np.uint8)
        
        # Scale factors
        scale_x = debug_w / w
        scale_y = debug_h / h
        
        # Draw detected horizon segments
        if detected_segments:
            colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 128, 0)]
            
            for i, line in enumerate(detected_segments):
                x1, y1, x2, y2 = line[0]
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                color = colors[i % len(colors)]
                cv2.line(debug_canvas, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        
        # Add label
        cv2.putText(debug_canvas, "Horizon Lines", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(debug_canvas, f"Segments: {len(detected_segments)}", (5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return debug_canvas
    
    @staticmethod
    def create_motion_debug_view(fg_mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create debug view for motion detection (foreground mask).
        
        Args:
            fg_mask: Foreground mask from background subtraction
            target_size: (width, height) for debug view
            
        Returns:
            Motion debug visualization as BGR image
        """
        # Convert to BGR and resize
        debug_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        debug_resized = cv2.resize(debug_mask_bgr, target_size)
        
        # Add label
        cv2.putText(debug_resized, "Motion Mask", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return debug_resized


class HorizonVisualizer:
    """Handles horizon line visualization with confidence-based styling."""
    
    @staticmethod
    def draw_horizon_line(frame: np.ndarray, horizon_coords: Optional[List[Tuple[int, int]]], 
                         confidence: float) -> None:
        """
        Draw horizon line on frame with confidence-based coloring.
        
        Args:
            frame: Input BGR frame to draw on
            horizon_coords: [(x1, y1), (x2, y2)] coordinates or None
            confidence: Confidence score (0.0 to 1.0)
        """
        if not horizon_coords or len(horizon_coords) < 2:
            return
        
        # Color based on confidence: green for high, red for low
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        thickness = 3
        
        # Draw horizon line
        cv2.line(frame, horizon_coords[0], horizon_coords[1], color, thickness)
        
        # Calculate and display angle
        (x1, y1), (x2, y2) = horizon_coords
        if x2 != x1:
            angle_rad = np.arctan((y2 - y1) / (x2 - x1))
            angle_deg = np.degrees(angle_rad)
            if angle_deg > 90:
                angle_deg = 180 - angle_deg
        else:
            angle_deg = 90
        
        # Add horizon information
        label = f"Horizon: conf={confidence:.2f}, angle={abs(angle_deg):.1f}°"
        cv2.putText(frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    @staticmethod
    def draw_horizon_regions(frame: np.ndarray, horizon_coords: List[Tuple[int, int]], 
                           sky_side_sign: int) -> None:
        """
        Draw visual indicators for sky and ground regions.
        
        Args:
            frame: Input BGR frame
            horizon_coords: [(x1, y1), (x2, y2)] horizon line coordinates
            sky_side_sign: +1 or -1 indicating which side is sky
        """
        if len(horizon_coords) < 2:
            return
        
        h, w = frame.shape[:2]
        (x1, y1), (x2, y2) = horizon_coords
        
        # Sample points for region indication
        sample_points = [
            (w // 4, h // 4),      # Top-left quadrant
            (3 * w // 4, h // 4),  # Top-right quadrant
            (w // 4, 3 * h // 4),  # Bottom-left quadrant
            (3 * w // 4, 3 * h // 4)  # Bottom-right quadrant
        ]
        
        for px, py in sample_points:
            # Determine which side of horizon this point is on
            if x2 != x1:
                expected_y = y1 + (y2 - y1) * (px - x1) / (x2 - x1)
                side_sign = 1 if py > expected_y else -1
            else:
                side_sign = 1 if px > x1 else -1
            
            # Draw indicator
            if side_sign == sky_side_sign:
                # Sky side - blue circle
                cv2.circle(frame, (px, py), 8, (255, 128, 0), 2)
                cv2.putText(frame, "S", (px-4, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)
            else:
                # Ground side - brown circle
                cv2.circle(frame, (px, py), 8, (0, 128, 255), 2)
                cv2.putText(frame, "G", (px-4, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 255), 1)


class FilterVisualizer:
    """Visualizes filtering results (horizon filtering, etc.)."""
    
    @staticmethod
    def draw_filtered_detections(frame: np.ndarray, all_detections: List[Tuple[int, int, int, int, float]], 
                               filtered_detections: List[Tuple[int, int, int, int, float]]) -> None:
        """
        Draw all detections with different colors for filtered vs unfiltered.
        
        Args:
            frame: Input BGR frame
            all_detections: All detected objects
            filtered_detections: Objects that passed horizon filtering
        """
        filtered_set = set(filtered_detections)
        
        for detection in all_detections:
            x, y, w, h, area = detection
            
            if detection in filtered_set:
                # Passed filtering - green
                color = (0, 255, 0)
                label_prefix = "✓"
            else:
                # Filtered out - red
                color = (0, 0, 255)
                label_prefix = "✗"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw area label
            label = f"{label_prefix} A:{int(area)}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


class PerformanceMonitor:
    """Monitors and displays performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames to average for FPS calculation
        """
        self.window_size = window_size
        self.frame_times: List[float] = []
        self.current_fps = 0.0
        self.frame_count = 0
    
    def update(self, frame_time: float) -> None:
        """
        Update performance metrics with new frame time.
        
        Args:
            frame_time: Time taken to process this frame (in seconds)
        """
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Keep only recent frame times
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Calculate current FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def draw_performance_info(self, frame: np.ndarray) -> None:
        """Draw performance information on frame."""
        h, w = frame.shape[:2]
        
        # Performance text
        fps_text = f"FPS: {self.current_fps:.1f}"
        frame_text = f"Frame: {self.frame_count}"
        
        # Position in top-right corner
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for better readability
        cv2.rectangle(frame, (w - fps_size[0] - 20, 5), (w - 5, 50), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, fps_text, (w - fps_size[0] - 15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, frame_text, (w - fps_size[0] - 15, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Performance status indicator
        if self.current_fps >= CONFIG.system.target_fps:
            status_color = (0, 255, 0)  # Green - good performance
            status_text = "REAL-TIME"
        elif self.current_fps >= CONFIG.system.target_fps * 0.75:
            status_color = (0, 255, 255)  # Yellow - moderate performance
            status_text = "MODERATE"
        else:
            status_color = (0, 0, 255)  # Red - poor performance
            status_text = "SLOW"
        
        cv2.putText(frame, status_text, (w - fps_size[0] - 15, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics."""
        if not self.frame_times:
            return {'fps': 0.0, 'avg_frame_time': 0.0, 'min_frame_time': 0.0, 'max_frame_time': 0.0}
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        min_time = min(self.frame_times)
        max_time = max(self.frame_times)
        
        return {
            'fps': self.current_fps,
            'avg_frame_time': avg_time * 1000,  # Convert to ms
            'min_frame_time': min_time * 1000,
            'max_frame_time': max_time * 1000
        }


class CompositeDebugView:
    """Creates composite debug views with multiple visualization panels."""
    
    @staticmethod
    def create_debug_panel(frame: np.ndarray, fg_mask: np.ndarray, 
                          detected_segments: List, debug_scale: float = 0.25) -> np.ndarray:
        """
        Create comprehensive debug panel with multiple views.
        
        Args:
            frame: Input BGR frame
            fg_mask: Foreground mask from motion detection
            detected_segments: Horizon line segments
            debug_scale: Scale factor for debug views
            
        Returns:
            Frame with debug panels added
        """
        h, w = frame.shape[:2]
        debug_size = (int(w * debug_scale), int(h * debug_scale))
        margin = CONFIG.video.debug_margin
        
        # Create debug views
        edges_debug = DebugVisualizer.create_edge_debug_view(frame, debug_size)
        hough_debug = DebugVisualizer.create_hough_debug_view(frame, detected_segments, debug_size)
        motion_debug = DebugVisualizer.create_motion_debug_view(fg_mask, debug_size)
        
        # Position debug views on frame
        debug_h, debug_w = debug_size[1], debug_size[0]
        
        # Bottom-left: Canny edges
        frame[h - debug_h - margin:h - margin, margin:margin + debug_w] = edges_debug
        
        # Bottom-center: Hough lines
        center_x = w // 2 - debug_w // 2
        frame[h - debug_h - margin:h - margin, center_x:center_x + debug_w] = hough_debug
        
        # Bottom-right: Motion mask
        frame[h - debug_h - margin:h - margin, w - debug_w - margin:w - margin] = motion_debug
        
        return frame
    
    @staticmethod
    def add_system_info(frame: np.ndarray, system_info: Dict[str, Any]) -> None:
        """Add system information overlay to frame."""
        # System info background
        info_height = 100
        cv2.rectangle(frame, (10, 10), (300, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, info_height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "Drone Detection System", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Version and mode info
        y_pos = 50
        line_height = 15
        
        info_items = [
            f"Mode: Classical CV",
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
            f"Detections: {system_info.get('detection_count', 0)}"
        ]
        
        for item in info_items:
            cv2.putText(frame, item, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += line_height