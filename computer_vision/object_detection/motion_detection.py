"""
Motion detection module using background subtraction and blob analysis.
Optimized for real-time drone detection with configurable parameters.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import CONFIG


class MotionDetector:
    """
    Detects moving objects using MOG2 background subtraction.
    Optimized for drone detection with aspect ratio and size filtering.
    """
    
    def __init__(self):
        """Initialize motion detector with configured parameters."""
        self.config = CONFIG.motion
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.bg_history,
            varThreshold=self.config.bg_var_threshold,
            detectShadows=self.config.bg_detect_shadows
        )
        
        # Pre-create morphological kernel for performance
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
    
    def detect_motion_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        """
        Detect moving objects in frame using background subtraction.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (foreground_mask, detections)
            detections: List of (x, y, w, h, area) tuples for valid objects
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise (optimized with pre-created kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Find contours with optimized parameters
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and extract valid detections
        detections = self._filter_contours(contours)
        
        return fg_mask, detections
    
    def _filter_contours(self, contours) -> List[Tuple[int, int, int, int, float]]:
        """
        Filter contours based on size and shape criteria suitable for drones.
        
        Args:
            contours: OpenCV contours from findContours
            
        Returns:
            List of valid detections as (x, y, w, h, area) tuples
        """
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - reject too small or too large objects
            if not (self.config.min_area < area < self.config.max_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio filtering for drone-like objects
            if h > 0:  # Avoid division by zero
                aspect_ratio = w / h
                if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                    continue
            else:
                continue
            
            detections.append((x, y, w, h, area))
        
        return detections
    
    def create_debug_visualization(self, fg_mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create debug visualization of motion detection results.
        
        Args:
            fg_mask: Foreground mask from detection
            target_size: (width, height) for debug view
            
        Returns:
            Resized debug visualization as BGR image
        """
        # Convert grayscale mask to BGR for consistency
        debug_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Resize to target size
        debug_resized = cv2.resize(debug_mask_bgr, target_size)
        
        return debug_resized
    
    def reset_background_model(self) -> None:
        """Reset the background subtraction model (useful for scene changes)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.bg_history,
            varThreshold=self.config.bg_var_threshold,
            detectShadows=self.config.bg_detect_shadows
        )