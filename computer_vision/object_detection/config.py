"""
Configuration management for drone detection system.
Contains all parameters and constants used throughout the pipeline.
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MotionDetectionConfig:
    """Configuration for motion detection using background subtraction."""
    # MOG2 Background Subtractor parameters
    bg_history: int = 500
    bg_var_threshold: float = 50.0
    bg_detect_shadows: bool = True
    
    # Blob filtering parameters
    min_area: int = 10
    max_area: int = 10000
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    
    # Morphological operations
    morph_kernel_size: int = 5


@dataclass
class HorizonDetectionConfig:
    """Configuration for horizon line detection."""
    # Canny edge detection
    canny_low_threshold: int = 30
    canny_high_threshold: int = 90
    blur_kernel_size: int = 7
    
    # Hough transform parameters - adaptive based on edge density
    hough_threshold_sparse: int = 10
    hough_threshold_moderate: int = 20
    hough_threshold_normal: int = 40
    
    # Minimum line length percentages of frame dimension
    hough_min_line_sparse: float = 0.02
    hough_min_line_moderate: float = 0.03
    hough_min_line_normal: float = 0.06
    
    # Maximum gap between line segments
    hough_max_gap_sparse: int = 50
    hough_max_gap_moderate: int = 40
    hough_max_gap_normal: int = 30
    
    # Temporal stability parameters
    horizon_history_length: int = 5
    horizon_detection_interval: int = 3
    horizon_persistence_frames: int = 15
    min_confidence_for_update: float = 0.3
    confidence_decay_rate: float = 0.98
    horizon_decay_rate: float = 0.9
    horizon_margin: int = 100
    
    # Line filtering parameters
    min_line_length_pixels: int = 30
    position_filter_top_threshold: float = 0.15
    angle_filter_vertical_min: float = 70.0
    angle_filter_vertical_max: float = 110.0
    
    # Region analysis parameters
    region_margin: int = 15
    min_region_size_ratio: float = 0.05
    min_region_size_pixels: int = 100


@dataclass
class VideoConfig:
    """Configuration for video capture and processing."""
    # Default capture settings
    default_width: int = 320
    default_height: int = 240
    default_fps: int = 30
    
    # Video codec for output
    output_codec: str = 'XVID'
    
    # Debug view parameters
    debug_view_scale: float = 0.25  # Debug views are 1/4 size
    debug_margin: int = 10


@dataclass
class SystemConfig:
    """System-wide configuration parameters."""
    # Performance settings
    target_fps: int = 45  # Minimum target FPS for real-time processing
    benchmark_frames: int = 100
    warmup_frames: int = 10
    
    # UI settings
    window_name: str = 'Drone Detection Pipeline'
    default_window_width: int = 640
    default_window_height: int = 480
    
    # Crosshair settings
    crosshair_size: int = 20
    crosshair_color: tuple = (0, 0, 255)  # Red
    crosshair_thickness: int = 2


class Config:
    """Main configuration class that holds all subsystem configurations."""
    
    def __init__(self):
        self.motion = MotionDetectionConfig()
        self.horizon = HorizonDetectionConfig()
        self.video = VideoConfig()
        self.system = SystemConfig()
    
    def get_all_params(self) -> Dict[str, Any]:
        """Return all configuration parameters as a dictionary."""
        return {
            'motion': self.motion.__dict__,
            'horizon': self.horizon.__dict__,
            'video': self.video.__dict__,
            'system': self.system.__dict__
        }
    
    def update_from_dict(self, params: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for section_name, section_params in params.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_params.items():
                    if hasattr(section, key):
                        setattr(section, key, value)


# Global configuration instance
CONFIG = Config()