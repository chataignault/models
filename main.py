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
        self.min_area = 10  # Minimum blob area
        self.max_area = 10000  # Maximum blob area
        
        # Horizon detection parameters
        self.horizon_line = None  # (rho, theta) format from Hough transform
        self.horizon_confidence = 0.0
        self.horizon_history = []  # Store recent horizon detections for smoothing
        self.max_history = 5  # Number of frames to keep for temporal smoothing
        self.horizon_detection_interval = 3  # Run horizon detection every N frames for performance
        self.last_horizon_detection_frame = 0
        self.detected_horizon_segments = []  # Store actual detected line segments for visualization
        
        # Enhanced temporal stability parameters
        self.horizon_persistence_frames = 15  # Keep good horizons longer (avoid panic)
        self.horizon_age = 0  # How many frames since horizon was last updated
        self.min_confidence_for_update = 0.3  # Lowered to allow cluster-based horizons to update
        self.confidence_decay_rate = 0.98  # Slower decay to maintain stability
        
        # HoughLinesP parameters - balanced for horizon detection with noise reduction
        self.hough_threshold_sparse = 8      # Slightly higher to reduce noise
        self.hough_threshold_moderate = 12   # Reduced noise while keeping sensitivity
        self.hough_threshold_normal = 18     # Balanced threshold
        self.hough_min_line_sparse = 0.02    # 2% of min dimension  
        self.hough_min_line_moderate = 0.03  # 3% of min dimension
        self.hough_min_line_normal = 0.06    # 6% of min dimension
        self.hough_max_gap_sparse = 50       # Reasonable gaps for fragmented horizons
        self.hough_max_gap_moderate = 40     # Gap tolerance
        self.hough_max_gap_normal = 30       # Stricter gaps for normal scenarios

        self.current_horizon_coord = []
        self.horizon_decay_rate = .9
        self.horizon_margin = 100

        self.sky_side_sign = 0

    def initialize_capture(self) -> bool:
        """Initialize video capture source."""
        try:
            # Try to convert to integer for camera index
            source = int(self.source) if self.source.isdigit() else self.source
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return False
            
            self.cap.set( cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set( cv2.CAP_PROP_FRAME_HEIGHT, 240)
                
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video source: {self.source}")
            print(f"Resolution: {width}x{height}, FPS: {fps}")
            
            # Initialize video writer if output path specified
            if self.output_path:
                # Use XVID codec which is more widely supported
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(
                    self.output_path, fourcc, fps, (width, height)
                )
                if not self.writer.isOpened():
                    print(f"Error: Could not initialize video writer for: {self.output_path}")
                    return False
                print(f"Video writer initialized: {self.output_path}")
                
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
                    # cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(
                    #     processed_frame, 
                    #     f"Area: {int(area)}", 
                    #     (x, y - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 
                    #     0.5, 
                    #     (0, 255, 0), 
                    #     1
                    # )
        
        # Create debug view showing foreground mask
        debug_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        return processed_frame, detections, debug_mask
    
    def detect_horizon_line(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Detect horizon line using edge detection and Hough transform.
        Works with arbitrary camera orientations (inverted, rotated, etc.).
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of ((rho, theta), confidence) where:
            - rho, theta: Hough line parameters (None if no horizon detected)
            - confidence: Detection confidence score (0.0 to 1.0)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (larger kernel for sparse edge scenarios)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Edge detection tuned for subtle horizon detection
        # Lower thresholds to catch subtle, continuous boundaries like horizons
        # Accept some noise to ensure we don't miss low-contrast horizon lines
        edges = cv2.Canny(blurred, 30, 90)
        
        # Calculate edge density for adaptive parameters
        edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1] * 255)
        
        # # If edge density is low, use more sensitive thresholds
        # if edge_density < 0.01:  # Less than 1% edge pixels
        #     edges = cv2.Canny(blurred, 20, 60)  # More sensitive
        # elif edge_density < 0.03:  # Less than 3% edge pixels  
        #     edges = cv2.Canny(blurred, 30, 90)  # Moderately sensitive
        # else:
        #     edges = cv2.Canny(blurred, 50, 150)  # Standard thresholds
        
        # Adaptive Hough line transform parameters based on edge density
        min_dimension = min(frame.shape[1], frame.shape[0])
        
        # Get shared parameters based on edge density
        threshold, min_line_length, max_line_gap = self._get_hough_parameters(edge_density, min_dimension)
            
        lines = cv2.HoughLinesP(
            edges,
            rho=1.,                    # Distance resolution in pixels
            theta=np.pi/180,          # Angle resolution (full 180° range)
            threshold=threshold,       # Use full threshold value
            minLineLength=min_line_length,  # Use full minimum line length
            maxLineGap=max_line_gap   # Adaptive max gap between line segments
        )
        
        if lines is None:
            return None, 0.0
        
        # Filter for horizon-specific lines: horizontal bias + ground-level focus
        horizon_lines = self._filter_for_horizon_lines(lines, frame.shape)
        
        if not horizon_lines:
            return None, 0.0
        
        # Group aligned line segments for better horizon detection (simplified)
        line_groups = self._group_aligned_lines(horizon_lines)
        
        # Convert line groups to horizon candidates with relaxed scoring
        horizon_candidates = []
        
        for group in line_groups:
            if not group:
                continue
                
            # Calculate representative line parameters for the group
            rho, theta, total_length, confidence = self._analyze_line_group(gray, group)
            
            # More lenient validation - focus on cluster analysis instead
            if confidence > 0.05:  # Lowered threshold
                # Store both the line parameters and the actual segments for clustering
                horizon_candidates.append((rho, theta, confidence, total_length, group))
        
        if not horizon_candidates:
            return None, 0.0
        
        # Use cluster-based selection instead of single best candidate
        best_cluster = self._find_horizon_cluster(horizon_candidates)
        
        if best_cluster is None:
            return None, 0.0
            
        # Extract results from best cluster
        median_rho, median_theta, cluster_confidence, cluster_segments = best_cluster
        
        # Store the actual line segments for visualization
        self.detected_horizon_segments = cluster_segments
        
        return (median_rho, median_theta), cluster_confidence
    
    def _analyze_horizon_regions(self, gray_frame: np.ndarray, rho: float, theta: float, line_length: float, line_coords: tuple = None) -> float:
        """
        Analyze regions above and below a potential horizon line.
        Handles both full-frame and partial/corner horizon lines.
        
        Args:
            gray_frame: Grayscale frame
            rho, theta: Line parameters in Hough space
            line_length: Length of the detected line
            line_coords: (x1, y1, x2, y2) for local analysis around partial horizons
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        h, w = gray_frame.shape
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # For partial/corner horizons, focus on local region around the line
        # if line_coords is not None:
        x1, y1, x2, y2 = line_coords
        
        # Expand region around the detected line for local analysis
        margin_x = max(50, int(line_length * 0.5))  # Expand based on line length
        margin_y = max(50, int(line_length * 0.5))
        
        # Define local region around the line
        min_x = max(0, min(x1, x2) - margin_x)
        max_x = min(w, max(x1, x2) + margin_x)
        min_y = max(0, min(y1, y2) - margin_y)
        max_y = min(h, max(y1, y2) + margin_y)
        
        # Create local coordinate system
        local_frame = gray_frame[min_y:max_y, min_x:max_x]
        local_h, local_w = local_frame.shape
        
        if local_h < 20 or local_w < 20:  # Too small for analysis
            return 0.0
            
        # Adjust line parameters for local coordinate system
        local_y, local_x = np.ogrid[:local_h, :local_w]
        # Convert global coordinates to local
        global_x = local_x + min_x
        global_y = local_y + min_y
        distances = global_x * cos_theta + global_y * sin_theta - rho
            
        # else:
        #     # Full-frame analysis (original approach)
        #     local_frame = gray_frame
        #     local_h, local_w = h, w
        #     local_y, local_x = np.ogrid[:local_h, :local_w]
        #     distances = local_x * cos_theta + local_y * sin_theta - rho
        
        # Create region masks with margin around the line
        margin = 8  # Slightly larger margin for partial horizons
        above_mask = distances < -margin
        below_mask = distances > margin
        
        # For sparse edge scenarios, allow smaller regions
        min_region_size = max(local_h * local_w * 0.05, 100)  # At least 5% or 100 pixels
        
        if np.sum(above_mask) < min_region_size or np.sum(below_mask) < min_region_size:
            return 0.0
        
        # Calculate features for both regions
        above_region = local_frame[above_mask]
        below_region = local_frame[below_mask]
        
        # Simplified confidence calculation - focus on key differentiators
        
        # Primary feature: Edge density difference (sky has fewer edges)
        edges_local = cv2.Canny(local_frame, 25, 75)
        above_edges = np.sum(edges_local[above_mask])
        below_edges = np.sum(edges_local[below_mask])
        above_size = np.sum(above_mask)
        below_size = np.sum(below_mask)
        
        if above_size == 0 or below_size == 0:
            return 0.0
            
        above_edge_density = above_edges / above_size
        below_edge_density = below_edges / below_size
        
        # Edge ratio - higher when regions are clearly different
        edge_diff = abs(above_edge_density - below_edge_density)
        edge_confidence = min(edge_diff * 10, 1.0)  # Scale and cap at 1.0
        
        # Secondary feature: Intensity difference (quick to compute)
        above_mean = np.mean(above_region)
        below_mean = np.mean(below_region)
        intensity_diff = abs(above_mean - below_mean) / 255.0
        
        # Basic length factor (simplified)
        min_length = min(local_w, local_h) * 0.1
        length_confidence = min(line_length / min_length, 1.0) * 0.5
        
        # Simple weighted combination
        confidence = edge_confidence * 0.7 + intensity_diff * 0.2 + length_confidence * 0.1
        
        return min(confidence, 1.0)
    
    def _get_hough_parameters(self, edge_density, min_dimension):
        """
        Get HoughLinesP parameters based on edge density.
        Optimized for detecting small corner horizons.
        """
        if edge_density < 0.01:  # Very sparse edges
            threshold = self.hough_threshold_sparse
            min_line_length = max(10, min_dimension * self.hough_min_line_sparse)  # Minimum 10 pixels
            max_line_gap = self.hough_max_gap_sparse
        elif edge_density < 0.03:  # Moderately sparse edges  
            threshold = self.hough_threshold_moderate
            min_line_length = max(12, min_dimension * self.hough_min_line_moderate)  # Minimum 12 pixels
            max_line_gap = self.hough_max_gap_moderate
        else:  # Normal edge density
            threshold = self.hough_threshold_normal
            min_line_length = max(15, min_dimension * self.hough_min_line_normal)  # Minimum 15 pixels
            max_line_gap = self.hough_max_gap_normal
            
        return threshold, min_line_length, max_line_gap
    
    def _filter_for_horizon_lines(self, lines, frame_shape):
        """
        Filter lines to focus on horizontal lines in ground-level regions.
        Rejects cloud edges and focuses on true horizon candidates.
        """
        h, w = frame_shape[:2]
        horizon_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle in degrees
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:  # Nearly vertical line
                continue
                
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            # Normalize angle to 0-180 range
            if angle_deg < 0:
                angle_deg += 180
            
            # Filter 1: Allow wider range of angles for rolled camera scenarios
            # Accept any reasonably oriented line (not nearly vertical)
            # Vertical lines (70-110°) are unlikely to be horizons
            is_reasonable_angle = not (70 <= angle_deg <= 110)
            
            if not is_reasonable_angle:
                continue
            
            # Filter 2: Flexible position filter for angled horizons
            # Allow horizons anywhere in frame (including corners) but prefer reasonable positions
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            
            # Skip only lines entirely in top 15% (very unlikely to be horizon)
            entirely_in_top = max_y < h * 0.15
            
            if entirely_in_top:
                continue
            
            # Filter 3: Length filter - prefer longer lines (likely horizon segments)
            length = np.sqrt(dx*dx + dy*dy)
            min_length = max(30, w * 0.05)  # At least 5% of frame width or 30 pixels
            
            if length < min_length:
                continue
            
            # Passed all filters - this is a horizon candidate
            horizon_lines.append(line)
        
        return horizon_lines
    
    def _group_aligned_lines(self, lines):
        """
        Simplified line grouping for speed - merge nearby horizontal lines.
        Since we've already filtered for horizontal lines, grouping is simpler.
        """
        if len(lines) == 0:
            return []
        
        # Since lines are already filtered for horizontal orientation,
        # we can use a simpler proximity-based grouping
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
                
            x1, y1, x2, y2 = line1[0]
            
            # Start new group with this line
            group_lines = [line1]
            group_total_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            used.add(i)
            
            # Find nearby lines to merge (simplified distance check)
            center_y1 = (y1 + y2) / 2
            
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                x3, y3, x4, y4 = line2[0]
                center_y2 = (y3 + y4) / 2
                
                # Check if lines are at similar Y positions (horizontal alignment)
                y_distance = abs(center_y1 - center_y2)
                if y_distance < 20:  # Within 20 pixels vertically
                    group_lines.append(line2)
                    group_total_length += np.sqrt((x4-x3)**2 + (y4-y3)**2)
                    used.add(j)
            
            # Only keep groups with reasonable length
            if group_total_length > 40:  # Minimum total length
                groups.append(group_lines)
        
        # Sort by total group length (longest first) 
        groups.sort(key=lambda g: sum(np.sqrt((line[0][2]-line[0][0])**2 + (line[0][3]-line[0][1])**2) for line in g), reverse=True)
        
        return groups[:3]  # Return top 3 groups only for speed
    
    def _analyze_line_group(self, gray_frame, group):
        """
        Simplified analysis for line groups - fast horizon candidate generation.
        """
        if not group:
            return 0, 0, 0, 0
        
        # Calculate total length, average position, and dynamic angle
        total_length = 0
        center_x = 0
        center_y = 0
        
        # For angle calculation - collect all line vectors
        dx_total = 0
        dy_total = 0
        
        for line in group:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            total_length += length
            center_x += (x1 + x2) / 2
            center_y += (y1 + y2) / 2
            
            # Weight line direction by length
            dx = x2 - x1
            dy = y2 - y1
            dx_total += dx
            dy_total += dy
        
        # Average center position
        center_x /= len(group)
        center_y /= len(group)
        
        # Calculate actual line angle (not forced horizontal)
        if abs(dx_total) < 1e-6 and abs(dy_total) < 1e-6:
            # Degenerate case - fallback to horizontal
            theta = np.pi / 2
            rho = center_y
        else:
            # Calculate normal direction (perpendicular to average line direction)
            line_length = np.sqrt(dx_total*dx_total + dy_total*dy_total)
            nx = -dy_total / line_length  # Normal x component
            ny = dx_total / line_length   # Normal y component
            
            # Calculate rho and theta from normal vector
            rho = abs(center_x * nx + center_y * ny)
            theta = np.arctan2(ny, nx)
            
            # Ensure consistent representation: rho >= 0, theta in [0, pi]
            if rho < 0:
                rho = -rho
                theta = theta + np.pi
            
            # Normalize theta to [0, pi] range
            while theta < 0:
                theta += np.pi
            while theta >= np.pi:
                theta -= np.pi
        
        # Simplified confidence based primarily on length and position
        # Longer lines = higher confidence, lines in expected horizon zone = higher confidence
        h, w = gray_frame.shape
        
        # Length confidence
        length_confidence = min(total_length / (w * 0.3), 1.0)  # Normalize by 30% of frame width
        
        # Position confidence - for angled horizons, check if line reasonably divides frame
        # Calculate what fraction of frame is above/below the detected line
        cos_theta_calc = np.cos(theta) if theta != np.pi/2 else 0
        sin_theta_calc = np.sin(theta) if theta != np.pi/2 else 1
        
        # Sample points across frame to see division ratio
        sample_points = [(w*0.25, h*0.5), (w*0.5, h*0.5), (w*0.75, h*0.5)]
        above_count = 0
        below_count = 0
        
        for x, y in sample_points:
            distance = x * cos_theta_calc + y * sin_theta_calc - rho
            if distance < 0:
                above_count += 1
            else:
                below_count += 1
        
        # Good horizons should divide frame into reasonable proportions
        total_samples = len(sample_points)
        if total_samples > 0:
            above_ratio = above_count / total_samples
            # Prefer ratios between 0.2 and 0.8 (not too extreme)
            if 0.2 <= above_ratio <= 0.8:
                position_confidence = 1.0
            else:
                position_confidence = 0.5
        else:
            position_confidence = 0.5
        
        # Combined confidence (simplified)
        confidence = length_confidence * 0.7 + position_confidence * 0.3
        
        return rho, theta, total_length, min(confidence, 1.0)
    
    def _validate_horizon_candidate(self, gray_frame, rho, theta, total_length):
        """
        Validate that a line candidate represents a true horizon, not cloud edges.
        True horizons separate sky from ground with distinct characteristics.
        """
        h, w = gray_frame.shape
        
        # Quick checks first - more flexible for angled horizons
        if total_length < max(w * 0.1, h * 0.1):  # Horizon should have reasonable length in either dimension
            return False
            
        # For angled horizons, rho position check doesn't make sense - skip position-based filtering
            
        # Create regions above and below the candidate horizon
        cos_theta = np.cos(theta)  
        sin_theta = np.sin(theta)
        
        y, x = np.ogrid[:h, :w]
        distances = x * cos_theta + y * sin_theta - rho
        
        # Define sky and ground regions with sufficient margin
        margin = 15
        sky_mask = distances < -margin
        ground_mask = distances > margin
        
        sky_size = np.sum(sky_mask)
        ground_size = np.sum(ground_mask)
        
        # Need sufficient area in both regions
        if sky_size < h*w*0.1 or ground_size < h*w*0.1:
            return False
        
        # Sky should be more uniform than ground (key discriminator for cloud edges)
        sky_region = gray_frame[sky_mask]
        ground_region = gray_frame[ground_mask]
        
        sky_std = np.std(sky_region)
        ground_std = np.std(ground_region)
        
        # True horizon: sky more uniform than ground
        # Cloud edges: both regions have similar (high) variation
        uniformity_ratio = sky_std / max(ground_std, 1.0)
        
        # Sky should be significantly more uniform
        if uniformity_ratio > 0.8:  # Sky not significantly more uniform
            return False
        
        # Additional check: sky typically brighter than ground in outdoor scenarios
        sky_mean = np.mean(sky_region)
        ground_mean = np.mean(ground_region) 
        
        # Allow some flexibility, but generally sky should be brighter
        if sky_mean < ground_mean * 0.9:  # Sky significantly darker than ground
            return False
            
        return True
    
    def _find_horizon_cluster(self, horizon_candidates):
        """
        Find the largest coherent cluster of horizon candidates and return median line.
        """
        if not horizon_candidates:
            return None
        
        if len(horizon_candidates) == 1:
            # Single candidate - just return it
            rho, theta, confidence, length, group = horizon_candidates[0]
            return rho, theta, confidence, group
        
        # Cluster candidates by rho similarity (position similarity)
        clusters = []
        used = set()
        
        for i, (rho1, theta1, conf1, len1, group1) in enumerate(horizon_candidates):
            if i in used:
                continue
                
            # Start new cluster
            cluster = [(rho1, theta1, conf1, len1, group1)]
            used.add(i)
            
            # Find similar candidates
            for j, (rho2, theta2, conf2, len2, group2) in enumerate(horizon_candidates):
                if j in used:
                    continue
                
                # Check rho similarity (position similarity)
                rho_diff = abs(rho1 - rho2)
                theta_diff = min(abs(theta1 - theta2), np.pi - abs(theta1 - theta2))  # Circular difference
                
                # Cluster if candidates are similar in position and angle
                if rho_diff < 30 and theta_diff < np.pi/12:  # 30 pixels, 15 degrees
                    cluster.append((rho2, theta2, conf2, len2, group2))
                    used.add(j)
            
            if len(cluster) >= 1:  # Keep clusters with at least 1 member
                clusters.append(cluster)
        
        if not clusters:
            return None
        
        # Find the best cluster (most members, then highest total confidence)
        best_cluster = max(clusters, key=lambda c: (len(c), sum(item[2] for item in c)))
        
        # Calculate median rho and theta from the best cluster
        cluster_rhos = [item[0] for item in best_cluster]
        cluster_thetas = [item[1] for item in best_cluster]
        cluster_confidences = [item[2] for item in best_cluster]
        
        # Median rho
        median_rho = np.median(cluster_rhos)
        
        # Circular median for theta (more complex but important for angles)
        if len(cluster_thetas) == 1:
            median_theta = cluster_thetas[0]
        else:
            # Convert to complex numbers for circular averaging
            complex_angles = [np.exp(1j * theta) for theta in cluster_thetas]
            mean_complex = np.mean(complex_angles)
            median_theta = np.angle(mean_complex)
            
            # Ensure theta is in [0, pi] range
            while median_theta < 0:
                median_theta += np.pi
            while median_theta >= np.pi:
                median_theta -= np.pi
        
        # Combine confidence from cluster (average weighted by length)
        total_length = sum(item[3] for item in best_cluster)
        if total_length > 0:
            weighted_confidence = sum(item[2] * item[3] for item in best_cluster) / total_length
        else:
            weighted_confidence = np.mean(cluster_confidences)
        
        # Combine all line segments from cluster
        cluster_segments = []
        for _, _, _, _, group in best_cluster:
            cluster_segments.extend(group)
        
        return median_rho, median_theta, weighted_confidence, cluster_segments
    
    def _create_hough_debug_view(self, gray_frame, edges, debug_size):
        """
        Create a debug visualization showing detected Hough lines.
        """
        h, w = gray_frame.shape
        debug_w, debug_h = debug_size
        
        # Create debug canvas (black background)
        debug_canvas = np.zeros((debug_h, debug_w, 3), dtype=np.uint8)
        
        # Calculate edge density for adaptive parameters (same as main detection)
        edge_density = np.sum(edges) / (h * w * 255)
        min_dimension = min(w, h)
        
        # Use same shared parameters as main detection
        threshold, min_line_length, max_line_gap = self._get_hough_parameters(edge_density, min_dimension)
        
        # Detect lines using same parameters as main algorithm
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is not None:
            # Scale coordinates to debug view size
            scale_x = debug_w / w
            scale_y = debug_h / h
            
            # Filter lines same as main algorithm
            horizon_lines = self._filter_for_horizon_lines(lines, gray_frame.shape)
            
            # Draw all detected lines in gray
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                cv2.line(debug_canvas, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (100, 100, 100), 1)
            
            # Draw filtered horizon lines in bright colors
            for i, line in enumerate(horizon_lines):
                x1, y1, x2, y2 = line[0]
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                
                # Use different colors for different horizon candidates
                colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 128, 0)]
                color = colors[i % len(colors)]
                cv2.line(debug_canvas, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        
        # Add label
        cv2.putText(debug_canvas, "Hough Lines", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if lines is not None:
            cv2.putText(debug_canvas, f"Total: {len(lines)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            cv2.putText(debug_canvas, f"Filtered: {len(horizon_lines)}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return debug_canvas
    
    
    def apply_horizon_filter(self, frame: np.ndarray, detections: list) -> list:
        """
        Filter detections to keep only objects above horizon line.
        Uses adaptive horizon detection that works with arbitrary camera orientations.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Filtered detections above horizon (sky side)
        """
        # Detect horizon line every few frames for performance optimization
        horizon_result, confidence = None, 0.0
        should_detect = (
            self.frame_count - self.last_horizon_detection_frame >= self.horizon_detection_interval or
            self.horizon_line is None  # Always detect if no horizon yet
        ) or True
        
        if should_detect:
            horizon_result, confidence = self.detect_horizon_line(frame)
            self.last_horizon_detection_frame = self.frame_count
        
        # Enhanced temporal stability: only update horizon if confidence is high enough
        should_update_horizon = (
            horizon_result is not None and 
            confidence > self.min_confidence_for_update
        )
        if should_update_horizon:
            # Add to history for temporal smoothing
            self.horizon_history.append((horizon_result, confidence, self.frame_count))
            
            # Keep only recent history
            if len(self.horizon_history) > self.max_history:
                self.horizon_history.pop(0)
                
            # Simplified temporal smoothing for speed
            if len(self.horizon_history) >= 2:
                # Use simple weighted average of recent detections
                recent_rho = 0
                recent_theta = 0
                total_weight = 0
                
                # Only use last 2-3 detections for speed
                for (rho, theta), conf, frame_num in self.horizon_history[-3:]:
                    weight = conf  # Simple confidence weighting
                    recent_rho += rho * weight
                    recent_theta += theta * weight  # Simplified - assume small angle changes
                    total_weight += weight
                
                if total_weight > 0:
                    self.horizon_line = (recent_rho / total_weight, recent_theta / total_weight)
                    self.horizon_confidence = confidence
            else:
                self.horizon_line = horizon_result
                self.horizon_confidence = confidence
            
            # Reset horizon age since we updated
            self.horizon_age = 0
        else:
            # No update this frame - age the current horizon
            self.horizon_age += 1
            
            # Apply confidence decay if horizon is getting old
            if self.horizon_line is not None and self.horizon_age > 0:
                # Gradual confidence decay
                self.horizon_confidence *= self.confidence_decay_rate
                
                # If horizon is too old or confidence too low, clear it
                if (self.horizon_age > self.horizon_persistence_frames or 
                    self.horizon_confidence < 0.1):
                    self.horizon_line = None
                    self.horizon_confidence = 0.0
                    self.horizon_age = 0
        
        # Use detected horizon or fallback (debug the condition)
        horizon_available = hasattr(self, 'detected_horizon_segments') and self.detected_horizon_segments
        # Debug info - add to frame info later
        # horizon_status = f"H_avail={horizon_available}, H_conf={self.horizon_confidence:.2f}"
        
        if horizon_available:
            
            # Draw detected horizon line
            self._draw_horizon_line(frame, self.horizon_confidence)
            
    
            p, q = self.current_horizon_coord
            # Filter detections based on which side of horizon they're on
            filtered_detections = self._filter_detections_by_horizon(
                frame, detections, p, q
            )
            for d in detections:
                x, y, w, h, area = d
                if d in filtered_detections:
                    colour = (0, 255, 0)
                else:
                    colour = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(
                    frame, 
                    f"Area: {int(area)}", 
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    colour, 
                    1
                )
        else:
            filtered_detections = detections
            for d in detections:
                x, y, w, h, area = d

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"Area: {int(area)}", 
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1
                )
        return filtered_detections
    
    def _draw_horizon_line(self, frame: np.ndarray, confidence: float) -> None:
        """Draw the actual detected horizon segments on the frame."""
        
        # Draw the actual detected line segments instead of infinite line
        if hasattr(self, 'detected_horizon_segments') and self.detected_horizon_segments:
            # Draw each detected segment
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))  # Green for high confidence, red for low
            
            dx, dy = 0., 0.
            x, y = 0., 0.
            for line in self.detected_horizon_segments:
                x1, y1, x2, y2 = line[0]
                dx += x2 - x1
                dy += y2 - y1
                x += x1
                y += y1
            n = len(self.detected_horizon_segments)
            dy_dx = dy / dx
            x, y = x / n, y / n
            if np.abs(dy_dx) < 1e-5:
                dy_dx = 1e-5    
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            y1, y2 = y - dy_dx * x, y + dy_dx * (width - x)
            x1, x2 = x - y / dy_dx, x + (height - y) / dy_dx

            frame_edge_intersection = []

            if 0. <= y1 and y1 <= height:
                frame_edge_intersection.append((0, int(float(y1))))
            if 0. <= y2 and y2 <= height:
                frame_edge_intersection.append((width, int(float(y2))))
            if 0. <= x1 and x1 <= width:
                frame_edge_intersection.append((int(float(x1)), 0))
            if 0. <= x2 and x2 <= width:
                frame_edge_intersection.append((int(float(x2)), height))
            
            if frame_edge_intersection[0][0] < frame_edge_intersection[1][0]:
                # order points
                frame_edge_intersection = frame_edge_intersection[::-1]

            p, q = frame_edge_intersection[0], frame_edge_intersection[1]
            if len(self.current_horizon_coord) == 0.:
                self.current_horizon_coord = frame_edge_intersection
            else:
                r = self.horizon_decay_rate
                self.current_horizon_coord[0] = (int(r * self.current_horizon_coord[0][0] + (1 - r) * p[0]), int(r * self.current_horizon_coord[0][1] + (1 - r) * p[1]))
                self.current_horizon_coord[1] = (int(r * self.current_horizon_coord[1][0] + (1 - r) * q[0]), int(r * self.current_horizon_coord[1][1] + (1 - r) * q[1]))
            assert len(frame_edge_intersection) == 2
            cv2.line(frame, self.current_horizon_coord[0], self.current_horizon_coord[1], color, 3)  # Thicker line for visibility
            
            # Add angle and confidence label
            theta = np.acos((p[1] - q[1]) / (p[0] - q[0]))
            angle_deg = np.degrees(theta)
            if angle_deg > 90:
                angle_deg = 180 - angle_deg  # Normalize to show roll angle
            
            # Add debug information about why we have these segments
            label = f"Horizon: conf={confidence:.2f}, angle={angle_deg:.1f}°, segments={len(self.detected_horizon_segments)}"
            cv2.putText(
                frame, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    
    def _filter_detections_by_horizon(self, frame: np.ndarray, detections: list, 
                                    p, q) -> list:
        """
        Filter detections based on which side of the horizon line they fall on.
        Determines which side is 'sky' based on region analysis.
        """
        if not detections:
            return []
        
        h, w = frame.shape[:2]
        
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        # Determine which side of the line represents 'sky' vs 'ground'
        # Sample a few points on each side and analyze texture
        if self.sky_side_sign == 0 or np.abs(dy / dx) > 10:
            # check that the sky side is not changing  
            self.sky_side_sign = self._determine_sky_side(frame, p, q)
        
        filtered_detections = []
        for x, y, w_box, h_box, area in detections:
            # Calculate detection center
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            
            # Calculate signed distance from center to horizon line
            # distance = center_x * cos_theta + center_y * sin_theta - rho
            yp = p[1] + dy / dx * (center_x - p[0])    
            # Keep detection if it's on the sky side
            if ((yp - center_y + self.horizon_margin) * self.sky_side_sign) > 0:  # Sky side
                
                filtered_detections.append((x, y, w_box, h_box, area))
                
        return filtered_detections
    
    def _determine_sky_side(self, frame: np.ndarray, p, q) -> int:
        """
        Determine which side of the horizon line represents sky vs ground.
        Uses multiple features: texture, color distribution, and edge gradients.
        Returns +1 or -1 indicating the sign of the distance for the sky side.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        n = np.sqrt(dx**2 + dy**2)
        cos_theta = dx / n
        sin_theta = -dy / n
        d = (dy / dx - dx / dy)

        if np.abs(d) < 1e-5:
            d = 1e-5
        x = (p[0] * dy / dx - p[1]) / d
        y = x * dx / dy

        rho = np.sqrt(x**2 + y**2)
        
        # Create masks for regions on each side of the horizon line
        y, x = np.ogrid[:h, :w]
        distances = x * cos_theta + y * sin_theta - rho
        margin = 15  # Larger margin to avoid horizon line itself
        
        positive_mask = distances > margin
        negative_mask = distances < -margin
        
        # Simplified and faster sky/ground determination
        pos_size = np.sum(positive_mask)
        neg_size = np.sum(negative_mask)
        
        if pos_size < 100 or neg_size < 100:  # Too small for reliable analysis
            return -1  # Default assumption
        
        # Primary discriminator: Edge density (sky has fewer edges)
        edges = cv2.Canny(gray, 25, 75)  # Single edge detection call
        pos_edge_density = np.sum(edges[positive_mask]) / pos_size
        neg_edge_density = np.sum(edges[negative_mask]) / neg_size
        
        # Secondary discriminator: Mean intensity (often sky is brighter)
        pos_intensity = np.mean(gray[positive_mask])
        neg_intensity = np.mean(gray[negative_mask])
        
        # Simple decision logic
        edge_diff = neg_edge_density - pos_edge_density  # Positive if negative side has more edges
        brightness_diff = pos_intensity - neg_intensity  # Positive if positive side is brighter
        
        # Combine with simple weights
        combined_score = edge_diff * 0.8 + brightness_diff * 0.2 / 255.0
        # Determine sky side based on combined score
        if combined_score < 0:
            # Negative side appears more "ground-like", so positive side is sky
            return 1  
        else:
            # Positive side appears more "ground-like", so negative side is sky
            return -1
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the detection pipeline."""
        self.frame_count += 1
        
        # Detect moving objects
        processed_frame, detections, debug_mask = self.detect_motion_objects(frame)
        
        # Apply horizon filtering
        filtered_detections = self.apply_horizon_filter(processed_frame, detections)
        
        # Add frame info with horizon debug status
        frame_info = f"Frame: {self.frame_count} | Detections: {len(filtered_detections)}"
        cv2.putText(
            processed_frame, 
            frame_info, 
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Add horizon debug info
        if hasattr(self, 'horizon_status'):
            cv2.putText(
                processed_frame,
                self.horizon_status,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
        
        # Create composite view for debugging
        h, w = frame.shape[:2]
        
        # Existing debug mask in bottom right corner
        debug_resized = cv2.resize(debug_mask, (w//4, h//4))
        processed_frame[h-debug_resized.shape[0]-10:h-10, 
                      w-debug_resized.shape[1]-10:w-10] = debug_resized
        
        # Add Canny edges debug view in bottom left corner
        gray_debug = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_debug = cv2.GaussianBlur(gray_debug, (7, 7), 0)
        edges_debug = cv2.Canny(blurred_debug, 30, 90)  # Match main detection thresholds
        
        # Convert edges to 3-channel for display
        edges_debug_color = cv2.cvtColor(edges_debug, cv2.COLOR_GRAY2BGR)
        edges_resized = cv2.resize(edges_debug_color, (w//4, h//4))
        
        # Place edges debug view in bottom left corner
        processed_frame[h-edges_resized.shape[0]-10:h-10, 
                      10:10+edges_resized.shape[1]] = edges_resized
        
        # Add HoughLinesP debug view in bottom center
        # Create Hough lines visualization
        hough_debug = self._create_hough_debug_view(gray_debug, edges_debug, (w//4, h//4))
        
        # Place Hough lines debug view in bottom center
        center_x = w//2 - hough_debug.shape[1]//2
        processed_frame[h-hough_debug.shape[0]-10:h-10,
                      center_x:center_x+hough_debug.shape[1]] = hough_debug
        
        return processed_frame
    
    def run(self) -> None:
        """Run the video processing pipeline."""
        if not self.initialize_capture():
            return
            
        print("Starting video processing... Press 'q' to quit")
        
        # Create resizable window and set size
        cv2.namedWindow('Drone Detection Pipeline', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Detection Pipeline', 640, 480)
        
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
