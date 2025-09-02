"""
Horizon detection module using edge detection and Hough transforms.
Supports arbitrary camera orientations and provides temporal stability.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from config import CONFIG


class HorizonDetector:
    """
    Detects horizon line using Canny edge detection and Hough line transforms.
    Includes temporal smoothing and validation for robust horizon tracking.
    """
    
    def __init__(self):
        """Initialize horizon detector with configured parameters."""
        self.config = CONFIG.horizon
        
        # Current horizon state
        self.horizon_line: Optional[Tuple[float, float]] = None  # (rho, theta)
        self.horizon_confidence: float = 0.0
        self.sky_side_sign: int = 0  # +1 or -1 indicating which side is sky
        
        # Temporal tracking
        self.horizon_history: List[Tuple[Tuple[float, float], float, int]] = []
        self.horizon_age: int = 0
        self.last_detection_frame: int = 0
        
        # Current visualization data
        self.detected_segments: List = []
        self.current_horizon_coords: List = []
    
    def detect_horizon(self, frame: np.ndarray, frame_number: int) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Detect horizon line in frame with temporal stability.
        
        Args:
            frame: Input BGR frame
            frame_number: Current frame number for temporal tracking
            
        Returns:
            Tuple of ((rho, theta), confidence) or (None, 0.0) if no horizon found
        """
        # Check if we should run detection this frame (performance optimization)
        should_detect = (
            frame_number - self.last_detection_frame >= self.config.horizon_detection_interval or
            self.horizon_line is None
        )
        
        if not should_detect:
            # Age existing horizon and apply decay
            self._age_horizon()
            return self.horizon_line, self.horizon_confidence
        
        # Run horizon detection
        horizon_result, confidence = self._detect_horizon_line(frame)
        self.last_detection_frame = frame_number
        
        # Update horizon with temporal stability
        self._update_horizon_with_stability(horizon_result, confidence, frame_number)
        
        return self.horizon_line, self.horizon_confidence
    
    def _detect_horizon_line(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Core horizon detection algorithm using edge detection and Hough transforms.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of ((rho, theta), confidence) or (None, 0.0)
        """
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.config.blur_kernel_size, self.config.blur_kernel_size), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.config.canny_low_threshold, self.config.canny_high_threshold)
        
        # Calculate edge density for adaptive parameters
        edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1] * 255)
        
        # Get adaptive Hough parameters
        threshold, min_line_length, max_line_gap = self._get_adaptive_hough_params(edge_density, frame.shape)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is None:
            return None, 0.0
        
        # Filter for horizon-like lines
        horizon_lines = self._filter_horizon_candidates(lines, frame.shape)
        if not horizon_lines:
            return None, 0.0
        
        # Group aligned lines
        line_groups = self._group_aligned_lines(horizon_lines)
        if not line_groups:
            return None, 0.0
        
        # Analyze groups and find best horizon candidate
        horizon_candidates = []
        for group in line_groups:
            if group:
                rho, theta, total_length, confidence = self._analyze_line_group(gray, group)
                if confidence > 0.05:
                    horizon_candidates.append((rho, theta, confidence, total_length, group))
        
        if not horizon_candidates:
            return None, 0.0
        
        # Find best cluster of candidates
        best_cluster = self._find_best_horizon_cluster(horizon_candidates)
        if best_cluster is None:
            return None, 0.0
        
        median_rho, median_theta, cluster_confidence, cluster_segments = best_cluster
        self.detected_segments = cluster_segments
        
        return (median_rho, median_theta), cluster_confidence
    
    def _get_adaptive_hough_params(self, edge_density: float, frame_shape: Tuple[int, int]) -> Tuple[int, int, int]:
        """Get Hough transform parameters based on edge density."""
        min_dimension = min(frame_shape[1], frame_shape[0])
        
        if edge_density < 0.01:  # Sparse edges
            threshold = self.config.hough_threshold_sparse
            min_line_length = max(20, int(min_dimension * self.config.hough_min_line_sparse))
            max_line_gap = self.config.hough_max_gap_sparse
        elif edge_density < 0.03:  # Moderate edges
            threshold = self.config.hough_threshold_moderate
            min_line_length = max(30, int(min_dimension * self.config.hough_min_line_moderate))
            max_line_gap = self.config.hough_max_gap_moderate
        else:  # Dense edges
            threshold = self.config.hough_threshold_normal
            min_line_length = max(40, int(min_dimension * self.config.hough_min_line_normal))
            max_line_gap = self.config.hough_max_gap_normal
        
        return threshold, min_line_length, max_line_gap
    
    def _filter_horizon_candidates(self, lines: np.ndarray, frame_shape: Tuple[int, int]) -> List:
        """Filter lines to keep only potential horizon candidates."""
        h, w = frame_shape[:2]
        horizon_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) < 1:  # Nearly vertical
                continue
            
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 180
            
            # Filter out vertical lines (unlikely to be horizons)
            if self.config.angle_filter_vertical_min <= angle_deg <= self.config.angle_filter_vertical_max:
                continue
            
            # Filter by position - skip lines entirely in top portion
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            if max_y < h * self.config.position_filter_top_threshold:
                continue
            
            # Filter by length
            length = np.sqrt(dx*dx + dy*dy)
            min_length = max(self.config.min_line_length_pixels, w * 0.05)
            if length < min_length:
                continue
            
            horizon_lines.append(line)
        
        return horizon_lines
    
    def _group_aligned_lines(self, lines: List) -> List[List]:
        """Group nearby aligned lines that might represent the same horizon."""
        if not lines:
            return []
        
        groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            x1, y1, x2, y2 = line1[0]
            group = [line1]
            group_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            used.add(i)
            
            center_y1 = (y1 + y2) / 2
            
            # Find nearby lines
            for j, line2 in enumerate(lines):
                if j in used:
                    continue
                
                x3, y3, x4, y4 = line2[0]
                center_y2 = (y3 + y4) / 2
                
                # Check vertical alignment
                if abs(center_y1 - center_y2) < 20:
                    group.append(line2)
                    group_length += np.sqrt((x4-x3)**2 + (y4-y3)**2)
                    used.add(j)
            
            if group_length > 40:
                groups.append(group)
        
        # Sort by total length
        groups.sort(key=lambda g: sum(np.sqrt((line[0][2]-line[0][0])**2 + (line[0][3]-line[0][1])**2) for line in g), reverse=True)
        return groups[:3]  # Return top 3 groups
    
    def _analyze_line_group(self, gray_frame: np.ndarray, group: List) -> Tuple[float, float, float, float]:
        """Analyze a group of aligned lines to extract horizon parameters."""
        if not group:
            return 0, 0, 0, 0
        
        total_length = 0
        center_x = 0
        center_y = 0
        dx_total = 0
        dy_total = 0
        
        for line in group:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            total_length += length
            center_x += (x1 + x2) / 2
            center_y += (y1 + y2) / 2
            
            dx_total += x2 - x1
            dy_total += y2 - y1
        
        center_x /= len(group)
        center_y /= len(group)
        
        # Calculate line parameters
        if abs(dx_total) < 1e-6 and abs(dy_total) < 1e-6:
            theta = np.pi / 2
            rho = center_y
        else:
            line_length = np.sqrt(dx_total*dx_total + dy_total*dy_total)
            nx = -dy_total / line_length
            ny = dx_total / line_length
            
            rho = abs(center_x * nx + center_y * ny)
            theta = np.arctan2(ny, nx)
            
            if rho < 0:
                rho = -rho
                theta = theta + np.pi
            
            while theta < 0:
                theta += np.pi
            while theta >= np.pi:
                theta -= np.pi
        
        # Calculate confidence based on length and position
        h, w = gray_frame.shape
        length_confidence = min(total_length / (w * 0.3), 1.0)
        
        # Position confidence based on frame division
        cos_theta_calc = np.cos(theta) if theta != np.pi/2 else 0
        sin_theta_calc = np.sin(theta) if theta != np.pi/2 else 1
        
        sample_points = [(w*0.25, h*0.5), (w*0.5, h*0.5), (w*0.75, h*0.5)]
        above_count = sum(1 for x, y in sample_points if x * cos_theta_calc + y * sin_theta_calc - rho < 0)
        above_ratio = above_count / len(sample_points)
        
        position_confidence = 1.0 if 0.2 <= above_ratio <= 0.8 else 0.5
        
        confidence = length_confidence * 0.7 + position_confidence * 0.3
        return rho, theta, total_length, min(confidence, 1.0)
    
    def _find_best_horizon_cluster(self, candidates: List) -> Optional[Tuple[float, float, float, List]]:
        """Find the best cluster of horizon candidates using position similarity."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            rho, theta, confidence, length, group = candidates[0]
            return rho, theta, confidence, group
        
        # Simple clustering by rho similarity
        clusters = []
        used = set()
        
        for i, (rho1, theta1, conf1, len1, group1) in enumerate(candidates):
            if i in used:
                continue
            
            cluster = [(rho1, theta1, conf1, len1, group1)]
            used.add(i)
            
            for j, (rho2, theta2, conf2, len2, group2) in enumerate(candidates):
                if j in used:
                    continue
                
                rho_diff = abs(rho1 - rho2)
                theta_diff = min(abs(theta1 - theta2), np.pi - abs(theta1 - theta2))
                
                if rho_diff < 30 and theta_diff < np.pi/12:
                    cluster.append((rho2, theta2, conf2, len2, group2))
                    used.add(j)
            
            if cluster:
                clusters.append(cluster)
        
        if not clusters:
            return None
        
        # Find best cluster
        best_cluster = max(clusters, key=lambda c: (len(c), sum(item[2] for item in c)))
        
        # Calculate median parameters
        cluster_rhos = [item[0] for item in best_cluster]
        cluster_thetas = [item[1] for item in best_cluster]
        cluster_confidences = [item[2] for item in best_cluster]
        
        median_rho = np.median(cluster_rhos)
        
        if len(cluster_thetas) == 1:
            median_theta = cluster_thetas[0]
        else:
            # Circular averaging for angles
            complex_angles = [np.exp(1j * theta) for theta in cluster_thetas]
            mean_complex = np.mean(complex_angles)
            median_theta = np.angle(mean_complex)
            while median_theta < 0:
                median_theta += np.pi
            while median_theta >= np.pi:
                median_theta -= np.pi
        
        # Weighted confidence
        total_length = sum(item[3] for item in best_cluster)
        if total_length > 0:
            weighted_confidence = sum(item[2] * item[3] for item in best_cluster) / total_length
        else:
            weighted_confidence = np.mean(cluster_confidences)
        
        # Combine all segments
        cluster_segments = []
        for _, _, _, _, group in best_cluster:
            cluster_segments.extend(group)
        
        return median_rho, median_theta, weighted_confidence, cluster_segments
    
    def _update_horizon_with_stability(self, horizon_result: Optional[Tuple[float, float]], 
                                     confidence: float, frame_number: int) -> None:
        """Update horizon with temporal stability and smoothing."""
        should_update = (
            horizon_result is not None and 
            confidence > self.config.min_confidence_for_update
        )
        
        if should_update:
            # Add to history
            self.horizon_history.append((horizon_result, confidence, frame_number))
            
            # Maintain history length
            if len(self.horizon_history) > self.config.horizon_history_length:
                self.horizon_history.pop(0)
            
            # Temporal smoothing
            if len(self.horizon_history) >= 2:
                recent_rho = 0
                recent_theta = 0
                total_weight = 0
                
                for (rho, theta), conf, _ in self.horizon_history[-3:]:
                    weight = conf
                    recent_rho += rho * weight
                    recent_theta += theta * weight
                    total_weight += weight
                
                if total_weight > 0:
                    self.horizon_line = (recent_rho / total_weight, recent_theta / total_weight)
                    self.horizon_confidence = confidence
            else:
                self.horizon_line = horizon_result
                self.horizon_confidence = confidence
            
            self.horizon_age = 0
        else:
            self._age_horizon()
    
    def _age_horizon(self) -> None:
        """Age the current horizon and apply confidence decay."""
        self.horizon_age += 1
        
        if self.horizon_line is not None and self.horizon_age > 0:
            self.horizon_confidence *= self.config.confidence_decay_rate
            
            if (self.horizon_age > self.config.horizon_persistence_frames or 
                self.horizon_confidence < 0.1):
                self.horizon_line = None
                self.horizon_confidence = 0.0
                self.horizon_age = 0
    
    def get_horizon_line_coords(self, frame_shape: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Get horizon line coordinates for drawing on frame.
        
        Args:
            frame_shape: (height, width) of the frame
            
        Returns:
            List of [(x1, y1), (x2, y2)] coordinates or None
        """
        if not self.detected_segments:
            return None
        
        h, w = frame_shape[:2]
        
        # Calculate average line direction from detected segments
        dx_total, dy_total = 0.0, 0.0
        x_center, y_center = 0.0, 0.0
        
        for line in self.detected_segments:
            x1, y1, x2, y2 = line[0]
            dx_total += x2 - x1
            dy_total += y2 - y1
            x_center += x1
            y_center += y1
        
        n_segments = len(self.detected_segments)
        dx_total /= n_segments
        dy_total /= n_segments
        x_center /= n_segments
        y_center /= n_segments
        
        # Calculate slope and extend to frame edges
        if abs(dx_total) < 1e-5:
            dx_total = 1e-5
        
        slope = dy_total / dx_total
        
        # Find intersections with frame edges
        y1 = y_center - slope * x_center  # y-intercept at x=0
        y2 = y_center + slope * (w - x_center)  # y at x=w
        x1 = x_center - y_center / slope  # x-intercept at y=0
        x2 = x_center + (h - y_center) / slope  # x at y=h
        
        intersections = []
        if 0 <= y1 <= h:
            intersections.append((0, int(y1)))
        if 0 <= y2 <= h:
            intersections.append((w, int(y2)))
        if 0 <= x1 <= w:
            intersections.append((int(x1), 0))
        if 0 <= x2 <= w:
            intersections.append((int(x2), h))
        
        if len(intersections) >= 2:
            # Apply temporal smoothing to coordinates
            p, q = intersections[0], intersections[1]
            if not self.current_horizon_coords:
                self.current_horizon_coords = [p, q]
            else:
                r = self.config.horizon_decay_rate
                p_smooth = (int(r * self.current_horizon_coords[0][0] + (1-r) * p[0]),
                           int(r * self.current_horizon_coords[0][1] + (1-r) * p[1]))
                q_smooth = (int(r * self.current_horizon_coords[1][0] + (1-r) * q[0]),
                           int(r * self.current_horizon_coords[1][1] + (1-r) * q[1]))
                self.current_horizon_coords = [p_smooth, q_smooth]
            
            return self.current_horizon_coords
        
        return None
    
    def determine_sky_side(self, frame: np.ndarray, horizon_coords: List[Tuple[int, int]]) -> int:
        """
        Determine which side of horizon is sky vs ground.
        
        Args:
            frame: Input BGR frame
            horizon_coords: [(x1, y1), (x2, y2)] horizon line coordinates
            
        Returns:
            +1 or -1 indicating sky side sign
        """
        if self.sky_side_sign != 0:
            return self.sky_side_sign
        
        if len(horizon_coords) < 2:
            return -1  # Default
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        (x1, y1), (x2, y2) = horizon_coords
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 1e-5:
            return -1
        
        # Create masks for regions on each side
        y, x = np.ogrid[:h, :w]
        
        # Line equation: (y - y1) = (dy/dx) * (x - x1)
        # Rearranged: dy*x - dx*y + dx*y1 - dy*x1 = 0
        line_values = dy * x - dx * y + dx * y1 - dy * x1
        
        margin = self.config.region_margin
        positive_mask = line_values > margin
        negative_mask = line_values < -margin
        
        pos_size = np.sum(positive_mask)
        neg_size = np.sum(negative_mask)
        
        if pos_size < 100 or neg_size < 100:
            self.sky_side_sign = -1
            return self.sky_side_sign
        
        # Analyze edge density (sky typically has fewer edges)
        edges = cv2.Canny(gray, 25, 75)
        pos_edge_density = np.sum(edges[positive_mask]) / pos_size
        neg_edge_density = np.sum(edges[negative_mask]) / neg_size
        
        # Analyze brightness (sky often brighter)
        pos_intensity = np.mean(gray[positive_mask])
        neg_intensity = np.mean(gray[negative_mask])
        
        # Decision logic
        edge_diff = neg_edge_density - pos_edge_density
        brightness_diff = (pos_intensity - neg_intensity) / 255.0
        
        combined_score = edge_diff * 0.8 + brightness_diff * 0.2
        
        self.sky_side_sign = 1 if combined_score < 0 else -1
        return self.sky_side_sign
    
    def reset(self) -> None:
        """Reset horizon detector state."""
        self.horizon_line = None
        self.horizon_confidence = 0.0
        self.sky_side_sign = 0
        self.horizon_history.clear()
        self.horizon_age = 0
        self.detected_segments.clear()
        self.current_horizon_coords.clear()