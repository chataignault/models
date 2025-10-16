"""
Tracking system module for the three operational phases of drone detection.
Handles Pre-Lock-On, Ground Tracking, and Flight Tracking phases.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from config import CONFIG


class OperationalPhase(Enum):
    """Three operational phases of the drone detection system."""
    PRE_LOCK_ON = "pre_lock_on"      # Scanning for targets, no specific target
    GROUND_TRACKING = "ground_tracking"  # Target acquired, tracking on ground
    FLIGHT_TRACKING = "flight_tracking"  # Target in flight, active tracking


@dataclass
class Target:
    """Represents a detected target with tracking information."""
    id: int
    center: Tuple[int, int]  # (x, y) center position
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    area: float
    confidence: float
    last_seen_frame: int
    track_history: List[Tuple[int, int]]  # History of center positions
    velocity: Tuple[float, float] = (0.0, 0.0)  # Estimated velocity (vx, vy)
    
    def update_position(self, center: Tuple[int, int], bbox: Tuple[int, int, int, int], 
                       area: float, frame_number: int) -> None:
        """Update target position and tracking information."""
        old_center = self.center
        self.center = center
        self.bbox = bbox
        self.area = area
        self.last_seen_frame = frame_number
        
        # Update track history (keep last 10 positions)
        self.track_history.append(center)
        if len(self.track_history) > 10:
            self.track_history.pop(0)
        
        # Estimate velocity
        if len(self.track_history) >= 2:
            prev_center = self.track_history[-2]
            self.velocity = (center[0] - prev_center[0], center[1] - prev_center[1])
    
    def predict_next_position(self) -> Tuple[int, int]:
        """Predict next position based on current velocity."""
        return (int(self.center[0] + self.velocity[0]), 
                int(self.center[1] + self.velocity[1]))


class TargetTracker:
    """
    Multi-target tracker that manages detection and tracking of multiple objects.
    Handles target association, track management, and phase transitions.
    """
    
    def __init__(self):
        """Initialize the target tracker."""
        self.targets: Dict[int, Target] = {}
        self.next_target_id = 1
        self.frame_number = 0
        
        # Tracking parameters
        self.max_track_distance = 50  # Maximum distance for track association
        self.max_frames_without_detection = 10  # Drop tracks after this many frames
        self.min_track_confidence = 0.3
        
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame_number: int) -> List[Target]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y, w, h, area) detection tuples
            frame_number: Current frame number
            
        Returns:
            List of active Target objects
        """
        self.frame_number = frame_number
        
        # Convert detections to center points
        detection_centers = []
        for x, y, w, h, area in detections:
            center = (x + w // 2, y + h // 2)
            detection_centers.append((center, (x, y, w, h), area))
        
        # Associate detections with existing tracks
        self._associate_detections(detection_centers)
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        return list(self.targets.values())
    
    def _associate_detections(self, detection_centers: List[Tuple[Tuple[int, int], Tuple[int, int, int, int], float]]) -> None:
        """Associate detections with existing tracks or create new tracks."""
        unassigned_detections = list(range(len(detection_centers)))
        unassigned_tracks = list(self.targets.keys())
        
        # Calculate distance matrix between tracks and detections
        distances = {}
        for track_id in unassigned_tracks:
            target = self.targets[track_id]
            predicted_pos = target.predict_next_position()
            
            for det_idx in unassigned_detections:
                center, _, _ = detection_centers[det_idx]
                distance = np.sqrt((predicted_pos[0] - center[0])**2 + (predicted_pos[1] - center[1])**2)
                distances[(track_id, det_idx)] = distance
        
        # Assign detections to tracks (greedy assignment based on minimum distance)
        assignments = []
        while unassigned_tracks and unassigned_detections and distances:
            # Find minimum distance assignment
            min_key = min(distances.keys(), key=lambda k: distances[k])
            min_distance = distances[min_key]
            
            if min_distance <= self.max_track_distance:
                track_id, det_idx = min_key
                assignments.append((track_id, det_idx))
                
                # Remove assigned track and detection
                unassigned_tracks.remove(track_id)
                unassigned_detections.remove(det_idx)
                
                # Remove all distances involving these
                distances = {k: v for k, v in distances.items() 
                           if k[0] != track_id and k[1] != det_idx}
            else:
                break
        
        # Update assigned tracks
        for track_id, det_idx in assignments:
            center, bbox, area = detection_centers[det_idx]
            self.targets[track_id].update_position(center, bbox, area, self.frame_number)
        
        # Create new tracks for unassigned detections
        for det_idx in unassigned_detections:
            center, bbox, area = detection_centers[det_idx]
            new_target = Target(
                id=self.next_target_id,
                center=center,
                bbox=bbox,
                area=area,
                confidence=1.0,
                last_seen_frame=self.frame_number,
                track_history=[center]
            )
            self.targets[self.next_target_id] = new_target
            self.next_target_id += 1
    
    def _remove_stale_tracks(self) -> None:
        """Remove tracks that haven't been updated recently."""
        stale_tracks = []
        for track_id, target in self.targets.items():
            frames_since_update = self.frame_number - target.last_seen_frame
            if frames_since_update > self.max_frames_without_detection:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self.targets[track_id]
    
    def get_primary_target(self) -> Optional[Target]:
        """
        Get the primary target (most centered in frame).
        Used for transitioning to Ground Tracking phase.
        """
        if not self.targets:
            return None
        
        # For simplicity, return the target closest to frame center
        # In a real system, this would be more sophisticated
        frame_center = (160, 120)  # Assuming 320x240 resolution
        
        closest_target = None
        min_distance = float('inf')
        
        for target in self.targets.values():
            distance = np.sqrt((target.center[0] - frame_center[0])**2 + 
                             (target.center[1] - frame_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_target = target
        
        return closest_target
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.targets.clear()
        self.next_target_id = 1


class PhaseManager:
    """
    Manages transitions between the three operational phases.
    Coordinates different tracking behaviors for each phase.
    """
    
    def __init__(self):
        """Initialize phase manager."""
        self.current_phase = OperationalPhase.PRE_LOCK_ON
        self.target_tracker = TargetTracker()
        self.primary_target: Optional[Target] = None
        self.phase_start_frame = 0
        
        # Phase transition parameters
        self.lock_on_stability_frames = 30  # Frames to confirm target before Ground Tracking
        self.flight_detection_threshold = 5.0  # Velocity threshold for Flight Tracking
        self.target_lost_frames = 60  # Frames before returning to Pre-Lock-On
        
        # State tracking
        self.target_stable_count = 0
        self.target_lost_count = 0
    
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame_number: int) -> Dict[str, Any]:
        """
        Update phase manager with new detections.
        
        Args:
            detections: List of detection tuples
            frame_number: Current frame number
            
        Returns:
            Dictionary with phase information and tracking results
        """
        # Update tracker with detections
        active_targets = self.target_tracker.update(detections, frame_number)
        
        # Phase-specific logic
        phase_info = self._update_phase_logic(active_targets, frame_number)
        
        return {
            'phase': self.current_phase,
            'active_targets': active_targets,
            'primary_target': self.primary_target,
            'phase_info': phase_info
        }
    
    def _update_phase_logic(self, active_targets: List[Target], frame_number: int) -> Dict[str, Any]:
        """Update phase-specific logic and handle transitions."""
        phase_info = {}
        
        if self.current_phase == OperationalPhase.PRE_LOCK_ON:
            phase_info = self._handle_pre_lock_on_phase(active_targets, frame_number)
            
        elif self.current_phase == OperationalPhase.GROUND_TRACKING:
            phase_info = self._handle_ground_tracking_phase(active_targets, frame_number)
            
        elif self.current_phase == OperationalPhase.FLIGHT_TRACKING:
            phase_info = self._handle_flight_tracking_phase(active_targets, frame_number)
        
        return phase_info
    
    def _handle_pre_lock_on_phase(self, active_targets: List[Target], frame_number: int) -> Dict[str, Any]:
        """Handle Pre-Lock-On phase logic."""
        phase_info = {'status': 'scanning', 'targets_detected': len(active_targets)}
        
        # Look for potential primary target
        potential_target = self.target_tracker.get_primary_target()
        
        if potential_target:
            if self.primary_target and self.primary_target.id == potential_target.id:
                # Same target - increment stability counter
                self.target_stable_count += 1
                phase_info['stability_count'] = self.target_stable_count
                
                # Check if target is stable enough for Ground Tracking
                if self.target_stable_count >= self.lock_on_stability_frames:
                    self._transition_to_ground_tracking(potential_target, frame_number)
                    phase_info['status'] = 'target_acquired'
            else:
                # New potential target
                self.primary_target = potential_target
                self.target_stable_count = 1
                phase_info['status'] = 'target_candidate'
        else:
            # No targets
            self.primary_target = None
            self.target_stable_count = 0
            self.target_lost_count = 0
        
        return phase_info
    
    def _handle_ground_tracking_phase(self, active_targets: List[Target], frame_number: int) -> Dict[str, Any]:
        """Handle Ground Tracking phase logic."""
        phase_info = {'status': 'tracking_ground'}
        
        if not self.primary_target:
            self._transition_to_pre_lock_on(frame_number)
            phase_info['status'] = 'target_lost'
            return phase_info
        
        # Check if primary target is still active
        target_still_active = any(t.id == self.primary_target.id for t in active_targets)
        
        if target_still_active:
            # Update primary target reference
            self.primary_target = next(t for t in active_targets if t.id == self.primary_target.id)
            self.target_lost_count = 0
            
            # Check for flight transition (based on velocity)
            velocity_magnitude = np.sqrt(self.primary_target.velocity[0]**2 + 
                                       self.primary_target.velocity[1]**2)
            
            phase_info['velocity'] = velocity_magnitude
            
            if velocity_magnitude > self.flight_detection_threshold:
                self._transition_to_flight_tracking(frame_number)
                phase_info['status'] = 'transitioning_to_flight'
                
        else:
            # Target lost
            self.target_lost_count += 1
            phase_info['lost_count'] = self.target_lost_count
            
            if self.target_lost_count >= self.target_lost_frames:
                self._transition_to_pre_lock_on(frame_number)
                phase_info['status'] = 'target_lost'
        
        return phase_info
    
    def _handle_flight_tracking_phase(self, active_targets: List[Target], frame_number: int) -> Dict[str, Any]:
        """Handle Flight Tracking phase logic."""
        phase_info = {'status': 'tracking_flight'}
        
        if not self.primary_target:
            self._transition_to_pre_lock_on(frame_number)
            phase_info['status'] = 'target_lost'
            return phase_info
        
        # Check if primary target is still active
        target_still_active = any(t.id == self.primary_target.id for t in active_targets)
        
        if target_still_active:
            self.primary_target = next(t for t in active_targets if t.id == self.primary_target.id)
            self.target_lost_count = 0
            
            velocity_magnitude = np.sqrt(self.primary_target.velocity[0]**2 + 
                                       self.primary_target.velocity[1]**2)
            phase_info['velocity'] = velocity_magnitude
            
            # Could add logic for returning to Ground Tracking if target lands
            # For now, stay in Flight Tracking until target is lost
            
        else:
            # Target lost in flight
            self.target_lost_count += 1
            phase_info['lost_count'] = self.target_lost_count
            
            if self.target_lost_count >= self.target_lost_frames:
                self._transition_to_pre_lock_on(frame_number)
                phase_info['status'] = 'target_lost'
        
        return phase_info
    
    def _transition_to_pre_lock_on(self, frame_number: int) -> None:
        """Transition to Pre-Lock-On phase."""
        self.current_phase = OperationalPhase.PRE_LOCK_ON
        self.phase_start_frame = frame_number
        self.primary_target = None
        self.target_stable_count = 0
        self.target_lost_count = 0
    
    def _transition_to_ground_tracking(self, target: Target, frame_number: int) -> None:
        """Transition to Ground Tracking phase."""
        self.current_phase = OperationalPhase.GROUND_TRACKING
        self.phase_start_frame = frame_number
        self.primary_target = target
        self.target_stable_count = 0
        self.target_lost_count = 0
    
    def _transition_to_flight_tracking(self, frame_number: int) -> None:
        """Transition to Flight Tracking phase."""
        self.current_phase = OperationalPhase.FLIGHT_TRACKING
        self.phase_start_frame = frame_number
        self.target_lost_count = 0
    
    def get_phase_display_info(self) -> Dict[str, Any]:
        """Get information for display purposes."""
        return {
            'phase_name': self.current_phase.value.replace('_', ' ').title(),
            'frame_duration': self.target_tracker.frame_number - self.phase_start_frame,
            'primary_target_id': self.primary_target.id if self.primary_target else None,
            'active_target_count': len(self.target_tracker.targets)
        }
    
    def reset(self) -> None:
        """Reset phase manager to initial state."""
        self.current_phase = OperationalPhase.PRE_LOCK_ON
        self.target_tracker.reset()
        self.primary_target = None
        self.phase_start_frame = 0
        self.target_stable_count = 0
        self.target_lost_count = 0


class TrackingVisualizer:
    """Handles visualization of tracking information and operational phases."""
    
    @staticmethod
    def draw_targets(frame: np.ndarray, targets: List[Target], primary_target: Optional[Target]) -> None:
        """Draw target bounding boxes and tracking information."""
        for target in targets:
            x, y, w, h = target.bbox
            center = target.center
            
            # Choose color based on whether this is primary target
            if primary_target and target.id == primary_target.id:
                color = (0, 0, 255)  # Red for primary target
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for other targets
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw center point
            cv2.circle(frame, center, 3, color, -1)
            
            # Draw target ID and area
            label = f"ID:{target.id} A:{int(target.area)}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw velocity vector if significant
            if abs(target.velocity[0]) > 1 or abs(target.velocity[1]) > 1:
                end_point = (int(center[0] + target.velocity[0] * 5),
                           int(center[1] + target.velocity[1] * 5))
                cv2.arrowedLine(frame, center, end_point, color, 2)
            
            # Draw track history
            if len(target.track_history) > 1:
                for i in range(1, len(target.track_history)):
                    cv2.line(frame, target.track_history[i-1], target.track_history[i], 
                            (255, 255, 0), 1)
    
    @staticmethod
    def draw_phase_info(frame: np.ndarray, phase_manager: PhaseManager, phase_info: Dict[str, Any]) -> None:
        """Draw operational phase information on frame."""
        display_info = phase_manager.get_phase_display_info()
        
        # Phase name
        phase_text = f"Phase: {display_info['phase_name']}"
        cv2.putText(frame, phase_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Phase-specific information
        y_offset = 120
        if 'status' in phase_info:
            status_text = f"Status: {phase_info['status']}"
            cv2.putText(frame, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        if 'velocity' in phase_info:
            vel_text = f"Velocity: {phase_info['velocity']:.1f}"
            cv2.putText(frame, vel_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        # Target count
        target_text = f"Targets: {display_info['active_target_count']}"
        cv2.putText(frame, target_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    @staticmethod
    def add_crosshair(frame: np.ndarray) -> None:
        """Add crosshair to frame center."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        s = CONFIG.system.crosshair_size
        color = CONFIG.system.crosshair_color
        thickness = CONFIG.system.crosshair_thickness
        
        cv2.line(frame, (cx-s, cy+s), (cx, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx+s, cy+s), color, thickness)