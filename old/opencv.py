import cv2
import mediapipe as mp
import numpy as np
import redis
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import time
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DanceMetrics:
    timestamp: str
    num_people: int
    num_dancers: int
    avg_movement: float
    crowd_density: float
    dance_intensity: float
    crowd_energy: float

class PersonTracker:
    def __init__(self, tracking_id: int):
        self.tracking_id = tracking_id
        self.movement_history = deque(maxlen=30)  # 1 second at 30fps
        self.previous_positions: Dict[int, np.ndarray] = {}
        self.is_dancing = False
        self.last_updated = time.time()
        
    def update_movement(self, landmarks) -> float:
        """Calculate movement for a single person based on key landmarks."""
        key_points = [
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP
        ]
        
        current_positions = {}
        for point in key_points:
            landmark = landmarks[point.value]
            current_positions[point.value] = np.array([landmark.x, landmark.y, landmark.z])
        
        if not self.previous_positions:
            self.previous_positions = current_positions
            return 0.0
            
        total_movement = sum(
            np.linalg.norm(current_positions[point_id] - self.previous_positions[point_id])
            for point_id in current_positions if point_id in self.previous_positions
        )
        
        self.previous_positions = current_positions
        avg_movement = total_movement / len(key_points)
        self.movement_history.append(avg_movement)
        self.last_updated = time.time()
        
        return avg_movement

class CrowdAnalyzer:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_size = 32  # Divide frame into 32x32 cells
        self.occupation_grid = np.zeros((self.grid_size, self.grid_size))
        
    def update_crowd_density(self, people_positions: List[np.ndarray]) -> float:
        """Calculate crowd density using grid occupation."""
        self.occupation_grid.fill(0)
        
        for position in people_positions:
            grid_x = int(position[0] * self.grid_size)
            grid_y = int(position[1] * self.grid_size)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.occupation_grid[grid_y, grid_x] = 1
                
        return np.mean(self.occupation_grid)

class EnhancedDanceDetector:
    def __init__(self):
        """Initialize the enhanced dance detection system."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,  # Use more accurate model
            enable_segmentation=True  # Help with occlusion
        )
        
        # Initialize Redis connection with retry logic
        self.redis_client = self._init_redis_connection()
        
        # Track multiple people
        self.people: Dict[int, PersonTracker] = {}
        self.next_id = 0
        
        # Analysis parameters
        self.dancing_threshold = 0.25
        self.movement_threshold = 0.1
        self.cleanup_threshold = 1.0  # Remove trackers after 1 second of no updates
        
        # Initialize crowd analyzer
        self.crowd_analyzer = None  # Will be initialized with the first frame
        
        # Metrics aggregation
        self.metrics_history = deque(maxlen=90)  # 3 seconds at 30fps
        
        # Background thread for metrics publishing
        self.metrics_thread = threading.Thread(target=self._publish_metrics_loop, daemon=True)
        self.running = True
        self.metrics_thread.start()
        
    def _init_redis_connection(self) -> redis.Redis:
        """Initialize Redis connection with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    socket_timeout=1,
                    socket_connect_timeout=1,
                    retry_on_timeout=True
                )
                client.ping()  # Test connection
                return client
            except redis.RedisError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                    raise
                time.sleep(1)

    def send_metrics_to_swarms(self, metrics: DanceMetrics):
        """Send metrics to the Swarms agent."""
        logger.info(f"Sending metrics to Swarms: {metrics}")
        # Example of sending data to the Swarms agent (modify as needed):
        # response = requests.post('http://swarms-agent-endpoint/metrics', json=metrics.__dict__)
        # if response.status_code != 200:
        #     logger.error(f"Failed to send metrics to Swarms: {response.text}")

    def _cleanup_stale_trackers(self):
        """Remove trackers for people who haven't been updated recently."""
        current_time = time.time()
        stale_ids = [
            pid for pid, person in self.people.items()
            if current_time - person.last_updated > self.cleanup_threshold
        ]
        for pid in stale_ids:
            del self.people[pid]
            
    def calculate_metrics(self) -> Optional[DanceMetrics]:
        """Calculate comprehensive dance metrics."""
        if not self.people:
            return None
            
        movements = []
        num_dancers = 0
        positions = []
        
        for person in self.people.values():
            if person.movement_history:
                avg_movement = np.mean(list(person.movement_history))
                movements.append(avg_movement)
                if avg_movement > self.dancing_threshold:
                    num_dancers += 1
                    
            if person.previous_positions:
                hip_position = person.previous_positions.get(
                    mp.solutions.pose.PoseLandmark.LEFT_HIP.value
                )
                if hip_position is not None:
                    positions.append(hip_position[:2])  # Only use x,y coordinates
                    
        if not movements:
            return None
            
        crowd_density = self.crowd_analyzer.update_crowd_density(positions)
        avg_movement = np.mean(movements)
        dance_intensity = np.mean([m for m in movements if m > self.dancing_threshold]) if num_dancers > 0 else 0
        
        crowd_energy = (
            0.4 * (num_dancers / max(len(self.people), 1)) +  # Proportion of people dancing
            0.3 * (avg_movement / self.dancing_threshold) +    # Overall movement level
            0.3 * crowd_density                               # How packed the space is
        )
        
        return DanceMetrics(
            timestamp=datetime.now().isoformat(),
            num_people=len(self.people),
            num_dancers=num_dancers,
            avg_movement=float(avg_movement),
            crowd_density=float(crowd_density),
            dance_intensity=float(dance_intensity),
            crowd_energy=float(crowd_energy)
        )
        
    def _publish_metrics_loop(self):
        """Background thread for publishing metrics to Redis and Swarms."""
        while self.running:
            try:
                metrics = self.calculate_metrics()
                if metrics:
                    # Publish to Redis
                    self.redis_client.xadd(
                        'dance_metrics',
                        {'data': json.dumps(metrics.__dict__)},
                        maxlen=1000
                    )
                    # Send metrics to Swarms
                    self.send_metrics_to_swarms(metrics)
            except Exception as e:
                logger.error(f"Error publishing metrics: {e}")
            time.sleep(0.1)  # Publish at 10Hz
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[DanceMetrics]]:
        """Process a single frame and return the annotated frame and metrics."""
        if self.crowd_analyzer is None:
            self.crowd_analyzer = CrowdAnalyzer(frame.shape[1], frame.shape[0])
            
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Make detection
        results = self.pose.process(image_rgb)
        
        # Convert back to BGR and make writeable
        image_rgb.flags.writeable = True
        annotated_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        self._cleanup_stale_trackers()
        
        if results.pose_landmarks:
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                if landmark.visibility < 0.5:
                    continue  # Skip low visibility landmarks
                    
                # Check if this person is already being tracked
                person_tracker = self.people.get(id)
                
                if person_tracker is None:
                    person_tracker = PersonTracker(tracking_id=self.next_id)
                    self.people[self.next_id] = person_tracker
                    self.next_id += 1
                    
                # Update movement
                avg_movement = person_tracker.update_movement(results.pose_landmarks.landmark)
                
                if avg_movement > self.movement_threshold:
                    person_tracker.is_dancing = True
                    
                # Draw landmarks on the frame for visualization
                self._draw_landmarks(annotated_frame, results.pose_landmarks, person_tracker.tracking_id)
                
        # Return the annotated frame and computed metrics
        metrics = self.calculate_metrics()
        return annotated_frame, metrics
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks, tracking_id: int):
        """Draw landmarks on the frame for visualization."""
        for idx, landmark in enumerate(landmarks.landmark):
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        cv2.putText(frame, f'ID: {tracking_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def stop(self):
        """Gracefully stop the metrics publishing thread."""
        self.running = False
        self.metrics_thread.join()
