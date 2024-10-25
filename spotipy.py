import redis
import json
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, List, Optional
import logging
from datetime import datetime
import openai
import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SongPerformance:
    song_id: str
    title: str
    artist: str
    energy: float
    danceability: float
    avg_crowd_energy: float
    peak_crowd_energy: float
    duration_played: int  # in seconds
    timestamp: str

class MusicController:
    def __init__(self):
        """Initialize Spotify client and authentication."""
        self.spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri="http://localhost:8888/callback",
            scope="user-modify-playback-state user-read-playback-state playlist-read-private"
        ))
        
        # Cache for song features
        self.song_features: Dict[str, Dict] = {}
        
    def get_current_track(self) -> Optional[Dict]:
        """Get currently playing track information."""
        try:
            current = self.spotify.current_playback()
            if current and current.get('item'):
                return {
                    'id': current['item']['id'],
                    'title': current['item']['name'],
                    'artist': current['item']['artists'][0]['name'],
                    'progress_ms': current['progress_ms'],
                    'duration_ms': current['item']['duration_ms']
                }
        except Exception as e:
            logger.error(f"Error getting current track: {e}")
        return None

    def get_song_features(self, song_id: str) -> Dict:
        """Get or fetch song audio features."""
        if song_id not in self.song_features:
            try:
                features = self.spotify.audio_features([song_id])[0]
                if features:  # Ensure features are retrieved
                    self.song_features[song_id] = features
            except Exception as e:
                logger.error(f"Error fetching song features: {e}")
                return {}
        return self.song_features[song_id]

    def change_song(self, song_id: str) -> bool:
        """Change to a specific song."""
        try:
            self.spotify.start_playback(uris=[f"spotify:track:{song_id}"])
            return True
        except Exception as e:
            logger.error(f"Error changing song: {e}")
            return False

class DJBrain:
    def __init__(self):
        """Initialize the DJ decision-making system."""
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.music_controller = MusicController()
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Performance tracking
        self.song_history: List[SongPerformance] = []
        self.current_song: Optional[SongPerformance] = None
        self.last_decision_time = datetime.now()
        self.min_song_duration = 60  # Minimum seconds to play a song
        
    def get_crowd_metrics(self) -> Dict:
        """Get latest crowd metrics from Redis."""
        try:
            # Get latest entry from Redis stream
            response = self.redis_client.xread(
                {'dance_metrics': '0-0'}, 
                count=1,
                block=0
            )
            if response:
                _, messages = response[0]
                _, data = messages[0]
                return json.loads(data[b'data'].decode())
        except Exception as e:
            logger.error(f"Error reading crowd metrics: {e}")
        return {}

    def analyze_current_performance(self) -> Dict:
        """Analyze how well the current song is performing."""
        if not self.current_song:
            return {}
            
        current_time = datetime.now()
        
        # Get current metrics
        metrics = self.get_crowd_metrics()
        if not metrics:
            return {}
        
        # Update current song performance
        self.current_song.avg_crowd_energy = (
            (self.current_song.avg_crowd_energy * self.current_song.duration_played + 
             metrics.get('crowd_energy', 0)) / (self.current_song.duration_played + 1)
        )
        self.current_song.peak_crowd_energy = max(
            self.current_song.peak_crowd_energy,
            metrics.get('crowd_energy', 0)
        )
        self.current_song.duration_played += 1
        
        return {
            'current_song': self.current_song,
            'current_metrics': metrics,
            'time_played': (current_time - datetime.fromisoformat(self.current_song.timestamp)).total_seconds()
        }

    def generate_llm_prompt(self, performance: Dict) -> str:
        """Generate prompt for LLM decision making."""
        current_song = performance.get('current_song')
        metrics = performance.get('current_metrics', {})
        
        prompt = f"""As an AI DJ, analyze the current dance floor situation and decide if we should change the song.

Current Situation:
- Crowd Energy: {metrics.get('crowd_energy', 0):.2f} (0-1 scale)
- Number of Dancers: {metrics.get('num_dancers', 0)}
- Total People: {metrics.get('num_people', 0)}
- Dance Intensity: {metrics.get('dance_intensity', 0):.2f}

Current Song:
- Title: {current_song.title if current_song else 'None'}
- Artist: {current_song.artist if current_song else 'None'}
- Time Played: {performance.get('time_played', 0) if current_song else 0}
- Average Crowd Energy: {current_song.avg_crowd_energy if current_song else 0:.2f}
- Peak Crowd Energy: {current_song.peak_crowd_energy if current_song else 0:.2f}

Recent Song History:
{self._format_song_history()}

Should I:
1. Keep playing the current song
2. Change to a higher energy song
3. Change to a lower energy song

Provide your decision as a JSON object with:
- decision: (1, 2, or 3)
- confidence: (0-1)
- reasoning: (brief explanation)
- target_energy: (0-1, if decision is 2 or 3)"""

        return prompt

    def _format_song_history(self) -> str:
        """Format recent song history for prompt."""
        history = []
        for song in reversed(self.song_history[-5:]):  # Last 5 songs
            history.append(
                f"- {song.title} by {song.artist}: "
                f"avg_energy={song.avg_crowd_energy:.2f}, "
                f"peak_energy={song.peak_crowd_energy:.2f}"
            )
        return "\n".join(history)

    def get_llm_decision(self, prompt: str) -> Dict:
        """Get decision from LLM."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an experienced DJ making decisions about music selection."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error getting LLM decision: {e}")
            return {"decision": 1, "confidence": 0.5, "reasoning": "Error in LLM, defaulting to current song"}

    def select_next_song(self, target_energy: float) -> Optional[str]:
        """Select next song based on target energy level."""
        try:
            # Get user's playlists
            playlists = self.music_controller.spotify.current_user_playlists()
            all_tracks = []
            
            # Collect tracks from playlists
            for playlist in playlists['items']:
                tracks = self.music_controller.spotify.playlist_tracks(playlist['id'])
                all_tracks.extend(tracks['items'])
            
            # Filter and score tracks
            candidates = []
            for track in all_tracks:
                if not track['track']['id']:
                    continue
                    
                # Get audio features
                features = self.music_controller.get_song_features(track['track']['id'])
                if not features:
                    continue
                
                # Calculate score based on how well it matches target energy
                energy_diff = abs(features['energy'] - target_energy)
                danceability_boost = features['danceability'] * 0.5
                score = 1 - energy_diff + danceability_boost
                
                candidates.append((track['track']['id'], score))
            
            # Sort by score and pick top candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                return candidates[0][0]
                
        except Exception as e:
            logger.error(f"Error selecting next song: {e}")
        return None

    def run_dj_system(self):
        """Run the DJ decision-making loop."""
        while True:
            self.current_song = self.music_controller.get_current_track()
            performance_data = self.analyze_current_performance()
            
            if performance_data and self.current_song:
                prompt = self.generate_llm_prompt(performance_data)
                decision = self.get_llm_decision(prompt)
                
                if decision['decision'] == 2:  # Change to a higher energy song
                    next_song_id = self.select_next_song(target_energy=min(1.0, decision.get('target_energy', 0.8)))
                    if next_song_id and self.music_controller.change_song(next_song_id):
                        new_song = self.music_controller.get_song_features(next_song_id)
                        if new_song:
                            new_performance = SongPerformance(
                                song_id=new_song['id'],
                                title=new_song['name'],
                                artist=new_song['artists'][0]['name'],
                                energy=new_song['energy'],
                                danceability=new_song['danceability'],
                                avg_crowd_energy=0,
                                peak_crowd_energy=0,
                                duration_played=0,
                                timestamp=datetime.now().isoformat()
                            )
                            self.song_history.append(new_performance)
                            logger.info(f"Changed to song: {new_performance.title} by {new_performance.artist}")
                
                elif decision['decision'] == 3:  # Change to a lower energy song
                    next_song_id = self.select_next_song(target_energy=max(0.0, decision.get('target_energy', 0.2)))
                    if next_song_id and self.music_controller.change_song(next_song_id):
                        new_song = self.music_controller.get_song_features(next_song_id)
                        if new_song:
                            new_performance = SongPerformance(
                                song_id=new_song['id'],
                                title=new_song['name'],
                                artist=new_song['artists'][0]['name'],
                                energy=new_song['energy'],
                                danceability=new_song['danceability'],
                                avg_crowd_energy=0,
                                peak_crowd_energy=0,
                                duration_played=0,
                                timestamp=datetime.now().isoformat()
                            )
                            self.song_history.append(new_performance)
                            logger.info(f"Changed to song: {new_performance.title} by {new_performance.artist}")
                
            time.sleep(10)  # Poll every 10 seconds

if __name__ == "__main__":
    dj_brain = DJBrain()
    dj_brain.run_dj_system()
