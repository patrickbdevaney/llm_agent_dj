import os
import cv2
import old.spotipy as spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
from swarms import Agent
from time import sleep
from typing import Optional
from loguru import logger
from tenacity import retry, wait_fixed, stop_after_attempt
from model import VisionAPIWrapper

# Load environment variables
load_dotenv()

# Configure logging
logger.add("dj_agent_log.log", rotation="1 MB", level="DEBUG")

# Spotify API initialization
spotify = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
        redirect_uri="http://localhost:8080",
        scope="user-read-playback-state user-modify-playback-state",
    )
)

# Vision model initialization for crowd analysis
vision_llm = VisionAPIWrapper(
    api_key=os.getenv("VISION_API_KEY"),
    max_tokens=500
)

# Task prompt to analyze crowd energy for acid techno selection
task = (
    "Analyze this real-time image of a crowd. Determine the overall energy level "
    "based on body movement, facial expressions, and crowd density. "
    "If the crowd is highly energetic, recommend a high-intensity acid techno song "
    "with fast tempo and heavy bass. If energy is lower, suggest a slower acid techno song."
)


@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def get_spotify_track(track_name: str) -> Optional[str]:
    """Fetches a track ID from Spotify by name."""
    logger.info(f"Fetching Spotify track for: {track_name}")
    try:
        results = spotify.search(q=track_name, type="track", limit=1)
        track_id = results["tracks"]["items"][0]["id"]
        logger.debug(f"Found track ID: {track_id} for track name: {track_name}")
        return track_id
    except Exception as e:
        logger.error(f"Error fetching Spotify track: {e}")
        raise


def analyze_crowd(frame_path: str) -> str:
    """Analyze crowd energy from an image and return a song recommendation."""
    logger.info("Analyzing crowd energy from image.")
    try:
        agent = Agent(
            agent_name="AcidTechnoDJ_CrowdAnalyzer",
            llm=vision_llm,
            max_loops=1
        )
        response = agent.run(task, frame_path)
        logger.debug(f"Crowd analysis result: {response}")
        return response  # Expecting a recommended song title
    except Exception as e:
        logger.error(f"Error analyzing crowd: {e}")
        raise


def play_song(song_name: str) -> None:
    """Plays a song on Spotify by name."""
    logger.info(f"Attempting to play song: {song_name}")
    track_id = get_spotify_track(song_name)
    if track_id:
        spotify.start_playback(uris=[f"spotify:track:{track_id}"])
        logger.info(f"Now playing track: {track_id}")
    else:
        logger.error(f"Could not play song: {song_name}")


def save_frame(frame: cv2.Mat, path: str = "temp_frame.jpg") -> str:
    """Saves a video frame as an image file."""
    cv2.imwrite(path, frame)
    logger.info(f"Frame saved to {path}")
    return path


@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def capture_video_feed() -> Optional[cv2.VideoCapture]:
    """Initializes and returns the video capture feed."""
    logger.info("Attempting to capture video feed.")
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Failed to open camera.")
        logger.debug("Camera feed opened successfully.")
        return camera
    except Exception as e:
        logger.error(f"Error capturing video feed: {e}")
        raise


def run_dj_agent():
    """Runs the AI DJ agent for real-time crowd analysis and music playback."""
    logger.info("Starting Acid Techno DJ Agent...")

    camera = capture_video_feed()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to capture frame from video feed.")
                break

            frame_path = save_frame(frame)
            recommended_song = analyze_crowd(frame_path)
            logger.info(f"Recommended song: {recommended_song}")
            play_song(recommended_song)

            sleep(5)  # Wait for 5 seconds before next frame analysis

    except Exception as e:
        logger.error(f"DJ Agent encountered an error: {e}")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        logger.info("DJ Agent has stopped.")


# Run the DJ agent
if __name__ == "__main__":
    run_dj_agent()
