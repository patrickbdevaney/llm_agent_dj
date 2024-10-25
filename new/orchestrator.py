import sys
import time
import base64
import rtmidi
from loguru import logger
from old.spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from VisionAPIWrapper import VisionAPIWrapper  # Import VisionAPIWrapper

# Spotify Authentication
sp = Spotify(auth_manager=SpotifyOAuth(
    client_id="your_client_id",
    client_secret="your_client_secret",
    redirect_uri="http://localhost:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

# Initialize Vision API
api_key = "your_openai_api_key"
vision_api = VisionAPIWrapper(api_key=api_key)

# Initialize the MIDI interface
midi_in = rtmidi.MidiIn()
available_ports = midi_in.get_ports()

if available_ports:
    midi_in.open_port(0)
else:
    logger.warning("No MIDI ports available.")

# Function to search and play a track on Spotify
def play_spotify_track(track_name: str):
    logger.info(f"Searching for track: {track_name}")
    results = sp.search(q=track_name, type="track", limit=1)
    tracks = results.get('tracks', {}).get('items', [])
    if tracks:
        track_uri = tracks[0]['uri']
        sp.start_playback(uris=[track_uri])
        logger.info(f"Playing track: {track_name}")
    else:
        logger.warning(f"Track {track_name} not found on Spotify.")

# Function to stop playback on Spotify
def stop_spotify_track():
    sp.pause_playback()
    logger.info("Playback stopped on Spotify.")

# MIDI event handling loop
def handle_midi_event(event):
    message, _ = event
    control, value = message[1], message[2]
    logger.info(f"MIDI event: Control {control}, Value {value}")

    if control == 1 and value > 0:  # Example: Start a track with control 1
        play_spotify_track("your favorite track")
    elif control == 2 and value > 0:  # Example: Start another track with control 2
        play_spotify_track("another favorite track")
    elif control == 3 and value > 0:  # Stop track with control 3
        stop_spotify_track()
    elif control == 4 and value > 0:  # Trigger VisionAPI
        logger.info("Triggering Vision API.")
        output = vision_api.run("Describe the scene in the image.", img="path/to/image.jpg")
        logger.info(f"VisionAPI Output: {output}")

# Main loop
def main():
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.info("DJ setup initialized. Waiting for MIDI input.")

    try:
        while True:
            # Listen for MIDI events
            midi_events = midi_in.get_message()
            if midi_events:
                handle_midi_event(midi_events)

            # Check every so often
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    finally:
        # Cleanup
        midi_in.close_port()

if __name__ == "__main__":
    main()
