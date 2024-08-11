import tempfile
import os
from moviepy.editor import VideoFileClip
from whisper import load_model  # Importing Whisper AI

def save_uploaded_video(uploaded_file):
    """Save the uploaded video file to a temporary location."""
    try:
        # Create a temporary directory to save the uploaded video file
        temp_dir = tempfile.mkdtemp()
        video_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save the uploaded video file to the temporary directory
        with open(video_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Check if the file was saved successfully
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Failed to save video file: {video_file_path}")
        
        return video_file_path
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the video file: {e}")

def extract_audio_from_video(video_file):
    """Extract audio from the video file and save it as a WAV file."""
    audio_file = "temp_audio.wav"
    try:
        with VideoFileClip(video_file) as video:
            video.audio.write_audiofile(audio_file, codec='pcm_s16le')
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Failed to extract audio file: {audio_file}")
        return audio_file
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting audio: {e}")

def process_video_voice(video_file):
    """Process the video file to extract voice and return the recognized text."""
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")
    
    model = load_model("medium")
    result = model.transcribe(video_file)
    return result['text']