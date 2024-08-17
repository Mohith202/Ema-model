import tempfile
import os
from moviepy.editor import VideoFileClip
from whisper import load_model  # Importing Whisper AI

def save_uploaded_video(uploaded_file):
    """Save the uploaded video file to a temporary location."""
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Check if uploaded_file is a string (path) or has an attribute 'name' (file-like object)
        if isinstance(uploaded_file, str):
            video_file_path = uploaded_file  # Assume it's a direct path to the file
        else:
            video_file_path = os.path.join(temp_dir, uploaded_file.name)
            # Save the uploaded video file to the temporary directory
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Check if the file was saved successfully
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Failed to save video file: {video_file_path}")
        
        print(f"Saving video to {video_file_path}")  # Add debug print to confirm path
        
        return video_file_path
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the video file: {e}")

def extract_audio_from_video(video_file):
    """Extract audio from the video file and save it as a WAV file."""
    audio_file = "temp_audio.wav"
    print(f"Extracting audio from {video_file} to {audio_file}")  # Add debug print to confirm path
    try:
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        with VideoFileClip(video_file) as video:
            video.audio.write_audiofile(audio_file, codec='pcm_s16le')
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Failed to extract audio file: {audio_file}")
        
        print(f"Extracting audio to {audio_file}")  # Add debug print to confirm path
        
        return audio_file
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting audio: {e}")

def process_video_voice(video_file):
    """Process the video file to extract voice and return the recognized text."""
    print(f"Processing video file: {video_file}")  # Add debug print to confirm path

    saved_video_file = save_uploaded_video(video_file)
    print(f"Saved video file: {saved_video_file}")  # Add debug print to confirm path

    audio_file = extract_audio_from_video(saved_video_file)
    print(f"Extracted audio file: {audio_file}")  # Add debug print to confirm path
    
    if not os.path.exists(saved_video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")
    
    model = load_model("small")
    result = model.transcribe(audio_file)
    return result['text']