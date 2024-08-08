import tempfile
import os
from moviepy.editor import VideoFileClip
import speech_recognition as sr

def save_uploaded_video(uploaded_file):
    """Save the uploaded video file to a temporary location."""
    temp_file_path = tempfile.mktemp(suffix=".mp4")  # Change suffix based on expected video format
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path

def extract_audio_from_video(video_file):
    """Extract audio from the video file and save it as a WAV file."""
    audio_file = "temp_audio.wav"
    with VideoFileClip(video_file) as video:
        video.audio.write_audiofile(audio_file, codec='pcm_s16le')
    return audio_file

def process_video_voice(video_file):
    """Process the video file to extract voice and return the recognized text."""
    audio_file = extract_audio_from_video(video_file)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    # Clean up the temporary audio file
    os.remove(audio_file)
    
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Audio could not be understood."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"