from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
import yt_dlp
from openai import OpenAI
import requests
from dotenv import load_dotenv
import json
import subprocess
import math
from datetime import datetime
import socket
from faster_whisper import WhisperModel
from pathlib import Path

load_dotenv()

# Initialize faster-whisper model
_whisper_model = None
_model_loading = False
_model_loaded = False

def get_whisper_model():
    """Get or initialize the faster-whisper model with local caching"""
    global _whisper_model, _model_loading, _model_loaded
    
    # Check if faster-whisper is disabled
    if os.getenv("DISABLE_FASTER_WHISPER", "false").lower() == "true":
        print("faster-whisper disabled via environment variable")
        return None
    
    # If model is already loaded, return it
    if _whisper_model is not None and _model_loaded:
        return _whisper_model
    
    # If model is currently loading, wait
    if _model_loading:
        print("Model is currently loading, please wait...")
        while _model_loading:
            import time
            time.sleep(0.1)
        return _whisper_model if _model_loaded else None
    
    # Start loading the model
    _model_loading = True
    print("Initializing faster-whisper model...")
    
    try:
        # Choose model size based on environment  var - default tiny
        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")  
        model_map = {
            "tiny": "Systran/faster-whisper-tiny",
            "base": "Systran/faster-whisper-base",
            "small": "Systran/faster-whisper-small",
            "medium": "Systran/faster-whisper-medium",
            "large": "Systran/faster-whisper-large-v3",
            "large-v2": "Systran/faster-whisper-large-v2",
            "distil-large": "Systran/faster-whisper-distil-large-v3",
            "turbo": "Systran/faster-whisper-turbo"
        }
        
        model_name = model_map.get(model_size, "Systran/faster-whisper-base")
        
        # Check if model is already downloaded locally
        model_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_path = None
        
        # Look for the model in the cache directory
        for cache_path in model_cache_dir.glob("**/*"):
            if cache_path.is_dir() and model_name.split("/")[-1] in cache_path.name:
                model_path = str(cache_path)
                print(f"Found cached model at: {model_path}")
                break
        
        if model_path:
            print(f"Using cached model: {model_name}")
        else:
            print(f"Model not found in cache, will download: {model_name}")
        
        # Try to use GPU if available, otherwise fall back to CPU
        device = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        _whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        print("faster-whisper model initialized successfully")
        _model_loaded = True
    except Exception as e:
        print(f"Failed to initialize faster-whisper model: {e}")
        print("Trying different compute type...")
        try:
            # Try with int8 for CPU
            _whisper_model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8"
            )
            print("faster-whisper model initialized on CPU with int8")
            _model_loaded = True
        except Exception as e2:
            print(f"int8 failed, trying default...")
            try:
                _whisper_model = WhisperModel(
                    model_name,
                    device="cpu"
                )
                print("faster-whisper model initialized on CPU with default settings")
                _model_loaded = True
            except Exception as e3:
                print(f"Failed to initialize faster-whisper: {e3}")
                _whisper_model = None
                _model_loaded = False
    
    _model_loading = False
    return _whisper_model if _model_loaded else None


def find_available_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

app = FastAPI(title="Active Recall API")

@app.on_event("startup")
async def startup_event():
    """Pre-load the whisper model on startup"""
    print("Starting up Active Recall API...")
    print("Pre-loading faster-whisper model...")
    
    # Pre-load the whisper model
    whisper_model = get_whisper_model()
    if whisper_model is not None:
        print("faster-whisper model pre-loaded successfully")
    else:
        print("faster-whisper model not available, will use OpenAI API fallback")
    
    print("API ready to serve requests!")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_api_key = os.getenv("GROQ_API_KEY")

async def transcribe_large_file(audio_file: str, file_size: int) -> str:
    """Transcribe a large audio file by chunking it into smaller segments"""
    print(f"Starting chunked transcription for {file_size / (1024*1024):.1f}MB file")
    
    # Calculate chunk size
    chunk_size_mb = 20
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # Get audio duration
    try:
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', audio_file
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Could not get duration, using default: {e}")
        duration = 3600  # Default to 1 hour
    
    # Calculate chunk duration
    chunk_duration = (chunk_size_bytes / file_size) * duration
    num_chunks = math.ceil(duration / chunk_duration)
    
    print(f"Will split into {num_chunks} chunks of ~{chunk_duration:.1f} seconds each")
    
    transcripts = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)
        
        # Create temporary chunk file
        chunk_file = f"{audio_file}_chunk_{i}.mp3"
        
        try:
            # Extract chunk using FFmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-i', audio_file,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',
                # Overwrite output file 
                '-y',
                chunk_file
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            # Check chunk file size
            chunk_size = os.path.getsize(chunk_file)
            print(f"Chunk {i+1}/{num_chunks}: {chunk_size / (1024*1024):.1f}MB, {end_time - start_time:.1f}s")
            
            # Transcribe chunk
            whisper_model = get_whisper_model()
            if whisper_model is not None:
                # Use faster-whisper for chunk
                segments, info = whisper_model.transcribe(
                    chunk_file,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Combine segments
                chunk_parts = []
                for segment in segments:
                    chunk_parts.append(segment.text.strip())
                chunk_transcript = " ".join(chunk_parts).strip()
            else:
                # Fallback to OpenAI API
                with open(chunk_file, "rb") as audio:
                    transcript_response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio
                    )
                chunk_transcript = transcript_response.text.strip()
            transcripts.append(chunk_transcript)
            print(f"Chunk {i+1} transcribed: {len(chunk_transcript)} characters")
            
        except Exception as e:
            print(f"Error transcribing chunk {i+1}: {e}")
            transcripts.append(f"[Error transcribing segment {i+1}]")
        finally:
            # Clean up chunk file
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
    
    # Combine all transcripts
    full_transcript = " ".join(transcripts)
    print(f"Combined transcript length: {len(full_transcript)} characters")
    
    return full_transcript

class VideoRequest(BaseModel):
    url: str

class RecallRequest(BaseModel):
    video_url: str
    user_recall: str

class RecallResponse(BaseModel):
    transcript: str
    comparison: dict
    score: float

class SaveRecallRequest(BaseModel):
    video_url: str
    user_recall: str
    transcript: str
    comparison: dict
    score: float
    video_title: str = ""

@app.get("/")
async def root():
    return {"message": "Active Recall API is running"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to check if the API is working"""
    return {
        "status": "ok",
        "message": "API is working",
        "openai_key": "present" if os.getenv("OPENAI_API_KEY") else "missing",
        "groq_key": "present" if os.getenv("GROQ_API_KEY") else "missing"
    }

@app.get("/test-video")
async def test_video():
    """Test video processing with a simple, short video"""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - short and usually accessible
    
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'test_%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(test_url, download=False)
            video_id = info.get('id')
            title = info.get('title', 'Unknown')
            
        return {
            "status": "success",
            "video_id": video_id,
            "title": title,
            "message": "Video info extracted successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Video processing failed"
        }

@app.get("/test-whisper")
async def test_whisper():
    """Test faster-whisper model initialization"""
    try:
        whisper_model = get_whisper_model()
        if whisper_model is not None:
            return {
                "status": "success",
                "message": "faster-whisper model initialized successfully",
                "device": "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu",
                "model_size": os.getenv("WHISPER_MODEL_SIZE", "tiny"),
                "model_loading": _model_loading,
                "model_loaded": _model_loaded,
                "openai_fallback": bool(os.getenv("OPENAI_API_KEY"))
            }
        else:
            return {
                "status": "error",
                "message": "faster-whisper model failed to initialize or is disabled",
                "fallback_available": bool(os.getenv("OPENAI_API_KEY")),
                "disabled": os.getenv("DISABLE_FASTER_WHISPER", "false").lower() == "true",
                "model_size": os.getenv("WHISPER_MODEL_SIZE", "tiny"),
                "model_loading": _model_loading,
                "model_loaded": _model_loaded
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing whisper: {str(e)}",
            "fallback_available": bool(os.getenv("OPENAI_API_KEY")),
            "model_loading": _model_loading,
            "model_loaded": _model_loaded
        }

# Removed transcripts endpoint - no longer saving transcripts to files

@app.post("/process-video", response_model=dict)
async def process_video(request: VideoRequest):
    """Download video and extract video ID for embedding"""
    try:
        # Extract video ID from URL
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=False)
            video_id = info.get('id')
            title = info.get('title', 'Unknown Title')
            
        return {
            "video_id": video_id,
            "title": title,
            "embed_url": f"https://www.youtube.com/embed/{video_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing video: {str(e)}")

@app.post("/start-transcription", response_model=dict)
async def start_transcription(request: VideoRequest):
    """Start downloading and transcribing video in background"""
    print(f"Starting transcription for video: {request.url}")
    
    try:
        # Download video audio with better error handling
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'no_color': True,
            'extractor_retries': 3,
            'noplaylist': True,  # Only download the single video, not the entire playlist
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            # Removed cookiesfrombrowser to avoid password prompts
            'cookiefile': None,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'add_header': [
                'Referer: https://www.youtube.com/',
                'Origin: https://www.youtube.com'
            ]
        }
        
        video_id = None
        audio_file = None
        
        try:
            print("Step 1: Creating yt-dlp instance")
            print(f"Step 1.5: URL to process: {request.url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Step 2: Extracting video info")
                # First, try to extract info without downloading
                info = ydl.extract_info(request.url, download=False)
                video_id = info.get('id')
                audio_file = f"{video_id}.mp3"
                print(f"Step 3: Video ID extracted: {video_id}")
                print(f"Step 3.5: Video title: {info.get('title', 'Unknown')}")
                
                print("Step 4: Starting download")
                # Try to download the audio
                ydl.download([request.url])
                print("Step 5: Download completed")
        except Exception as e:
            # Try with different format options
            print(f"First download attempt failed: {e}")
            try:
                ydl_opts_fallback = {
                    'format': 'worstaudio/worst',  # Try worst quality first
                    'outtmpl': '%(id)s.%(ext)s',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '64',  # Lower quality
                    }],
                    'quiet': True,
                    'no_warnings': True,
                    'nocheckcertificate': True,
                    'ignoreerrors': False,
                    'no_color': True,
                    'noplaylist': True,  # Only download the single video, not the entire playlist
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-us,en;q=0.5',
                        'Accept-Encoding': 'gzip,deflate',
                        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    },
                    'no_check_certificate': True,
                    'prefer_insecure': True,
                    'add_header': [
                        'Referer: https://www.youtube.com/',
                        'Origin: https://www.youtube.com'
                    ]
                }
                
                with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                    ydl.download([request.url])
                    
            except Exception as e2:
                print(f"Fallback download also failed: {e2}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not download video audio. This video might be restricted or unavailable. Try a different video. Error: {str(e2)}"
                )
        
        # Check if audio file exists
        if not os.path.exists(audio_file):
            raise HTTPException(
                status_code=400,
                detail="Audio file was not created. YouTube download may have failed."
            )
        
        # Check file size limit (OpenAI has 25MB limit)
        file_size = os.path.getsize(audio_file)
        print(f"Step 6: Audio file exists, size: {file_size} bytes")
        
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        if file_size > max_size:
            print(f"File too large ({file_size / (1024*1024):.1f}MB), will chunk into smaller segments")
            # Don't remove the file yet, we'll chunk it
        
        # Transcribe with faster-whisper (local) or OpenAI API (fallback)
        try:
            print("Step 7: Starting transcription")
            
            # Try faster-whisper first
            print("🔍 Checking faster-whisper model availability...")
            whisper_model = get_whisper_model()
            if whisper_model is not None:
                print("faster-whisper model is ready")
                print("Using faster-whisper for local transcription")
                if file_size > max_size:
                    # Use chunking for large files
                    transcript = await transcribe_large_file(audio_file, file_size)
                else:
                    # Use faster-whisper for normal files
                    print("Starting transcription with faster-whisper...")
                    
                    # Add timeout protection for transcription
                    import asyncio
                    import concurrent.futures
                    
                    def transcribe_with_timeout():
                        segments, info = whisper_model.transcribe(
                            audio_file,
                            beam_size=1,  # Reduced for speed
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        # Combine all segments into one transcript
                        print("Processing transcription segments...")
                        transcript_parts = []
                        segment_count = 0
                        for segment in segments:
                            transcript_parts.append(segment.text.strip())
                            segment_count += 1
                            if segment_count % 10 == 0:  # Progress indicator every 10 segments
                                print(f"Processed {segment_count} segments...")
                        
                        transcript = " ".join(transcript_parts).strip()
                        print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
                        print(f"Transcription completed: {len(transcript)} characters")
                        return transcript
                    
                    # Run transcription w timeout
                    try:
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            transcript = await asyncio.wait_for(
                                loop.run_in_executor(executor, transcribe_with_timeout),
                                timeout=120 
                            )
                    except asyncio.TimeoutError:
                        print("Transcription timed out after 10 minutes")
                        raise HTTPException(
                            status_code=408,
                            detail="Transcription timed out. The video might be too long or the model is taking too long to process. Try a shorter video or check your system resources."
                        )
            else:
                # Fallback to OpenAI API if faster-whisper is not available
                print("faster-whisper not available, falling back to OpenAI API")
                if not os.getenv("OPENAI_API_KEY"):
                    raise HTTPException(
                        status_code=500,
                        detail="No transcription method available. Please install faster-whisper or set OPENAI_API_KEY"
                    )
                
                if file_size > max_size:
                    # Use chunking for large files
                    transcript = await transcribe_large_file(audio_file, file_size)
                else:
                    # Use OpenAI API for normal files
                    with open(audio_file, "rb") as audio:
                        transcript_response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio
                        )
                    transcript = transcript_response.text.strip()
            
            print(f"Step 8: Transcription completed, length: {len(transcript)}")
            
            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print("Step 9: Audio file cleaned up")
                
            # Store transcript in memory
            if not hasattr(start_transcription, 'transcripts'):
                start_transcription.transcripts = {}
            
            start_transcription.transcripts[video_id] = transcript
                
        except Exception as e:
            # Clean up audio file even if transcription fails
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"Step 9: Audio file cleaned up after error")
            import traceback
            transcription_error = f"Transcription failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(transcription_error)
            raise HTTPException(
                status_code=500,
                detail=transcription_error
            )
        
        return {
            "status": "success",
            "video_id": video_id,
            "transcript_length": len(transcript),
            "message": "Transcription completed successfully"
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error in start_transcription: {str(e)}"
        full_traceback = traceback.format_exc()
        print(error_msg)
        print(f"Full traceback: {full_traceback}")
        raise HTTPException(status_code=500, detail=f"{error_msg}\n\nTraceback:\n{full_traceback}")

@app.post("/compare-recall", response_model=RecallResponse)
async def compare_recall(request: RecallRequest):
    """Compare user recall with pre-transcribed video content"""
    print(f"Starting compare_recall for video: {request.video_url}")
    print(f"User recall length: {len(request.user_recall)}")
    
    #test response 
    if request.user_recall == "test":
        return RecallResponse(
            transcript="Test transcript",
            comparison={
                "correct": ["Test successful"],
                "incorrect": ["Test mode"],
                "missed": ["Test mode"],
                "score": 100,
                "feedback": "Test mode working"
            },
            score=100
        )
    
    try:
        # Extract video ID from URL
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.video_url, download=False)
            video_id = info.get('id')
        
        # Check for pre-transcribed version
        if hasattr(start_transcription, 'transcripts') and video_id in start_transcription.transcripts:
            transcript = start_transcription.transcripts[video_id]
            print(f"Using pre-transcribed content, length: {len(transcript)}")
        else:
            # Fallback to transcribe now
            print("No pre-transcribed content found, falling back to live transcription")
            
            # Download and transcribe the video
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': '%(id)s.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            audio_file = None
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([request.video_url])
                    audio_file = f"{video_id}.mp3"
                
                if not os.path.exists(audio_file):
                    raise HTTPException(status_code=400, detail="Could not download video audio")
                
                file_size = os.path.getsize(audio_file)
                max_size = 25 * 1024 * 1024  # 25MB
                
                if file_size > max_size:
                    transcript = await transcribe_large_file(audio_file, file_size)
                else:
                    # Try faster-whisper first
                    whisper_model = get_whisper_model()
                    if whisper_model is not None:
                        print("Using faster-whisper for fallback transcription")
                        segments, info = whisper_model.transcribe(
                            audio_file,
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        # Combine all segments
                        transcript_parts = []
                        for segment in segments:
                            transcript_parts.append(segment.text.strip())
                        
                        transcript = " ".join(transcript_parts).strip()
                    else:
                        # Fallback to OpenAI API
                        with open(audio_file, "rb") as audio:
                            transcript_response = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio
                            )
                        transcript = transcript_response.text.strip()
                
                # Store for future use
                if not hasattr(start_transcription, 'transcripts'):
                    start_transcription.transcripts = {}
                start_transcription.transcripts[video_id] = transcript
                
            except Exception as e:
                if audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)
                raise HTTPException(
                    status_code=500,
                    detail=f"Live transcription failed: {str(e)}"
                )
            finally:
                if audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)
        
        # Compare with Groq or fallback
        comparison_prompt = f"""
        Compare your recall with the actual transcript and provide detailed feedback.
        
        TRANSCRIPT:
        {transcript}
        
        YOUR RECALL:
        {request.user_recall}
        
        Please analyze and provide:
        1. What you got correct (focus on concepts and ideas, not exact wording)
        2. What you got wrong or misunderstood (only factual errors, not language differences)
        3. What you missed entirely (important concepts or points)
        4. A percentage score (0-100) based on conceptual accuracy

        Refer to the user as "you" in your response.
        
        IMPORTANT GUIDELINES:
        - Focus on concepts and ideas, not exact vocabulary or language
        - Don't penalize informal language, abbreviations, or different word choices
        - Only mark things as "incorrect" if they are factually wrong
        - Names of people, algorithms, companies, and specific technical terms should be accurate
        - Be generous with understanding - if the concept is right, the language doesn't matter, unless it's a specific technical term.
        
        Return your response as a JSON object with these keys:
        - correct: list of concepts you got right
        - incorrect: list of factual errors you made
        - missed: list of important concepts you missed
        - score: percentage score (0-100)
        - feedback: overall feedback and suggestions
        
        IMPORTANT: Return ONLY the JSON object, no markdown formatting, no code blocks, no additional text.
        """
        
        # Compare with Groq using OpenAI-compatible API
        if groq_api_key:
            try:
                groq_url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "moonshotai/kimi-k2-instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": comparison_prompt
                        }
                    ],
                    "temperature": 0.3
                }
                
                response = requests.post(groq_url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    comparison_text = result["choices"][0]["message"]["content"]
                else:
                    comparison_text = f"Groq API error: {response.status_code} - {response.text}"
            except Exception as e:
                comparison_text = f"Could not analyze with Groq API: {str(e)}"
        else:
            
            comparison_text = f"""
            {{
                "correct": ["Demo mode - transcript successfully generated"],
                "incorrect": ["Add GROQ_API_KEY to backend/.env for AI analysis"],
                "missed": ["Full AI-powered comparison available with API key"],
                "score": 75,
                "feedback": "Great! The transcription worked. Your recall can now be compared with this transcript: {transcript[:300]}... To get detailed AI analysis, add your GROQ_API_KEY to the backend/.env file."
            }}
            """
        
        # Try to parse JSON response
        try:
            # Clean up the response text - remove markdown formatting
            cleaned_text = comparison_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]  # Remove ```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove trailing ```
            
            cleaned_text = cleaned_text.strip()
            
            comparison = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw response: {comparison_text}")
            # Fallback if JSON parsing fails
            comparison = {
                "correct": ["Analysis completed"],
                "incorrect": ["Could not parse detailed results"],
                "missed": ["See feedback for details"],
                "score": 50,
                "feedback": f"JSON parsing error: {str(e)}\n\nRaw response: {comparison_text}"
            }
        
        return RecallResponse(
            transcript=transcript,
            comparison=comparison,
            score=comparison.get("score", 0)
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Error in compare_recall: {str(e)}"
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/save-recall")
async def save_recall(request: SaveRecallRequest):
    """Save recall response and analysis to local data folder"""
    try:
        # Create data directory if it doesn't exist
        data_dir = "../data/recalls"
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract video ID for filename
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.video_url, download=False)
            video_id = info.get('id')
        
        filename = f"{video_id}_{timestamp}.json"
        filepath = os.path.join(data_dir, filename)
        
        # Prepare data to save
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "video_url": request.video_url,
            "video_title": request.video_title,
            "video_id": video_id,
            "user_recall": request.user_recall,
            "transcript": request.transcript,
            "comparison": request.comparison,
            "score": request.score,
            "transcript_length": len(request.transcript),
            "recall_length": len(request.user_recall)
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": f"Recall saved to {filename}",
            "filename": filename,
            "filepath": filepath
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error saving recall: {str(e)}"
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 