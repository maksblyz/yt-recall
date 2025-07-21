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

load_dotenv()

app = FastAPI(title="Active Recall API")

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
    
    # Calculate chunk size (aim for ~20MB chunks to be safe)
    chunk_size_mb = 20
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # Get audio duration using FFmpeg
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
                '-y',  # Overwrite output file
                chunk_file
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            # Check chunk file size
            chunk_size = os.path.getsize(chunk_file)
            print(f"Chunk {i+1}/{num_chunks}: {chunk_size / (1024*1024):.1f}MB, {end_time - start_time:.1f}s")
            
            # Transcribe chunk
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
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        video_id = None
        audio_file = None
        
        try:
            print("Step 1: Creating yt-dlp instance")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Step 2: Extracting video info")
                # First, try to extract info without downloading
                info = ydl.extract_info(request.url, download=False)
                video_id = info.get('id')
                audio_file = f"{video_id}.mp3"
                print(f"Step 3: Video ID extracted: {video_id}")
                
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
        
        # Transcribe with Whisper using OpenAI API
        try:
            print("Step 7: Starting transcription")
            
            if file_size > max_size:
                # Use chunking for large files
                transcript = await transcribe_large_file(audio_file, file_size)
            else:
                # Use normal transcription for smaller files
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
                
            # Store transcript in memory (in production, you'd use a database)
            # For now, we'll store it in a simple dict with video_id as key
            if not hasattr(start_transcription, 'transcripts'):
                start_transcription.transcripts = {}
            
            start_transcription.transcripts[video_id] = transcript
                
        except Exception as e:
            # Clean up audio file even if transcription fails
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"Step 9: Audio file cleaned up after error")
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}"
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
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/compare-recall", response_model=RecallResponse)
async def compare_recall(request: RecallRequest):
    """Compare user recall with pre-transcribed video content"""
    print(f"Starting compare_recall for video: {request.video_url}")
    print(f"User recall length: {len(request.user_recall)}")
    
    # Add a simple test response first
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
        
        # Check if we have a pre-transcribed version
        if hasattr(start_transcription, 'transcripts') and video_id in start_transcription.transcripts:
            transcript = start_transcription.transcripts[video_id]
            print(f"Using pre-transcribed content, length: {len(transcript)}")
        else:
            # Fallback: transcribe now (this shouldn't happen in normal flow)
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
            # Provide a more helpful demo response
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
            comparison = json.loads(comparison_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            comparison = {
                "correct": ["Analysis completed"],
                "incorrect": ["Could not parse detailed results"],
                "missed": ["See feedback for details"],
                "score": 50,
                "feedback": comparison_text
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