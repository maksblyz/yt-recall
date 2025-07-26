# Active Recall Learning App

A simple web application that helps users practice active recall learning with YouTube videos.

## How it works

1. User enters a YouTube video link
2. The video is displayed on the page
3. User clicks the "Recall" button
4. A text box appears where the user types everything they remember
5. The app transcribes the video using Whisper and compares the user's response with the transcript
6. Results show what was correct, incorrect, or missed

## Tech Stack

- **Frontend**: React with TypeScript
- **Backend**: FastAPI (Python)
- **Transcription**: OpenAI Whisper API
- **Comparison**: Groq API (cheap and fast)

## Quick Start

### Option 1: Automated Setup
```bash
# Run the setup script
./setup.sh
```

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Create .env file with your API keys
echo "OPENAI_API_KEY=your_openai_api_key" > .env
echo "GROQ_API_KEY=your_groq_api_key" >> .env

# Start the backend
uvicorn main:app --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Option 3: Docker Setup
```bash
# Set environment variables
export OPENAI_API_KEY=your_openai_api_key
export GROQ_API_KEY=your_groq_api_key

# Start with Docker Compose
docker-compose up --build
```

## API Keys Required

You'll need to get API keys from:

1. **Groq API Key**: https://console.groq.com/keys
   - Used for text comparison
   - Very cheap and fast
  
2. (Optional) **OpenAI API Key**: https://platform.openai.com/api-keys
   - Used for Whisper transcription
   - Costs ~$0.006 per minute of audio

## Usage

1. Open the app in your browser (http://localhost:3000)
2. Paste a YouTube video URL
3. Watch the video
4. Click "Recall" and type what you remember
5. Get detailed feedback on your recall accuracy

## Features

- ✅ YouTube video embedding
- ✅ Audio transcription with Whisper
- ✅ Intelligent recall comparison
- ✅ Detailed feedback with scoring
- ✅ Modern, responsive UI
- ✅ Docker support for easy deployment

## Development

The app is structured as follows:

```
active-recall/
├── backend/           # FastAPI server
│   ├── main.py       # Main API endpoints
│   └── requirements.txt
├── frontend/         # React app
│   ├── src/
│   │   ├── App.tsx   # Main component
│   │   └── index.tsx
│   └── package.json
└── README.md
```

## Contributing

This is an open source project! Feel free to submit issues and pull requests. 
