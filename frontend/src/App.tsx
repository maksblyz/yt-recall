import React, { useState } from 'react';

interface VideoInfo {
  video_id: string;
  title: string;
  embed_url: string;
}

interface RecallResults {
  transcript: string;
  comparison: {
    correct: string[];
    incorrect: string[];
    missed: string[];
    score: number;
    feedback: string;
  };
  score: number;
}

function App() {
  const [videoUrl, setVideoUrl] = useState('');
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [showRecall, setShowRecall] = useState(false);
  const [userRecall, setUserRecall] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<RecallResults | null>(null);
  const [transcribing, setTranscribing] = useState(false);

  const handleVideoSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoUrl.trim()) return;

    setLoading(true);
    setError('');
    setVideoInfo(null);
    setShowRecall(false);
    setResults(null);

    try {
      // First, get video info
      const response = await fetch('/process-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: videoUrl }),
      });

      if (!response.ok) {
        throw new Error('Failed to process video');
      }

      const data = await response.json();
      setVideoInfo(data);
      setLoading(false); // Reset loading state when video info is ready

      // Then immediately start transcription
      setTranscribing(true);
      const transcriptResponse = await fetch('/start-transcription', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: videoUrl }),
      });

      if (!transcriptResponse.ok) {
        const errorText = await transcriptResponse.text();
        console.error('Transcription failed:', errorText);
        throw new Error(`Failed to start transcription: ${errorText}`);
      }

      const transcriptData = await transcriptResponse.json();
      console.log('Transcription started:', transcriptData);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setTranscribing(false);
    }
  };

  const handleRecallSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userRecall.trim() || !videoInfo) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/compare-recall', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_url: videoUrl,
          user_recall: userRecall,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Recall processing failed:', errorText);
        throw new Error(`Failed to process recall: ${errorText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setVideoUrl('');
    setVideoInfo(null);
    setShowRecall(false);
    setUserRecall('');
    setResults(null);
    setError('');
    setTranscribing(false);
  };

  const handleSaveRecall = async () => {
    if (!results || !videoInfo) return;

    try {
      const response = await fetch('/save-recall', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_url: videoUrl,
          user_recall: userRecall,
          transcript: results.transcript,
          comparison: results.comparison,
          score: results.score,
          video_title: videoInfo.title,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to save recall: ${errorText}`);
      }

      const data = await response.json();
      alert(`✅ Recall saved successfully!\nFile: ${data.filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save recall');
    }
  };

  const handleTryAgain = () => {
    setResults(null);
    setUserRecall('');
    setShowRecall(false);
  };

  return (
    <div className="container">
      <div className="card">
        <h1 style={{ textAlign: 'center', marginBottom: '30px', color: '#333' }}>
          🧠 Active Recall Learning
        </h1>
        
        {!videoInfo && (
          <form onSubmit={handleVideoSubmit}>
            <div style={{ marginBottom: '20px' }}>
              <label htmlFor="videoUrl" style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                Enter YouTube Video URL:
              </label>
              <input
                id="videoUrl"
                type="url"
                className="input"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=..."
                required
              />
            </div>
            <button type="submit" className="btn" disabled={loading}>
              {loading ? 'Processing...' : 'Load Video'}
            </button>
          </form>
        )}

        {error && <div className="error">{error}</div>}

        {loading && <div className="loading">Processing...</div>}

        {transcribing && <div className="loading">Transcribing video audio...</div>}

        {videoInfo && !showRecall && !results && (
          <div>
            <h2 style={{ marginBottom: '20px' }}>{videoInfo.title}</h2>
            <div className="video-container">
              <iframe
                src={videoInfo.embed_url}
                title={videoInfo.title}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
            <div style={{ textAlign: 'center', marginTop: '20px' }}>
              <button 
                className="btn" 
                onClick={() => setShowRecall(true)}
                style={{ fontSize: '18px', padding: '15px 30px' }}
              >
                🧠 Start Recall Exercise
              </button>
            </div>
          </div>
        )}

        {showRecall && !results && (
          <div>
            <h2 style={{ marginBottom: '20px' }}>What do you remember?</h2>
            <p style={{ marginBottom: '20px', color: '#666' }}>
              Type everything you can remember from the video. Be as detailed as possible!
            </p>
            <form onSubmit={handleRecallSubmit}>
              <textarea
                className="textarea"
                value={userRecall}
                onChange={(e) => setUserRecall(e.target.value)}
                placeholder="Start typing what you remember from the video..."
                required
              />
              <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                <button type="submit" className="btn" disabled={loading}>
                  {loading ? 'Analyzing...' : 'Submit Recall'}
                </button>
                <button type="button" className="btn" onClick={() => setShowRecall(false)} style={{ background: '#6c757d' }}>
                  Back to Video
                </button>
              </div>
            </form>
          </div>
        )}

        {results && (
          <div>
            <h2 style={{ marginBottom: '20px' }}>Recall Results</h2>
            
            <div className="score">
              Your Score: {results.score.toFixed(1)}%
            </div>

            <div className="results">
              <div className="results-section">
                <h4>✅ What you got right:</h4>
                <ul>
                  {results.comparison.correct.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="results-section">
                <h4>❌ What you got wrong or misunderstood:</h4>
                <ul>
                  {results.comparison.incorrect.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="results-section">
                <h4>🔍 What you missed:</h4>
                <ul>
                  {results.comparison.missed.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="results-section">
                <h4>💡 Feedback:</h4>
                <p>{results.comparison.feedback}</p>
              </div>
            </div>

            <div style={{ marginTop: '30px', textAlign: 'center', display: 'flex', gap: '15px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button 
                className="btn" 
                onClick={handleSaveRecall} 
                style={{ fontSize: '16px', background: '#28a745' }}
              >
                💾 Save Results
              </button>
              <button 
                className="btn" 
                onClick={handleTryAgain} 
                style={{ fontSize: '16px', background: '#ffc107', color: '#000' }}
              >
                🔄 Try Again
              </button>
              <button 
                className="btn" 
                onClick={resetApp} 
                style={{ fontSize: '16px', background: '#6c757d' }}
              >
                🆕 New Exercise
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 