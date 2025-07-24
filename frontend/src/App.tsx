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

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const autoGrow = (el: HTMLTextAreaElement) => {
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  };


  const handleVideoSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoUrl.trim()) return;

    setLoading(true);
    setError('');
    setVideoInfo(null);
    setShowRecall(false);
    setResults(null);

    try {
      const res = await fetch(`${API_BASE_URL}/process-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: videoUrl }),
      });
      if (!res.ok) throw new Error('Failed to process video');
      const data = await res.json();
      setVideoInfo(data);
      setLoading(false);

      /* auto‑kick transcription */
      setTranscribing(true);
      const tRes = await fetch(`${API_BASE_URL}/start-transcription`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: videoUrl }),
      });
      if (!tRes.ok) throw new Error(await tRes.text());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error');
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
      const res = await fetch(`${API_BASE_URL}/compare-recall`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: videoUrl, user_recall: userRecall }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResults(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error');
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
      const res = await fetch(`${API_BASE_URL}/save-recall`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_url: videoUrl,
          user_recall: userRecall,
          transcript: results.transcript,
          comparison: results.comparison,
          score: results.score,
          video_title: videoInfo.title,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      alert(`Recall saved!\nFile: ${data.filename}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed');
    }
  };

  // ui

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#fff',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: 20,
      }}
    >
      {/* url form*/}
      {!videoInfo && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          width: '100%',
          height: '100vh'
        }}>
          <form
            onSubmit={handleVideoSubmit}
            style={{ width: '100%', maxWidth: 500, textAlign: 'center' }}
          >
            <label style={{ display: 'block', marginBottom: 15, fontSize: 18 }}>
              Enter YouTube URL:
            </label>

            <input
              type="url"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              required
              style={{
                width: '100%',
                padding: 15,
                fontSize: 16,
                border: '2px solid #ddd',
                borderRadius: 8,
                marginBottom: 20,
              }}
            />

            <button
              type="submit"
              disabled={loading}
              style={{
                background: '#000',
                color: '#fff',
                border: 'none',
                padding: '15px 30px',
                borderRadius: 8,
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1,
                fontSize: 16,
                fontWeight: 500,
              }}
            >
              {loading ? 'Processing…' : 'Load Video'}
            </button>
          </form>
        </div>
      )}

      {videoInfo && !showRecall && !results && (
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center', 
          alignItems: 'center', 
          width: '100%', 
          maxWidth: 800,
          height: '100vh'
        }}>
          <iframe
            src={videoInfo.embed_url}
            title={videoInfo.title}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            style={{ width: '100%', height: 400, marginBottom: 30 }}
          />
          <button
            onClick={() => setShowRecall(true)}
            style={{
              background: '#000',
              color: '#fff',
              padding: '15px 30px',
              border: 'none',
              borderRadius: 8,
              cursor: 'pointer',
              fontSize: 16,
              fontWeight: 500,
            }}
          >
            Recall
          </button>
        </div>
      )}

      {showRecall && !results && (
        <section
          style={{
            width: '100%',
            maxWidth: '8.5in',
            margin: '0 auto',
            padding: '80px 40px 120px',
            boxSizing: 'border-box',
          }}
          onClick={() => (document.getElementById('recall-textarea') as HTMLTextAreaElement)?.focus()}
        >
          <textarea
            id="recall-textarea"
            value={userRecall}
            onChange={(e) => setUserRecall(e.target.value)}
            onInput={(e) => autoGrow(e.currentTarget)}
            placeholder="Type everything you recall…"
            autoFocus
            style={{
              width: '100%',
              border: 'none',
              outline: 'none',
              resize: 'none',
              overflow: 'hidden', // ✨ no inner scroll
              fontSize: 16,
              lineHeight: 1.6,
              fontFamily: 'Arial, sans-serif',
              background: 'transparent',
              whiteSpace: 'pre-wrap',
            }}
          />
        </section>
      )}

      {showRecall && !results && (
        <div
          style={{
            position: 'fixed',
            bottom: 60,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(255,255,255,0.6)',
            backdropFilter: 'blur(10px)',
            borderRadius: 15,
            padding: '15px 25px',
            display: 'flex',
            gap: 10,
            zIndex: 1000,
          }}
        >
          <button
            onClick={handleRecallSubmit}
            disabled={loading}
            style={{
              background: '#000',
              color: '#fff',
              border: 'none',
              padding: '12px 20px',
              borderRadius: 15,
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? 'Analyzing…' : 'Submit'}
          </button>

          <button
            onClick={() => {
              setUserRecall('');
              setShowRecall(false);
            }}
            style={{
              background: '#fff',
              color: '#000',
              border: '2px solid #000',
              padding: '12px 20px',
              borderRadius: 15,
              cursor: 'pointer',
            }}
          >
            Try Again
          </button>

          <button
            onClick={handleSaveRecall}
            style={{
              background: '#fff',
              color: '#000',
              border: '2px solid #000',
              padding: '12px 20px',
              borderRadius: 15,
              cursor: 'pointer',
            }}
          >
            Save
          </button>
        </div>
      )}

      {results && (
        <div style={{ 
          width: '100%', 
          maxWidth: 800, 
          padding: '60px 40px 120px',
          lineHeight: 1.6,
          fontSize: 16
        }}>
          <h2 style={{ 
            fontSize: 32, 
            fontWeight: 600, 
            marginBottom: 20,
            color: '#1a1a1a'
          }}>
            Recall Results
          </h2>

          <p style={{ 
            fontSize: 18, 
            fontWeight: 500, 
            marginBottom: 25,
            color: '#333'
          }}>
            Your Score: {results.score.toFixed(1)}%
          </p>

          <div>
            <h4 style={{ 
              fontSize: 20, 
              fontWeight: 600, 
              marginBottom: 15,
              marginTop: 0,
              color: '#1a1a1a'
            }}>
              Correct
            </h4>
            <ul style={{ 
              marginLeft: 40,
              marginBottom: 25
            }}>
              {results.comparison.correct.map((c, i) => (
                <li key={i} style={{ marginBottom: 8 }}>{c}</li>
              ))}
            </ul>

            <h4 style={{ 
              fontSize: 20, 
              fontWeight: 600, 
              marginBottom: 15,
              marginTop: 30,
              color: '#1a1a1a'
            }}>
              Incorrect
            </h4>
            <ul style={{ 
              marginLeft: 40,
              marginBottom: 25
            }}>
              {results.comparison.incorrect.map((c, i) => (
                <li key={i} style={{ marginBottom: 8 }}>{c}</li>
              ))}
            </ul>

            <h4 style={{ 
              fontSize: 20, 
              fontWeight: 600, 
              marginBottom: 15,
              marginTop: 30,
              color: '#1a1a1a'
            }}>
              Missed
            </h4>
            <ul style={{ 
              marginLeft: 40,
              marginBottom: 25
            }}>
              {results.comparison.missed.map((c, i) => (
                <li key={i} style={{ marginBottom: 8 }}>{c}</li>
              ))}
            </ul>

            <h4 style={{ 
              fontSize: 20, 
              fontWeight: 600, 
              marginBottom: 15,
              marginTop: 30,
              color: '#1a1a1a'
            }}>
              Feedback
            </h4>
            <p style={{ 
              marginLeft: 0,
              marginBottom: 25,
              color: '#333'
            }}>
              {results.comparison.feedback}
            </p>
          </div>

          <div
            style={{
              position: 'fixed',
              bottom: 60,
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(255,255,255,0.6)',
              backdropFilter: 'blur(10px)',
              borderRadius: 15,
              padding: '15px 25px',
              display: 'flex',
              gap: 10,
              zIndex: 1000,
            }}
          >
            <button 
              onClick={handleSaveRecall} 
              style={{
                background: '#000',
                color: '#fff',
                border: 'none',
                padding: '12px 20px',
                borderRadius: 15,
                cursor: 'pointer',
                fontSize: 16,
                fontWeight: 500,
              }}
            >
              Save
            </button>
            <button 
              onClick={() => { setResults(null); setShowRecall(false); }} 
              style={{
                background: '#fff',
                color: '#000',
                border: '2px solid #000',
                padding: '12px 20px',
                borderRadius: 15,
                cursor: 'pointer',
                fontSize: 16,
                fontWeight: 500,
              }}
            >
              Try Again
            </button>
            <button 
              onClick={resetApp} 
              style={{
                background: '#fff',
                color: '#000',
                border: '2px solid #000',
                padding: '12px 20px',
                borderRadius: 15,
                cursor: 'pointer',
                fontSize: 16,
                fontWeight: 500,
              }}
            >
              New Video
            </button>
          </div>
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default App;
