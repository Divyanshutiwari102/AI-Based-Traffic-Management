import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

const POLLING_INTERVAL_MS = 1500;

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState(null);

  const handleFileChange = (e) => {
    // Convert FileList to array and set to state
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleSubmit = async (e) => {
    setLoading(true);
    e.preventDefault();
    // Ensure exactly 4 files are selected
    if (selectedFiles.length !== 4) {
      alert('Please upload exactly 4 videos.');
      return;
    }

    const formData = new FormData();
    // Append all selected files to FormData
    selectedFiles.forEach(file => formData.append('videos', file));

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setJobId(response.data.job_id);
    } catch (error) {
      console.error('Error uploading files:', error);
      setLoading(false);
    }
  };

  React.useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:5000/status/${jobId}`);
        const { state, result: payload, error } = response.data;
        if (state === 'SUCCESS') {
          setResult(payload.timings || payload);
          setLoading(false);
          setJobId(null);
          clearInterval(interval);
        } else if (state === 'FAILURE') {
          setResult({ error: error || 'Processing failed' });
          setLoading(false);
          setJobId(null);
          clearInterval(interval);
        }
      } catch (error) {
        const message = `Unable to fetch job status${error?.message ? `: ${error.message}` : ''}`;
        setResult({ error: message });
        setLoading(false);
        setJobId(null);
        clearInterval(interval);
      }
    }, POLLING_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [jobId]);

  return (
    <div className="App">
      <h1>🚗 AI Based Traffic Management</h1>
      <hr/>

      <div className='main-container'>
        <div className='left'>
          <section id="hero" className="hero">
            <h2>🚦 Optimize Traffic Flow with AI 🤖</h2>
            <p>Enhance your city's traffic management with our smart adaptive system. Our technology optimizes traffic light timings based on real-time data to reduce congestion and improve traffic flow.</p>
          </section>
          <section id="upload" className="upload">
            <h2>📹 Upload Your Traffic Videos</h2>
            <p>Select 4 videos showing different roads at an intersection. Our system will analyze these videos to provide optimized traffic light timings for smoother traffic flow.</p>
            <form onSubmit={handleSubmit}>
              <input 
                type="file" 
                multiple 
                accept="video/*" 
                onChange={handleFileChange} 
              />
              <br/>
              <button type="submit">Run Model</button>
            </form>
          </section>
        </div>

        <section id="result" className="result">
          {!loading && !result && (
            <p className='placeholder'>Optimization results will show here <br/><span>🚦🚦🚦🚦</span></p>
          )}
          {loading && <p className='loader'>Processing videos, it may take a few minutes...</p>}
          {result && !result.error && (
            <>
              <h2>✅ Optimization Results</h2>
              <p>Your traffic light timings have been optimized. Here are the recommended green times for each direction:</p>
              <ul>
                <li>🚦 North: <span id="north-time">{result.north}</span> seconds</li>
                <li>🚦 South: <span id="south-time">{result.south}</span> seconds</li>
                <li>🚦 West: <span id="west-time">{result.west}</span> seconds</li>
                <li>🚦 East: <span id="east-time">{result.east}</span> seconds</li>
              </ul>
            </>
          )}
        </section>
        {result && result.error && (
          <div>
            <h2>Error:</h2>
            <p>{result.error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
