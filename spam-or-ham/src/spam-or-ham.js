import React, { useState } from 'react';

const SpamDetector = () => {
  const [subject, setSubject] = useState('');
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    const baseurl = process.env.REACT_APP_API_URL;
    const api = '/predict'; 
    const url = `${baseurl}${api}`;

    const payload = {
      subject,
      message
    };
    console.log("Sending request to:", url);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      console.log("Response headers:", [...response.headers]);
      
      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData);
      } else {
        const data = await response.json();
        setResult(data);
      }
    } catch (err) {
      setError(err.toString());
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '2rem auto', fontFamily: 'Comic Neue, cursive' }}>
      <h1>Spam or Ham</h1>
      {/* Spam or ham logo */}
      <img 
        src="/spam-or-ham.png" 
        alt="Spam or Ham Detector" 
        style={{ display: 'block', margin: '0 auto 1rem', maxWidth: '100%' }}
      />
      <p>Enter a subject and message to determine if it is fake or real!</p>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="subject" style={{ display: 'block', marginBottom: '0.5rem' }}>
            Subject:
          </label>
          <input
            type="text"
            id="subject"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            required
            style={{ width: '100%', padding: '0.5rem' }}
          />
        </div>
        <div style={{ marginBottom: '1rem' }}>
          <label htmlFor="message" style={{ display: 'block', marginBottom: '0.5rem' }}>
            Message:
          </label>
          <textarea
            id="message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            required
            rows="6"
            style={{ width: '100%', padding: '0.5rem' }}
          ></textarea>
        </div>
        <button type="submit" disabled={loading} style={{ padding: '0.5rem 1rem' }}>
          {loading ? 'Analyzing...' : 'Submit'}
        </button>
      </form>
      {error && (
        <div style={{ marginTop: '1rem', color: 'red' }}>
          <strong>Error:</strong> {JSON.stringify(error)}
        </div>
      )}
      {result && (
      <div className={`result-box ${result.prediction === "spam" ? "spam" : "ham"}`}>
        <h2>Analysis Result</h2>
        <p>
          <strong>Subject:</strong> {subject}
        </p>
        <p>
          <strong>Message:</strong> {message}
        </p>
        <p>
          <strong>Verdict:</strong> {result.prediction.toUpperCase()}
        </p>
      </div>
    )}
    </div>
  );
};

export default SpamDetector;
