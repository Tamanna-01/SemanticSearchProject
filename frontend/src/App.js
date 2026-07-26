import React, { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
        }),
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error(error);
      alert("Unable to connect to the backend server.");
    }

    setLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="app">
      <div className="hero">
        <div className="hero-content">
          <h1>AI Semantic Search</h1>

          <div className="search-box">
            <input
              type="text"
              placeholder="Example: Can you use RStudio with Azure ML?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyPress}
            />

            <button onClick={handleSearch} disabled={loading}>
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        </div>
      </div>

      <div className="content">
        <div className="section-header">
          <div>
            <h2>Top 5 Results</h2>
            <span>Ranked by semantic similarity</span>
          </div>
        </div>

        {!loading && results.length === 0 && (
          <div className="empty-state">
            <h3>No Results Yet</h3>

            <p>
              Enter a question above to search across the AI Show transcript
              collection.
            </p>
          </div>
        )}

        {loading && (
          <div className="loading-card">
            <div className="loader"></div>

            <p>Searching transcript embeddings...</p>
          </div>
        )}

        {!loading &&
          results.map((res, index) => (
            <div className="result-card" key={index}>
              <div className="result-number">{index + 1}</div>

              <div className="result-body">
                <div className="result-header">
                  <div>
                    <h3>{res.title}</h3>

                    <p className="speaker">Speaker: {res.speaker}</p>
                  </div>

                  <div className="score">
                    <span className="score-label">Similarity Score</span>
                    <span className="score-value">
                      {res.similarity_score.toFixed(4)}
                    </span>
                  </div>
                </div>

                <p className="summary">Summary: {res.summary}</p>

                <div className="bottom-row">
                  <div className="timestamp">{res.timestamp}</div>

                  <a
                    href={res.youtube_link}
                    target="_blank"
                    rel="noreferrer"
                    className="youtube-button"
                  >
                    Watch on YouTube
                  </a>
                </div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

export default App;
