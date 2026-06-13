import { useState } from "react";
import axios from "axios";
import ReactWordcloud from "react-wordcloud";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeVideo = async () => {
    if (!url) {
      alert("Please enter a YouTube URL");
      return;
    }

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/analyze",
        {
          youtube_url: url,
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Error analyzing video");
    } finally {
      setLoading(false);
    }
  };

  const pieData = result
    ? [
        {
          name: "Positive",
          value: result.positive_comments,
        },
        {
          name: "Negative",
          value: result.negative_comments,
        },
      ]
    : [];

  const barData = result
    ? [
        {
          sentiment: "Positive",
          count: result.positive_comments,
        },
        {
          sentiment: "Negative",
          count: result.negative_comments,
        },
      ]
    : [];

  const getWordCloudData = () => {
    if (!result?.comments) return [];

    const words = result.comments
      .join(" ")
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/);

    const stopWords = [
      "the",
      "is",
      "a",
      "an",
      "and",
      "to",
      "of",
      "in",
      "for",
      "this",
      "that",
      "it",
      "on",
      "with",
      "you",
      "your",
      "from",
      "have",
      "been",
      "was",
      "are",
      "they",
      "will",
      "their",
    ];

    const freq = {};

    words.forEach((word) => {
      if (
        word.length > 3 &&
        !stopWords.includes(word)
      ) {
        freq[word] = (freq[word] || 0) + 1;
      }
    });

    return Object.entries(freq)
      .map(([text, value]) => ({
        text,
        value,
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 100);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f6f8",
        padding: "30px",
        fontFamily: "Arial",
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "auto",
        }}
      >
        <h1
          style={{
            textAlign: "center",
            marginBottom: "30px",
          }}
        >
          🎬 YouTube Comment Analyzer
        </h1>

        <div
          style={{
            display: "flex",
            gap: "10px",
            marginBottom: "20px",
          }}
        >
          <input
            type="text"
            placeholder="Paste YouTube URL"
            value={url}
            onChange={(e) =>
              setUrl(e.target.value)
            }
            style={{
              flex: 1,
              padding: "14px",
              borderRadius: "10px",
              border: "1px solid #ccc",
            }}
          />

          <button
            onClick={analyzeVideo}
            style={{
              padding: "14px 25px",
              borderRadius: "10px",
              border: "none",
              cursor: "pointer",
              background: "#2563eb",
              color: "white",
            }}
          >
            Analyze
          </button>
        </div>

        {loading && (
          <h2>Analyzing Comments...</h2>
        )}

        {result && (
          <>
            <div
              style={{
                display: "flex",
                gap: "20px",
                flexWrap: "wrap",
                marginBottom: "20px",
              }}
            >
              <div
                style={{
                  flex: 1,
                  background: "white",
                  padding: "20px",
                  borderRadius: "12px",
                }}
              >
                <h3>Total Comments</h3>
                <h1>{result.total_comments}</h1>
              </div>

              <div
                style={{
                  flex: 1,
                  background: "#dcfce7",
                  padding: "20px",
                  borderRadius: "12px",
                }}
              >
                <h3>Positive</h3>
                <h1>{result.positive_comments}</h1>
              </div>

              <div
                style={{
                  flex: 1,
                  background: "#fee2e2",
                  padding: "20px",
                  borderRadius: "12px",
                }}
              >
                <h3>Negative</h3>
                <h1>{result.negative_comments}</h1>
              </div>
            </div>

            <div
              style={{
                display: "flex",
                gap: "20px",
                flexWrap: "wrap",
              }}
            >
              <div
                style={{
                  background: "white",
                  padding: "20px",
                  borderRadius: "12px",
                }}
              >
                <h2>🥧 Pie Chart</h2>

                <PieChart width={450} height={300}>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    outerRadius={100}
                    label
                  >
                    <Cell fill="#22c55e" />
                    <Cell fill="#ef4444" />
                  </Pie>

                  <Tooltip />
                  <Legend />
                </PieChart>
              </div>

              <div
                style={{
                  background: "white",
                  padding: "20px",
                  borderRadius: "12px",
                }}
              >
                <h2>📊 Bar Chart</h2>

                <BarChart
                  width={450}
                  height={300}
                  data={barData}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sentiment" />
                  <YAxis />
                  <Tooltip />

                  <Bar
                    dataKey="count"
                    fill="#3b82f6"
                  />
                </BarChart>
              </div>
            </div>
<div
  style={{
    background: "white",
    padding: "20px",
    borderRadius: "12px",
    marginTop: "20px",
  }}
>
  <h2>☁️ Word Cloud</h2>

  <img
    src={result.wordcloud_url}
    alt="Word Cloud"
    style={{
      width: "100%",
      maxWidth: "900px",
      borderRadius: "10px",
    }}
  />
</div>

          </>
        )}
      </div>
    </div>
  );
}

export default App;