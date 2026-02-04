import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

// Backend base URL: /api when using Vite proxy (dev), or set VITE_API_URL for production (e.g. backend URL)
const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

interface QueryResponse {
  answer?: string;
  sources?: string[];
}

interface QueryError {
  response?: {
    data?: {
      detail?: string | Array<{ msg?: string }>;
    };
  };
  message?: string;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const { data } = await axios.post<QueryResponse>(
        `${API_BASE_URL}/query`,
        {
          query: userMessage,
          top_k: 5,
        }
      );
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer ?? "",
          sources: Array.isArray(data.sources) ? data.sources : [],
        },
      ]);
    } catch (err) {
      const error = err as QueryError;
      const raw =
        error.response?.data?.detail ?? error.message ?? "An error occurred";
      const msg = Array.isArray(raw)
        ? raw
            .map((e) => (typeof e === "string" ? e : e.msg || String(e)))
            .join(" ")
        : String(raw);
      setError(msg);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>SOPRAG</h1>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="messages-container">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-bubble">{msg.content}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="message-sources">
                Sources: {msg.sources.join(", ")}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-bubble">...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="input-field"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="send-button"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
