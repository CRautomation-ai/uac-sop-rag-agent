import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

// Backend base URL: /api when using Vite proxy (dev), or set VITE_API_URL for production (e.g. backend URL)
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setError(null);
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const { data } = await axios.post(`${API_BASE_URL}/query`, {
        query: userMessage,
        top_k: 5
      });
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer ?? '', sources: Array.isArray(data.sources) ? data.sources : [] }
      ]);
    } catch (err) {
      const raw = err.response?.data?.detail ?? err.message ?? 'An error occurred';
      const msg = Array.isArray(raw) ? raw.map((e) => e.msg || e).join(' ') : raw;
      setError(msg);
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>SOP RAG</h1>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="messages-container">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-bubble">{msg.content}</div>
            {msg.sources?.length > 0 && (
              <div className="message-sources">Sources: {msg.sources.join(', ')}</div>
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
          <button type="submit" disabled={loading || !input.trim()} className="send-button">
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
