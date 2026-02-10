import React, { useState } from "react";
import axios from "axios";
import { API_BASE_URL, AUTH_TOKEN_KEY } from "../constants/chat";
import Spinner from "./Spinner";

interface PasswordGateProps {
  onSuccess: () => void;
}

export const PasswordGate: React.FC<PasswordGateProps> = ({ onSuccess }) => {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!password.trim() || loading) return;
    setError(null);
    setLoading(true);
    try {
      const { data } = await axios.post<{ token: string }>(`${API_BASE_URL}/auth`, {
        password: password.trim(),
      });
      localStorage.setItem(AUTH_TOKEN_KEY, data.token);
      onSuccess();
    } catch (err) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      setError(status === 401 ? "Invalid password" : "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Sign in</h1>
        <p>Enter the company password to access the SOP RAG agent.</p>
      </div>
      <div className="password-gate">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="input-field"
            disabled={loading}
            autoFocus
          />
          <button
            type="submit"
            disabled={loading || !password.trim()}
            className="send-button"
          >
            {loading ? <Spinner /> : "Continue"}
          </button>
        </form>
        {error && <div className="error-message">{error}</div>}
      </div>
    </div>
  );
};

export default PasswordGate;
