import React from "react";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  loading: boolean;
  placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  loading,
  placeholder = "Ask Bolt...",
}) => (
  <div className="input-container">
    <form onSubmit={onSubmit} className="input-form">
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="input-field"
        disabled={loading}
      />
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="send-button"
      >
        Send
      </button>
    </form>
  </div>
);
