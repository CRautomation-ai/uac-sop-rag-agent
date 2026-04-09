import React, { useRef } from "react";
import logo from "../assets/logo.jpg";
import { MAX_FILE_UPLOAD_MB, MAX_TOTAL_UPLOAD_MB, WELCOME_TEXT } from "../constants/chat";
import { useChat } from "../hooks/useChat";
import { MessageBubble } from "./MessageBubble";
import { WelcomeView } from "./WelcomeView";
import { ChatInput } from "./ChatInput";
import { LoadingDots } from "./LoadingDots";

interface ChatInterfaceProps {
  onUnauthorized?: () => void;
  onLogout?: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onUnauthorized, onLogout }) => {
  const {
    messages,
    input,
    setInput,
    loading,
    error,
    uploadLoading,
    uploadStatus,
    messagesEndRef,
    handleSubmit,
    uploadDocuments,
  } = useChat(onUnauthorized);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const isWelcomeOnly =
    messages.length === 1 &&
    messages[0].role === "assistant" &&
    messages[0].content === WELCOME_TEXT;

  return (
    <div className="app-container">
      <div className="header">
        <img src={logo} alt="SOP RAG" className="header-logo" />
        <div className="upload-controls">
          <div className="header-actions">
            <button
              type="button"
              className="header-button"
              disabled={uploadLoading}
              onClick={() => fileInputRef.current?.click()}
            >
              {uploadLoading ? "Uploading..." : "Upload Files"}
            </button>
            <button
              type="button"
              className="header-button"
              disabled={uploadLoading}
              onClick={() => folderInputRef.current?.click()}
            >
              {uploadLoading ? "Uploading..." : "Upload Folder"}
            </button>
          </div>
          <div className="upload-limit-warning">
            Max upload: {MAX_TOTAL_UPLOAD_MB} MB per upload, {MAX_FILE_UPLOAD_MB} MB per file.
          </div>
        </div>
        {onLogout && (
          <button type="button" className="logout-button" onClick={onLogout}>
            Log out
          </button>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.doc,.docx"
          multiple
          className="hidden-file-input"
          onChange={(e) => {
            const selected = Array.from(e.target.files || []);
            void uploadDocuments(selected);
            e.target.value = "";
          }}
        />
        <input
          ref={folderInputRef}
          type="file"
          accept=".pdf,.doc,.docx"
          multiple
          className="hidden-file-input"
          {...({
            webkitdirectory: "",
            directory: "",
          } as React.InputHTMLAttributes<HTMLInputElement>)}
          onChange={(e) => {
            const selected = Array.from(e.target.files || []);
            void uploadDocuments(selected);
            e.target.value = "";
          }}
        />
      </div>

      {error && <div className="error-message">{error}</div>}
      {uploadStatus && <div className="upload-status">{uploadStatus}</div>}

      <div className="messages-container">
        {isWelcomeOnly ? (
          <WelcomeView />
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
          </>
        )}
        {loading && <LoadingDots />}
        <div ref={messagesEndRef} />
      </div>

      <ChatInput
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        loading={loading}
      />
    </div>
  );
};

export default ChatInterface;
