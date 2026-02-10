import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../types/chat";

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => (
  <div className={`message ${message.role}`}>
    <div className="message-bubble">
      {message.role === "assistant" ? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>
      ) : (
        message.content
      )}
    </div>
    {message.sources && message.sources.length > 0 && (
      <div className="message-sources">
        Sources: {message.sources.join(", ")}
      </div>
    )}
  </div>
);
