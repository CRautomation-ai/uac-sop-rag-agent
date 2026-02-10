import React from "react";

export const LoadingDots: React.FC = () => (
  <div className="message assistant">
    <div className="message-bubble loading-bubble">
      <span className="loading-dots">
        <span className="loading-dot" />
        <span className="loading-dot" />
        <span className="loading-dot" />
      </span>
    </div>
  </div>
);
