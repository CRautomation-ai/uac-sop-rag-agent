import React from "react";
import { WELCOME_TEXT } from "../constants/chat";

export const WelcomeView: React.FC = () => (
  <div className="welcome-center">
    <p className="welcome-text">{WELCOME_TEXT}</p>
  </div>
);
