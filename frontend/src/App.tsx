import { useState } from "react";
import ChatInterface from "./components/ChatInterface";
import PasswordGate from "./components/PasswordGate";
import { AUTH_TOKEN_KEY } from "./constants/chat";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => !!localStorage.getItem(AUTH_TOKEN_KEY)
  );

  const handleUnauthorized = () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <PasswordGate onSuccess={() => setIsAuthenticated(true)} />;
  }
  return (
    <ChatInterface
      onUnauthorized={handleUnauthorized}
      onLogout={handleUnauthorized}
    />
  );
}

export default App;
