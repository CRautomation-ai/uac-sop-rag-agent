import { useState, useRef, useEffect } from "react";
import axios from "axios";
import type { Message, QueryResponse, QueryError } from "../types/chat";
import { API_BASE_URL, AUTH_TOKEN_KEY, SESSION_STORAGE_KEY, WELCOME_TEXT } from "../constants/chat";
import { getLast3Pairs } from "../utils/chat";

function parseError(err: unknown): string {
  const error = err as QueryError;
  const raw =
    error.response?.data?.detail ?? error.message ?? "An error occurred";
  return Array.isArray(raw)
    ? raw
        .map((e) => (typeof e === "string" ? e : e.msg || String(e)))
        .join(" ")
    : String(raw);
}

function loadMessagesFromStorage(): Message[] {
  try {
    const s = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (s) {
      const parsed = JSON.parse(s) as Message[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (_) {}
  return [{ role: "assistant", content: WELCOME_TEXT }];
}

export function useChat(onUnauthorized?: () => void) {
  const [messages, setMessages] = useState<Message[]>(loadMessagesFromStorage);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(messages));
    } catch (_) {}
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);
    const previousMessages = getLast3Pairs(messages);
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      const { data } = await axios.post<QueryResponse>(
        `${API_BASE_URL}/query`,
        {
          query: userMessage,
          top_k: 5,
          previous_messages: previousMessages,
        },
        {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
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
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401 && onUnauthorized) {
        onUnauthorized();
        return;
      }
      const msg = parseError(err);
      setError(msg);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return {
    messages,
    input,
    setInput,
    loading,
    error,
    messagesEndRef,
    handleSubmit,
  };
}
