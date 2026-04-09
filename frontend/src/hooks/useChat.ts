import { useState, useRef, useEffect } from "react";
import axios from "axios";
import type {
  Message,
  QueryResponse,
  QueryError,
  UploadDocumentsResponse,
} from "../types/chat";
import {
  API_BASE_URL,
  AUTH_TOKEN_KEY,
  SESSION_STORAGE_KEY,
  WELCOME_TEXT,
  MAX_FILE_UPLOAD_MB,
  MAX_TOTAL_UPLOAD_MB,
} from "../constants/chat";
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
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
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

  const uploadDocuments = async (files: File[]) => {
    if (!files.length || uploadLoading) return;

    setUploadStatus(null);
    const maxFileBytes = MAX_FILE_UPLOAD_MB * 1024 * 1024;
    const maxTotalBytes = MAX_TOTAL_UPLOAD_MB * 1024 * 1024;
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);

    const oversizedFile = files.find((file) => file.size > maxFileBytes);
    if (oversizedFile) {
      const message = `Folder is too large. ${oversizedFile.name} exceeds the ${MAX_FILE_UPLOAD_MB} MB per-file limit.`;
      window.alert(message);
      setUploadStatus(message);
      return;
    }
    if (totalBytes > maxTotalBytes) {
      const message = `Folder is too large. Max upload per request is ${MAX_TOTAL_UPLOAD_MB} MB.`;
      window.alert(message);
      setUploadStatus(message);
      return;
    }

    const formData = new FormData();
    for (const file of files) {
      const relativePath =
        ((file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name).replace(/\\/g, "/");
      formData.append("files", file, relativePath);
    }

    setUploadLoading(true);
    try {
      const token = localStorage.getItem(AUTH_TOKEN_KEY);
      const { data } = await axios.post<UploadDocumentsResponse>(
        `${API_BASE_URL}/upload-documents`,
        formData,
        {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        }
      );

      const status = [
        `${data.files_processed}/${data.files_received} files processed`,
        data.skipped_files.length ? `${data.skipped_files.length} skipped` : "",
        data.failed_files.length ? `${data.failed_files.length} failed` : "",
      ]
        .filter(Boolean)
        .join(" | ");

      setUploadStatus(status);
    } catch (err) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401 && onUnauthorized) {
        onUnauthorized();
        return;
      }
      if (status === 413) {
        const message = `Folder is too large. Max upload per request is ${MAX_TOTAL_UPLOAD_MB} MB.`;
        window.alert(message);
        setUploadStatus(message);
        return;
      }
      setUploadStatus(`Upload failed: ${parseError(err)}`);
    } finally {
      setUploadLoading(false);
    }
  };

  return {
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
  };
}
