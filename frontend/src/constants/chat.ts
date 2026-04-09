export const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";
export const AUTH_TOKEN_KEY = "sop_rag_token";
export const SESSION_STORAGE_KEY = "chat_messages";
export const MAX_TOTAL_UPLOAD_MB = Number(import.meta.env.VITE_MAX_TOTAL_UPLOAD_MB || 4);
export const MAX_FILE_UPLOAD_MB = Number(import.meta.env.VITE_MAX_FILE_UPLOAD_MB || 4);
export const WELCOME_TEXT =
  "Hi I'm Bolt, I'm here to help you find SOPs, processes and answers for UAC. Let's go!";
