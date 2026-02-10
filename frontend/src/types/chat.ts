export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export interface QueryResponse {
  answer?: string;
  sources?: string[];
}

export interface QueryError {
  response?: {
    data?: {
      detail?: string | Array<{ msg?: string }>;
    };
  };
  message?: string;
}
