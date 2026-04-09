export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export interface QueryResponse {
  answer?: string;
  sources?: string[];
}

export interface UploadDocumentsResponse {
  message: string;
  files_received: number;
  files_processed: number;
  chunks_processed: number;
  skipped_files: string[];
  failed_files: string[];
}

export interface QueryError {
  response?: {
    data?: {
      detail?: string | Array<{ msg?: string }>;
    };
  };
  message?: string;
}
