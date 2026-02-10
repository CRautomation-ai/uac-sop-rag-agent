import type { Message } from "../types/chat";

export function getLast3Pairs(
  msgs: Message[]
): { query: string; answer: string }[] {
  const pairs: { query: string; answer: string }[] = [];
  for (let i = 0; i < msgs.length - 1; i += 1) {
    if (msgs[i].role === "user" && msgs[i + 1].role === "assistant") {
      pairs.push({ query: msgs[i].content, answer: msgs[i + 1].content });
    }
  }
  return pairs.slice(-3);
}
