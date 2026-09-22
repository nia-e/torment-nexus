import type { ServerEvent, State } from "./types";

const TOKEN_KEY = "torment-nexus.launch-token";
export function readLaunchToken(
  location: Pick<Location, "hash" | "pathname" | "search">,
  storage: Pick<Storage, "getItem" | "setItem">,
  replace: (url: string) => void,
): string {
  const hash = location.hash.slice(1);
  const params = new URLSearchParams(hash);
  const token =
    params.get("token") ?? (hash && !hash.includes("=") ? hash : null);
  if (token) {
    storage.setItem(TOKEN_KEY, token);
    replace(location.pathname + location.search);
    return token;
  }
  return storage.getItem(TOKEN_KEY) ?? "";
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
export class Api {
  constructor(private token: string) {}
  private headers() {
    return {
      Authorization: `Bearer ${this.token}`,
      "Content-Type": "application/json",
    };
  }
  private async response<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(path, {
      ...options,
      headers: this.headers(),
      credentials: "omit",
      cache: "no-store",
    });
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new ApiError(
        `Server returned ${response.status} without a JSON response. Check the launcher logs.`,
        response.status,
      );
    }
    if (!response.ok) {
      const error =
        typeof body === "object" && body !== null && "error" in body
          ? String(body.error)
          : `Request failed (${response.status}).`;
      throw new ApiError(error, response.status);
    }
    return body as T;
  }
  state(signal?: AbortSignal): Promise<State> {
    return this.response("/api/state", { signal });
  }
  action<T = Record<string, unknown>>(
    action: string,
    fields: Record<string, unknown> = {},
  ): Promise<T> {
    return this.response("/api/action", {
      method: "POST",
      body: JSON.stringify({ action, ...fields }),
    });
  }
  async events(
    onEvent: (event: ServerEvent) => void,
    onConnect: () => void,
    signal: AbortSignal,
  ): Promise<void> {
    const response = await fetch("/api/events", {
      headers: this.headers(),
      credentials: "omit",
      cache: "no-store",
      signal,
    });
    if (!response.ok)
      throw new ApiError(
        `Event stream unavailable (${response.status}).`,
        response.status,
      );
    if (!response.body)
      throw new Error("This browser does not support response streaming.");
    onConnect();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const parser = new SseParser(onEvent);
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          parser.push(decoder.decode());
          break;
        }
        parser.push(decoder.decode(value, { stream: true }));
      }
    } finally {
      reader.releaseLock();
    }
  }
}

/** Handles chunk boundaries, CRLF and comments; never puts the launch token in a URL. */
export class SseParser {
  private buffer = "";
  constructor(private onEvent: (event: ServerEvent) => void) {}
  push(chunk: string): void {
    this.buffer += chunk;
    let match: RegExpExecArray | null;
    while ((match = /\r?\n\r?\n/.exec(this.buffer))) {
      const frame = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const payload = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (!payload) continue;
      let event: ServerEvent;
      try {
        event = JSON.parse(payload) as ServerEvent;
      } catch {
        continue;
      }
      if (
        typeof event === "object" &&
        event !== null &&
        typeof event.kind === "string"
      )
        this.onEvent({ ...event, data: event.data ?? {} });
    }
    if (this.buffer.length > 4 * 1024 * 1024)
      throw new Error(
        "Oversized event frame. Reconnecting from a saved snapshot.",
      );
  }
}
