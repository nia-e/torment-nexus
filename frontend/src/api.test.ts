import { describe, expect, it, vi } from "vitest";
import { Api, readLaunchToken, SseParser } from "./api";

describe("launch token", () => {
  it("removes launch credentials from URL and retains them only in session storage", () => {
    const storage = { setItem: vi.fn(), getItem: vi.fn() };
    const replace = vi.fn();
    expect(
      readLaunchToken(
        { hash: "#token=private-session", pathname: "/", search: "" },
        storage,
        replace,
      ),
    ).toBe("private-session");
    expect(storage.setItem).toHaveBeenCalledWith(
      "torment-nexus.launch-token",
      "private-session",
    );
    expect(replace).toHaveBeenCalledExactlyOnceWith("/");
  });
  it("rehydrates token after refresh without putting it in a query string", async () => {
    const storage = {
      setItem: vi.fn(),
      getItem: vi.fn().mockReturnValue("retained"),
    };
    const replace = vi.fn();
    const token = readLaunchToken(
      { hash: "", pathname: "/", search: "" },
      storage,
      replace,
    );
    const fetch = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({ models: [] }) });
    vi.stubGlobal("fetch", fetch);
    await new Api(token).state();
    expect(fetch.mock.calls[0][0]).toBe("/api/state");
    expect(fetch.mock.calls[0][1].headers.Authorization).toBe(
      "Bearer retained",
    );
    expect(replace).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
describe("SSE framing", () => {
  it("handles split unicode-safe decoded chunks, CRLF and comments", () => {
    const receive = vi.fn();
    const parser = new SseParser(receive);
    parser.push(': keepalive\r\n\r\ndata: {"kind":"token","data":{"text":"hé');
    parser.push('llo"}}\r');
    parser.push("\n\r\n");
    expect(receive).toHaveBeenCalledExactlyOnceWith({
      kind: "token",
      data: { text: "héllo" },
    });
  });
  it("accepts multiple and multiline events and ignores malformed frames", () => {
    const receive = vi.fn();
    const parser = new SseParser(receive);
    parser.push(
      'data: null\n\ndata: nonsense\n\ndata: {"kind":"state",\ndata: "data":{}}\n\ndata: {"kind":"completed","data":{}}\n\n',
    );
    expect(receive.mock.calls.map((call) => call[0].kind)).toEqual([
      "state",
      "completed",
    ]);
  });
});
