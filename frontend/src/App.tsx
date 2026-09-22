import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { Api, ApiError, readLaunchToken } from "./api";
import {
  expandLayerAxes,
  finitePercent,
  LiveControls,
  requestedAxes,
  signature,
} from "./controls";
import { TokenReconciler } from "./stream";
import {
  conversationHistory,
  conversationRuns,
  startupModelId,
} from "./workflows";
import {
  checkpointRefs,
  completedOutput,
  directRecipeStages,
  jobCheckpoints,
  stageLabels,
  stageNames,
  type StageName,
} from "./stages";
import {
  emptyState,
  recordTime,
  running,
  type Axis,
  type AppliedMixture,
  type CodexModel,
  type HfListing,
  type Job,
  type Json,
  type Message,
  type Model,
  type Recipe,
  type Roles,
  type Run,
  type Sampling,
  type ServerEvent,
  type State,
  type Vector,
} from "./types";

type Act = <T = Record<string, unknown>>(
  action: string,
  fields?: Record<string, unknown>,
  message?: string,
) => Promise<T | null>;
type Dialog = "models" | "concept" | "history" | "import" | "mix" | null;
type ManagedKind = "vectors" | "recipes" | "presets" | "conversations";
type Management = {
  kind: ManagedKind;
  id: string;
  name: string;
  operation: "rename" | "delete";
};
const recordLabels: Record<ManagedKind, string> = {
  vectors: "vector",
  recipes: "recipe",
  presets: "saved mix",
  conversations: "conversation",
};
type Notice = { id: number; message: string; kind: "error" | "success" };
type Workspace = {
  modelId: string;
  axes: Axis[];
  prompt: string;
  mode: "scratchpad" | "chat";
  conversationId: string;
  raw: boolean;
  sampling: Sampling;
  system: string;
  prefixMessages: Message[];
  baselineRestore: Axis[] | null;
  selfModification: boolean;
};
const defaultSampling: Sampling = {
  seed: 42,
  temperature: 0.7,
  top_p: 0.95,
  max_tokens: 256,
};
const defaultWorkspace: Workspace = {
  modelId: "",
  axes: [],
  prompt: "",
  mode: "scratchpad",
  conversationId: "",
  raw: false,
  sampling: defaultSampling,
  system: "",
  prefixMessages: [],
  baselineRestore: null,
  selfModification: false,
};
const activeStatuses = new Set(["queued", "running"]);
function loadWorkspace(): Workspace {
  try {
    const saved = JSON.parse(
      localStorage.getItem("torment-nexus.workspace") ?? "{}",
    );
    return {
      ...defaultWorkspace,
      ...saved,
      sampling: { ...defaultSampling, ...saved.sampling },
    };
  } catch {
    return defaultWorkspace;
  }
}
function formatSize(size?: number): string {
  if (size === undefined || !Number.isFinite(size)) return "size unknown";
  return size > 1e9
    ? `${(size / 1e9).toFixed(1)} GB`
    : `${Math.max(1, size / 1e6).toFixed(0)} MB`;
}
function shortId(id?: string) {
  return id ? id.slice(0, 10) : "—";
}
function dateLabel(seconds?: number) {
  return seconds
    ? new Date(seconds * 1000).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "just now";
}
function labelModel(model: CodexModel) {
  return model.displayName ?? model.display_name ?? model.id;
}
function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}
function jsonText(value: unknown) {
  return JSON.stringify(value ?? null, null, 2);
}
function downloadJson(filename: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([jsonText(value)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    plus: <path d="M12 5v14M5 12h14" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    send: (
      <>
        <path d="m4 4 17 8-17 8 3-8-3-8Z" />
        <path d="M7 12h14" />
      </>
    ),
    stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
    sliders: (
      <>
        <path d="M4 7h16M4 17h16" />
        <circle cx="8" cy="7" r="3" />
        <circle cx="16" cy="17" r="3" />
      </>
    ),
    archive: (
      <>
        <rect x="4" y="7" width="16" height="13" rx="2" />
        <path d="M3 4h18v4H3zM9 12h6" />
      </>
    ),
    history: (
      <>
        <path d="M3 11a9 9 0 1 1 2 7M3 4v7h7" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    chevron: <path d="m8 5 7 7-7 7" />,
    download: (
      <>
        <path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    code: (
      <>
        <path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="12" height="12" rx="2" />
        <path d="M16 8V4H4v12h4" />
      </>
    ),
    reset: (
      <>
        <path d="M4 10a8 8 0 1 1 1 8M4 3v7h7" />
      </>
    ),
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 5 5" />
      </>
    ),
    bolt: <path d="m13 2-9 12h7l-1 8 10-13h-7l1-7Z" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] ?? paths.code}
    </svg>
  );
}
function NexusMark({ large = false }: { large?: boolean }) {
  return (
    <div className={`nexus-mark ${large ? "large" : ""}`} aria-hidden="true">
      <span />
      <span />
      <i />
    </div>
  );
}
function Badge({
  children,
  tone = "",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-symbol">⊹</div>
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}
function Modal({
  title,
  eyebrow,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const node = dialog.current;
    node?.showModal();
    node
      ?.querySelector<HTMLElement>(
        '[autofocus], textarea, input:not([type="file"]), select',
      )
      ?.focus();
    return () => {
      node?.close();
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className={`modal ${wide ? "wide" : ""}`}
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === dialog.current) onClose();
      }}
      aria-label={title}
    >
      <header className="modal-header">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2 title={title}>{title}</h2>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close dialog"
        >
          <Icon name="close" />
        </button>
      </header>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}

export default function App() {
  const [token] = useState(() =>
    readLaunchToken(window.location, sessionStorage, (url) =>
      history.replaceState(null, "", url),
    ),
  );
  const api = useMemo(() => new Api(token), [token]);
  const [state, setState] = useState<State>(emptyState);
  const [workspace, setWorkspace] = useState<Workspace>(loadWorkspace);
  const [connection, setConnection] = useState<
    "connecting" | "live" | "recovering" | "offline" | "unauthorized"
  >("connecting");
  const [hasSnapshot, setHasSnapshot] = useState(false);
  const [notices, setNotices] = useState<Notice[]>([]);
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [dialog, setDialog] = useState<Dialog>(null);
  const [forkJobId, setForkJobId] = useState<string | null>(null);
  const [inspect, setInspect] = useState<{
    kind: "vector" | "recipe" | "run";
    id: string;
  } | null>(null);
  const [libraryTab, setLibraryTab] = useState<"vectors" | "recipes">(
    "vectors",
  );
  const [filter, setFilter] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(
    () => localStorage.getItem("torment-nexus.library-open") !== "false",
  );
  const [mixerOpen, setMixerOpen] = useState(
    () => localStorage.getItem("torment-nexus.mixer-open") !== "false",
  );
  const focusChat = !libraryOpen && !mixerOpen;
  useEffect(() => {
    localStorage.setItem("torment-nexus.library-open", String(libraryOpen));
    localStorage.setItem("torment-nexus.mixer-open", String(mixerOpen));
  }, [libraryOpen, mixerOpen]);
  const [management, setManagement] = useState<Management | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [focusedRunId, setFocusedRunId] = useState(
    () => localStorage.getItem("torment-nexus.focused-run") ?? "",
  );
  useEffect(() => {
    localStorage.setItem("torment-nexus.focused-run", focusedRunId);
  }, [focusedRunId]);
  const [showJobs, setShowJobs] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pendingRevision, setPendingRevision] = useState<number | null>(null);
  const [ack, setAck] = useState<{
    revision: number;
    firstToken: number;
  } | null>(null);
  const [streamingText, setStreamingText] = useState<Record<string, string>>(
    {},
  );
  const live = useRef<LiveControls | null>(null);
  const tokenReconciler = useRef(new TokenReconciler());
  const baselineDraft = useRef<Axis[] | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const delayedRefresh = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loadingSnapshot = useRef(false);
  const pollAbort = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const workspaceRef = useRef(workspace);
  workspaceRef.current = workspace;
  const notice = useCallback(
    (message: string, kind: "error" | "success" = "success") => {
      const id = Date.now() + Math.random();
      setNotices((current) => [...current.slice(-3), { id, message, kind }]);
      if (kind === "success")
        setTimeout(
          () => setNotices((current) => current.filter((n) => n.id !== id)),
          6000,
        );
    },
    [],
  );
  const updateWorkspace = useCallback(
    (patch: Partial<Workspace>) =>
      setWorkspace((current) => ({ ...current, ...patch })),
    [],
  );
  useEffect(() => {
    localStorage.setItem("torment-nexus.workspace", JSON.stringify(workspace));
  }, [workspace]);

  const refresh = useCallback(async () => {
    if (!token || loadingSnapshot.current) return;
    loadingSnapshot.current = true;
    const controller = new AbortController();
    pollAbort.current = controller;
    try {
      const next = await api.state(controller.signal);
      setStreamingText((current) => ({
        ...current,
        ...Object.fromEntries(
          (next.runs ?? []).map((run) => [
            run.id,
            tokenReconciler.current.snapshot(run),
          ]),
        ),
      }));
      setState({
        ...emptyState,
        ...next,
        engine: { ...emptyState.engine, ...next.engine },
        codex: { ...emptyState.codex, ...next.codex },
      });
      setHasSnapshot(true);
      setConnection((current) =>
        current === "unauthorized" || current === "offline"
          ? "recovering"
          : current,
      );
    } catch (error) {
      if (controller.signal.aborted) return;
      setConnection(
        error instanceof ApiError && [401, 403].includes(error.status)
          ? "unauthorized"
          : "offline",
      );
    } finally {
      loadingSnapshot.current = false;
    }
  }, [api, token]);
  const queueRefresh = useCallback(
    (delay = 250) => {
      if (delayedRefresh.current !== null) return;
      delayedRefresh.current = setTimeout(() => {
        delayedRefresh.current = null;
        void refresh();
      }, delay);
    },
    [refresh],
  );
  const onEvent = useCallback(
    (event: ServerEvent) => {
      if (
        event.kind === "token" &&
        event.run_id &&
        typeof event.data.text === "string" &&
        typeof event.data.index === "number"
      ) {
        const output = tokenReconciler.current.token(
          event.run_id,
          event.data.index,
          event.data.text,
        );
        if (output !== undefined)
          setStreamingText((current) => ({
            ...current,
            [event.run_id!]: output,
          }));
        queueRefresh(400);
      } else if (
        event.kind === "controls_requested" &&
        event.run_id === live.current?.runId &&
        event.data.source !== "user" &&
        Array.isArray(event.data.coefficients)
      ) {
        const axes = requestedAxes(
          workspaceRef.current.axes,
          event.data.coefficients as Axis[],
        );
        const revision = Number(event.data.revision);
        if (live.current?.acceptExternal(axes, revision)) {
          updateWorkspace({ axes });
          setPendingRevision(revision);
        }
        queueRefresh();
      } else if (
        event.kind === "applied" &&
        event.run_id === live.current?.runId
      ) {
        const revision = Number(event.data.revision);
        const firstToken = Number(event.data.first_token_index);
        if (Number.isFinite(revision) && Number.isFinite(firstToken)) {
          if (live.current?.acknowledge(revision)) setPendingRevision(null);
          setAck({ revision, firstToken });
        }
        queueRefresh();
      } else {
        if (event.kind === "error") {
          const message = event.data.message ?? event.data.error;
          if (typeof message === "string") notice(message, "error");
          setShowJobs(true);
        }
        queueRefresh(event.kind === "completed" ? 20 : 250);
      }
    },
    [notice, queueRefresh, updateWorkspace],
  );
  useEffect(() => {
    if (!token) {
      setConnection("unauthorized");
      return;
    }
    const controller = new AbortController();
    let retry: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    const connect = async () => {
      try {
        await api.events(
          onEvent,
          () => {
            failures = 0;
            setConnection("live");
            void refresh();
          },
          controller.signal,
        );
      } catch (error) {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && [401, 403].includes(error.status)) {
          setConnection("unauthorized");
          return;
        }
      }
      if (controller.signal.aborted) return;
      setConnection((current) =>
        current === "offline" ? current : "recovering",
      );
      retry = setTimeout(
        () => void connect(),
        Math.min(1000 * 2 ** failures++, 15000),
      );
    };
    void refresh();
    void connect();
    const poll = setInterval(() => void refresh(), 4000);
    return () => {
      controller.abort();
      pollAbort.current?.abort();
      clearInterval(poll);
      clearTimeout(retry);
      if (delayedRefresh.current !== null) clearTimeout(delayedRefresh.current);
      delayedRefresh.current = null;
    };
  }, [api, onEvent, refresh, token]);

  const act = useCallback<Act>(
    async (action, fields = {}, success) => {
      setBusy((current) => new Set(current).add(action));
      try {
        const result = await api.action(action, fields);
        if (success) notice(success);
        await refresh();
        return result as never;
      } catch (error) {
        notice(errorMessage(error), "error");
        if (error instanceof ApiError && [401, 403].includes(error.status))
          setConnection("unauthorized");
        return null;
      } finally {
        setBusy((current) => {
          const next = new Set(current);
          next.delete(action);
          return next;
        });
      }
    },
    [api, refresh, notice],
  );
  const models = useMemo(
    () => [...state.models].sort(recordTime),
    [state.models],
  );
  const model = models.find((m) => m.id === workspace.modelId);
  const currentModelId = model?.id ?? "";
  const activeRun = state.runs.find((run) => activeStatuses.has(run.status));
  const inferenceBusy =
    (state.engine.status === "running" && !activeRun) ||
    state.engine.status === "loading" ||
    state.jobs.some(
      (job) =>
        activeStatuses.has(job.status) &&
        [
          "generation",
          "generate",
          "extraction",
          "extract",
          "load_model",
          "previews",
        ].includes(job.kind),
    );
  const frozen =
    !!activeRun || busy.has("generate") || busy.has("new_conversation");
  const compatibleVectors = state.vectors.filter(
    (vector) =>
      vector.model_id === currentModelId ||
      (model?.fingerprint && vector.model_fingerprint === model.fingerprint),
  );
  // Visibility is library-only: archived vectors still resolve mixes and history.
  const archivedVectorIds = new Set(
    (state.vector_visibility ?? [])
      .filter((entry) => entry.archived)
      .map((entry) => entry.id),
  );
  const archivedCount = compatibleVectors.filter(
    (vector) => !vector.deleted && archivedVectorIds.has(vector.id),
  ).length;
  const visibleVectors = compatibleVectors.filter(
    (vector) =>
      !vector.deleted &&
      (showArchived || !archivedVectorIds.has(vector.id)) &&
      vector.name.toLowerCase().includes(filter.toLowerCase()),
  );
  const knownAxes = workspace.axes.filter((axis) =>
    compatibleVectors.some((vector) => vector.id === axis.vector_id),
  );
  const selectedVectors = compatibleVectors
    .filter((vector) => knownAxes.some((axis) => axis.vector_id === vector.id))
    .map((vector) => ({
      vector,
      axes: knownAxes
        .filter((axis) => axis.vector_id === vector.id)
        .sort((a, b) => a.layer - b.layer),
    }));
  useEffect(() => {
    if (!hasSnapshot || frozen) return;
    const axes = expandLayerAxes(workspace.axes, state.vectors);
    if (signature(axes) !== signature(workspace.axes))
      updateWorkspace({ axes });
  }, [hasSnapshot, frozen, state.vectors, workspace.axes, updateWorkspace]);
  const engineReady =
    state.engine.model_id === currentModelId &&
    !["unloaded", "loading", "error", "failed"].includes(state.engine.status);
  const activeJobs = state.jobs.filter((job) => activeStatuses.has(job.status));
  const problemJobs = state.jobs.filter(
    (job) =>
      !job.attention_dismissed &&
      ["failed", "interrupted"].includes(job.status),
  );
  // The badge and drawer must agree, even when a failure predates recent history.
  const pinnedJobs = [...activeJobs, ...problemJobs].sort(recordTime);
  const pinnedIds = new Set(pinnedJobs.map((job) => job.id));
  const visibleJobs = [
    ...pinnedJobs,
    ...[...state.jobs]
      .sort(recordTime)
      .filter((job) => !pinnedIds.has(job.id))
      .slice(0, 40),
  ];

  useEffect(() => {
    if (
      hasSnapshot &&
      !state.models.some((model) => model.id === workspace.modelId)
    ) {
      const modelId = startupModelId(state, workspace.modelId);
      if (modelId !== workspace.modelId) updateWorkspace({ modelId });
    }
  }, [
    hasSnapshot,
    state.models,
    state.engine.model_id,
    state.runs,
    state.vectors,
    state.vector_visibility,
    updateWorkspace,
    workspace.modelId,
  ]);
  useEffect(() => {
    if (!activeRun) {
      live.current?.dispose();
      live.current = null;
      setPendingRevision(null);
      const restore =
        baselineDraft.current ?? workspaceRef.current.baselineRestore;
      if (restore) {
        updateWorkspace({ axes: restore, baselineRestore: null });
        baselineDraft.current = null;
      }
      return;
    }
    if (activeRun.baseline_of && baselineDraft.current === null) {
      baselineDraft.current =
        workspaceRef.current.baselineRestore ?? workspaceRef.current.axes;
      updateWorkspace({ baselineRestore: baselineDraft.current });
    }
    const initialAxes = requestedAxes(
      activeRun.axes,
      activeRun.requested_controls?.at(-1)?.coefficients,
    );
    const initialRevision = Math.max(
      0,
      ...(activeRun.requested_controls ?? []).map((c) => c.revision),
      ...(activeRun.applied_controls ?? []).map((c) => c.revision),
    );
    live.current = new LiveControls(
      activeRun.id,
      initialAxes,
      initialRevision,
      (snapshot) => api.action("controls", snapshot),
      (revision) => setPendingRevision(revision),
      (error) =>
        notice(
          `Control update failed: ${errorMessage(error)}. Adjust again to retry.`,
          "error",
        ),
    );
    updateWorkspace({ modelId: activeRun.model_id, axes: initialAxes });
    setFocusedRunId(activeRun.baseline_of ?? activeRun.id);
    const applied = activeRun.applied_controls?.at(-1);
    setAck(
      applied
        ? {
            revision: applied.revision,
            firstToken: applied.first_token_index ?? 0,
          }
        : null,
    );
    if (initialRevision > (applied?.revision ?? -1))
      setPendingRevision(initialRevision);
    return () => {
      live.current?.dispose();
      live.current = null;
    };
    // A run freezes selection once. Snapshot refreshes must not reset live slider edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRun?.id, api, notice, updateWorkspace]);
  useEffect(() => {
    if (!live.current) return;
    try {
      live.current.update(workspace.axes);
    } catch (error) {
      notice(errorMessage(error), "error");
    }
  }, [workspace.axes, notice]);
  useEffect(() => {
    if (!activeRun || !live.current) return;
    const requested = activeRun.requested_controls?.at(-1);
    if (
      requested?.source &&
      requested.source !== "user" &&
      live.current.acceptExternal(
        requestedAxes(activeRun.axes, requested.coefficients),
        requested.revision,
      )
    ) {
      updateWorkspace({
        axes: requestedAxes(activeRun.axes, requested.coefficients),
      });
      setPendingRevision(requested.revision);
    }
    const latest = activeRun.applied_controls?.at(-1);
    if (latest && live.current.acknowledge(latest.revision)) {
      setPendingRevision(null);
      setAck({
        revision: latest.revision,
        firstToken: latest.first_token_index ?? 0,
      });
    }
  }, [
    activeRun?.applied_controls,
    activeRun?.requested_controls,
    updateWorkspace,
  ]);

  const runs = useMemo(
    () => state.runs.filter((run) => !run.deleted).sort(recordTime),
    [state.runs],
  );
  const currentConversation = state.conversations.find(
    (c) => c.id === workspace.conversationId && !c.deleted,
  );
  const currentPreset = state.presets.find(
    (p) =>
      p.id === selectedPresetId && p.model_id === currentModelId && !p.deleted,
  );
  const recordActions = (kind: ManagedKind, id: string, name: string) => (
    <span className="record-actions">
      <button
        aria-label={`Rename ${recordLabels[kind]} ${name}`}
        onClick={() => setManagement({ kind, id, name, operation: "rename" })}
      >
        Rename
      </button>
      <button
        aria-label={`Delete ${recordLabels[kind]} ${name}`}
        disabled={frozen || inferenceBusy}
        onClick={() => setManagement({ kind, id, name, operation: "delete" })}
      >
        Delete
      </button>
    </span>
  );
  const visibleRuns =
    workspace.mode === "chat"
      ? workspace.conversationId
        ? conversationRuns(
            state.runs,
            workspace.conversationId,
            currentConversation?.run_ids,
          )
        : []
      : ([
          runs.find((run) => run.id === focusedRunId) ??
            runs.find(
              (run) =>
                run.model_id === currentModelId &&
                !run.conversation_id &&
                !run.baseline_of,
            ),
        ].filter(Boolean) as Run[]);
  const latestRun = visibleRuns.at(-1);
  const runOutput = useCallback(
    (run: Run) => {
      const streamed = streamingText[run.id] ?? "";
      return streamingText[run.id] !== undefined
        ? streamed
        : (run.output ?? "");
    },
    [streamingText],
  );
  useEffect(() => {
    const node = transcriptRef.current;
    if (node && node.scrollHeight - node.scrollTop - node.clientHeight < 220)
      node.scrollTop = node.scrollHeight;
  }, [streamingText, visibleRuns.length]);

  const selectModel = (id: string) => {
    if (frozen) return;
    updateWorkspace({
      modelId: id,
      axes: [],
      conversationId: "",
      prefixMessages: [],
    });
    setFocusedRunId("");
  };
  const toggleVector = (vector: Vector) => {
    if (frozen) return;
    const has = knownAxes.some((axis) => axis.vector_id === vector.id);
    updateWorkspace({
      axes: has
        ? knownAxes.filter((axis) => axis.vector_id !== vector.id)
        : [
            ...knownAxes,
            ...vector.layers
              .filter((layer) => layer.usable !== false)
              .map((layer) => ({
                vector_id: vector.id,
                layer: layer.layer,
                percent: 0,
              })),
          ],
    });
  };
  const changeAxis = (id: string, layer: number, percent: number) =>
    updateWorkspace({
      axes: workspace.axes.map((axis) =>
        axis.vector_id === id && axis.layer === layer
          ? { ...axis, percent }
          : axis,
      ),
    });
  const newChat = async () => {
    if (frozen) return;
    const result = await act<{ id: string }>("new_conversation", {
      title: "Untitled experiment",
    });
    if (result) {
      updateWorkspace({
        mode: "chat",
        conversationId: result.id,
        prefixMessages: [],
        prompt: "",
      });
      setFocusedRunId("");
    }
  };
  const switchToChat = async () => {
    if (frozen || workspace.mode === "chat") return;
    const prompt =
      workspace.prompt.trim() ===
      [...(latestRun?.messages ?? [])]
        .reverse()
        .find((message) => message.role === "user")?.content
        ? ""
        : workspace.prompt;
    if (
      latestRun &&
      conversationRuns(
        state.runs,
        workspace.conversationId,
        currentConversation?.run_ids,
      ).at(-1)?.id !== latestRun.id
    ) {
      const result = await act<{ id: string }>("new_conversation", {
        title:
          [...latestRun.messages]
            .reverse()
            .find((message) => message.role === "user")
            ?.content.slice(0, 64) || "Continued experiment",
        from_run_id: latestRun.id,
      });
      if (!result) return;
      updateWorkspace({
        mode: "chat",
        conversationId: result.id,
        prefixMessages: [],
        prompt,
      });
    } else {
      updateWorkspace({ mode: "chat", prefixMessages: [], prompt });
    }
  };
  const startRun = async (fields: {
    messages: Message[];
    axes: Axis[];
    sampling: Sampling;
    model_id: string;
    raw: boolean;
    conversation_id?: string;
    baseline_of?: string;
    self_modification?: boolean;
  }) => {
    const result = await act<{ run_id: string; job_id: string }>(
      "generate",
      fields,
    );
    if (result) {
      setFocusedRunId(result.run_id);
      setState((current) =>
        current.runs.some((run) => run.id === result.run_id)
          ? current
          : {
              ...current,
              runs: [
                ...current.runs,
                {
                  id: result.run_id,
                  created_at: Date.now() / 1000,
                  ...fields,
                  status: "running",
                  output: "",
                },
              ],
            },
      );
    }
    return result;
  };
  const generate = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!workspace.prompt.trim() || !model || frozen) return;
    let conversationId =
      workspace.mode === "chat" ? workspace.conversationId : undefined;
    if (workspace.mode === "chat" && !conversationId) {
      const result = await act<{ id: string }>("new_conversation", {
        title: workspace.prompt.trim().slice(0, 64),
      });
      if (!result) return;
      conversationId = result.id;
      updateWorkspace({ conversationId });
    }
    let messages: Message[] =
      workspace.mode === "scratchpad" && workspace.prefixMessages.length
        ? [...workspace.prefixMessages]
        : workspace.system.trim()
          ? [{ role: "system", content: workspace.system }]
          : [];
    if (
      workspace.mode === "chat" &&
      currentConversation?.id === conversationId
    ) {
      if (latestRun)
        messages = conversationHistory(latestRun, runOutput(latestRun));
      else if (currentConversation?.messages?.length)
        messages = [...currentConversation.messages];
    }
    messages.push({ role: "user", content: workspace.prompt.trim() });
    const result = await startRun({
      model_id: model.id,
      messages,
      axes: knownAxes,
      sampling: workspace.sampling,
      self_modification: workspace.selfModification,
      raw: workspace.raw,
      ...(conversationId ? { conversation_id: conversationId } : {}),
    });
    if (result && workspace.mode === "chat") updateWorkspace({ prompt: "" });
  };
  const compare = async (run: Run) => {
    const result = await startRun({
      model_id: run.model_id,
      messages: run.messages,
      axes: [],
      sampling: run.sampling,
      self_modification: false,
      raw: run.raw ?? false,
      baseline_of: run.id,
    });
    if (result) setFocusedRunId(run.id);
  };
  const duplicate = async (run: Run) => {
    if (frozen) return;
    const result = await act<{ id: string }>("new_conversation", {
      title: `${
        [...run.messages]
          .reverse()
          .find((message) => message.role === "user")
          ?.content.slice(0, 56) || "Experiment"
      } (fork)`,
      from_run_id: run.id,
    });
    if (!result) return;
    const lastApplied = run.applied_controls?.at(-1);
    const applied = run.requested_controls?.find(
      (event) => event.revision === lastApplied?.revision,
    );
    updateWorkspace({
      modelId: run.model_id,
      axes: requestedAxes(run.axes, applied?.coefficients),
      sampling: run.sampling,
      selfModification:
        run.self_modification ?? run.self_tools_available ?? false,
      prefixMessages: [],
      mode: "chat",
      conversationId: result.id,
      raw: run.raw ?? false,
      prompt: "",
      system:
        run.messages.find((message) => message.role === "system")?.content ??
        "",
    });
    setFocusedRunId(run.id);
    setDialog(null);
    notice("Chat forked through this reply. The original is unchanged.");
  };
  const exportVector = async (vector: Vector) => {
    const bundle = await act("export_vector", { vector_id: vector.id });
    if (bundle)
      downloadJson(
        `${vector.name.replace(/[^a-z0-9_-]+/gi, "-").toLowerCase()}-${shortId(vector.id)}.json`,
        bundle,
      );
  };

  if (connection === "unauthorized")
    return (
      <div className="auth-screen">
        <NexusMark large />
        <span className="eyebrow">LOCAL STEERING LAB</span>
        <h1>One launch. One private session.</h1>
        <p>
          Open the browser link printed by the Torment Nexus launcher. Its
          launch token stays in this tab, not in your URL or model history.
        </p>
        <div className="error-box">
          {token
            ? "This session token was rejected. Restarting the app creates a new launch link."
            : "No launch token found. Use the full launch URL, including its #token fragment."}
        </div>
        <button className="button primary" onClick={() => void refresh()}>
          Try connection again <Icon name="arrow" />
        </button>
      </div>
    );

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
          <NexusMark />
          <div>
            <strong>
              torment nexus<span className="brand-period">.</span>
            </strong>
            <span className="brand-subtitle">LOCAL ACTIVATION LAB</span>
          </div>
        </a>
        <div className="topbar-center">
          <span
            className={`status-dot ${connection === "live" ? "green" : "amber"}`}
          />
          <span>
            {connection === "live"
              ? "Local inference · private session"
              : connection === "recovering"
                ? "Reconnecting · saved state available"
                : connection === "offline"
                  ? "Launcher disconnected"
                  : "Connecting to your lab"}
          </span>
        </div>
        <div className="topbar-actions">
          <button className="quiet-button" onClick={() => setDialog("history")}>
            <Icon name="history" />
            History<span className="counter">{state.runs.length}</span>
          </button>
          <button className="session-pill" onClick={() => setDialog("models")}>
            <span className={`status-dot ${engineReady ? "green" : ""}`} />
            {connection === "offline"
              ? "Snapshot only"
              : engineReady
                ? "Engine ready"
                : "Engine offline"}
            <Icon name="chevron" size={13} />
          </button>
        </div>
      </header>

      {connection === "offline" && (
        <div className="connection-banner">
          The local server is unavailable. Your last saved snapshot is still
          shown; pending controls are not assumed applied.{" "}
          <button onClick={() => void refresh()}>Reconnect</button>
        </div>
      )}
      <main
        className={`workbench ${!libraryOpen ? "library-collapsed" : ""} ${!mixerOpen ? "mixer-collapsed" : ""} ${focusChat ? "chat-focused" : ""}`}
      >
        <aside className="library-panel panel" id="library-panel">
          <div className="panel-heading">
            <span className="eyebrow">01 / MATERIALS</span>
            <button
              className="icon-button"
              title="Import vector bundle"
              aria-label="Import vector bundle"
              onClick={() => setDialog("import")}
            >
              <Icon name="download" size={16} />
            </button>
          </div>
          <button
            className="model-selector"
            onClick={() => setDialog("models")}
          >
            <div className="model-glyph">◈</div>
            <div>
              <span className="micro-label">ACTIVE MODEL</span>
              <strong>{model?.name ?? "Choose a model"}</strong>
              <span className="dim">
                {model
                  ? `${formatSize(model.size_bytes)} · GGUF`
                  : "Import locally or download from HF"}
              </span>
            </div>
            <Icon name="chevron" size={15} />
          </button>
          {model && !engineReady && (
            <button
              className="button engine-load"
              disabled={frozen || inferenceBusy || busy.has("load_model")}
              onClick={() =>
                void act(
                  "load_model",
                  { model_id: model.id },
                  "Model load requested. Watch jobs for progress.",
                )
              }
            >
              <Icon name="bolt" size={14} />
              {busy.has("load_model") ? "Loading…" : "Load into engine"}
            </button>
          )}
          <div className="library-heading">
            <h2>Your directions</h2>
            <Badge>{visibleVectors.length}</Badge>
          </div>
          <button
            className="button primary full"
            onClick={() => setDialog("concept")}
            disabled={!model}
          >
            <Icon name="plus" size={16} />
            Create a concept
          </button>
          <div className="segmented library-tabs">
            <button
              className={libraryTab === "vectors" ? "selected" : ""}
              onClick={() => setLibraryTab("vectors")}
            >
              Vectors
            </button>
            <button
              className={libraryTab === "recipes" ? "selected" : ""}
              onClick={() => setLibraryTab("recipes")}
            >
              Recipes{" "}
              <span>{state.recipes.filter((r) => !r.deleted).length}</span>
            </button>
          </div>
          <label className="search-field">
            <Icon name="search" size={15} />
            <input
              placeholder="Filter your library"
              aria-label="Filter your library"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </label>
          {libraryTab === "vectors" && (archivedCount > 0 || showArchived) && (
            <label className="archive-toggle">
              <input
                type="checkbox"
                checked={showArchived}
                onChange={(event) => setShowArchived(event.target.checked)}
              />
              Show archived ({archivedCount})
            </label>
          )}
          <div className="library-list">
            {libraryTab === "vectors" ? (
              <>
                {visibleVectors.map((vector) => {
                  const archived = archivedVectorIds.has(vector.id);
                  const selected = knownAxes.some(
                    (axis) => axis.vector_id === vector.id,
                  );
                  const best = vector.layers.find(
                    (layer) => layer.layer === vector.selected_layer,
                  );
                  return (
                    <article
                      className={`vector-card ${selected ? "chosen" : ""}`}
                      key={vector.id}
                    >
                      <button
                        className="vector-choice"
                        onClick={() => toggleVector(vector)}
                        disabled={frozen}
                        aria-pressed={selected}
                        aria-label={`${selected ? "Remove" : "Add"} ${vector.name} ${selected ? "from" : "to"} mix`}
                      >
                        <span className="checkbox">
                          {selected && <Icon name="check" size={13} />}
                        </span>
                        <strong title={vector.name}>{vector.name}</strong>
                      </button>
                      <div className="vector-meta">
                        <span>L{vector.selected_layer}</span>
                        <span>
                          {best?.auc !== undefined
                            ? `${(best.auc * 100).toFixed(0)}% separation`
                            : "Separation unmeasured"}
                        </span>
                        <button
                          onClick={() =>
                            setInspect({ kind: "vector", id: vector.id })
                          }
                          title="Vector details"
                          aria-label={`Inspect ${vector.name}`}
                        >
                          <Icon name="chevron" size={14} />
                        </button>
                      </div>
                      <div className="vector-library-actions">
                        {recordActions("vectors", vector.id, vector.name)}
                        {archived && <span>Archived</span>}
                        <button
                          aria-label={`${archived ? "Restore" : "Archive"} ${vector.name}`}
                          title={
                            archived
                              ? "Show in the library again"
                              : "Hide from the library; keep current mix, saved runs, and provenance"
                          }
                          disabled={busy.has("set_vector_archived")}
                          onClick={() =>
                            void act("set_vector_archived", {
                              vector_id: vector.id,
                              archived: !archived,
                            })
                          }
                        >
                          {archived ? "Restore" : "Archive"}
                        </button>
                      </div>
                    </article>
                  );
                })}
                {!compatibleVectors.some((v) => !v.deleted) && (
                  <Empty
                    title={
                      state.vectors.some((v) => !v.deleted)
                        ? "No vectors for this model"
                        : "A little potential energy"
                    }
                    action={
                      state.vectors.some((v) => !v.deleted) ? (
                        <button
                          className="button secondary"
                          onClick={() => setDialog("models")}
                        >
                          Choose another model
                        </button>
                      ) : undefined
                    }
                  >
                    {state.vectors.some((v) => !v.deleted)
                      ? "Your saved vectors belong to other models. Switch models to use them."
                      : "Create a concept to turn a contrast into a direction. No personality claims required."}
                  </Empty>
                )}
                {compatibleVectors.some((v) => !v.deleted) &&
                  !visibleVectors.length && (
                    <Empty title="Nothing visible here">
                      {filter
                        ? "Try another filter."
                        : "Your archived directions are still available with Show archived."}
                    </Empty>
                  )}
              </>
            ) : (
              <>
                <p className="panel-intro">Shared across all models.</p>
                {state.recipes
                  .filter(
                    (recipe) =>
                      !recipe.deleted &&
                      (recipe.display_name ?? recipe.concept)
                        .toLowerCase()
                        .includes(filter.toLowerCase()),
                  )
                  .sort(recordTime)
                  .map((recipe) => (
                    <div className="recipe-entry" key={recipe.id}>
                      <button
                        className="recipe-card"
                        onClick={() =>
                          setInspect({ kind: "recipe", id: recipe.id })
                        }
                      >
                        <div>
                          <strong title={recipe.display_name ?? recipe.concept}>
                            {recipe.display_name ?? recipe.concept}
                          </strong>
                          <Badge>v{recipe.version}</Badge>
                        </div>
                        <span>
                          {dateLabel(recipe.created_at)}
                          <Icon name="chevron" size={14} />
                        </span>
                      </button>
                      {recordActions(
                        "recipes",
                        recipe.id,
                        recipe.display_name ?? recipe.concept,
                      )}
                    </div>
                  ))}
                {!state.recipes.some((recipe) => !recipe.deleted) && (
                  <Empty title="The paper trail starts here">
                    Every design, dataset, and revision stays inspectable. Edits
                    make a new version.
                  </Empty>
                )}
              </>
            )}
          </div>
          <div className="library-footer">
            <button onClick={() => setDialog("models")}>
              Manage models <Icon name="arrow" size={13} />
            </button>
          </div>
        </aside>

        <section className="experiment-panel panel">
          <div className="experiment-heading">
            <div>
              <h1>Conversation</h1>
            </div>
            <div className="panel-toggles">
              <button
                aria-label="Toggle library"
                aria-controls="library-panel"
                aria-expanded={libraryOpen}
                onClick={() => setLibraryOpen(!libraryOpen)}
              >
                {libraryOpen ? "‹" : "›"} Library
              </button>
              <button
                aria-label="Toggle mixer"
                aria-controls="mixer-panel"
                aria-expanded={mixerOpen}
                onClick={() => setMixerOpen(!mixerOpen)}
              >
                Mix {mixerOpen ? "›" : "‹"}
              </button>
            </div>
            <span className="experiment-number">
              EXPERIMENT
              <br />
              <b>{String(state.runs.length + 1).padStart(3, "0")}</b>
            </span>
          </div>
          <div className="experiment-toolbar">
            <div className="segmented">
              <button
                className={workspace.mode === "scratchpad" ? "selected" : ""}
                onClick={() => {
                  if (frozen) return;
                  setFocusedRunId(latestRun?.id ?? "");
                  updateWorkspace({ mode: "scratchpad" });
                }}
                disabled={frozen}
              >
                Scratchpad
              </button>
              <button
                className={workspace.mode === "chat" ? "selected" : ""}
                onClick={() => void switchToChat()}
                disabled={frozen}
              >
                Chat
              </button>
            </div>
            <div className="toolbar-right">
              <button
                className="quiet-button focus-chat"
                onClick={() => {
                  setLibraryOpen(focusChat);
                  setMixerOpen(focusChat);
                }}
                aria-pressed={focusChat}
              >
                {focusChat ? "Show panels" : "Focus chat"}
              </button>
              {workspace.mode === "chat" && (
                <button
                  className="quiet-button"
                  onClick={() => void newChat()}
                  disabled={frozen}
                >
                  <Icon name="plus" size={14} />
                  New chat
                </button>
              )}
              <button
                className={`icon-button ${showSettings ? "active" : ""}`}
                title="Sampling settings"
                aria-label="Sampling settings"
                aria-expanded={showSettings}
                onClick={() => setShowSettings(!showSettings)}
              >
                <Icon name="sliders" size={17} />
              </button>
            </div>
          </div>
          <div className="generation-options">
            <label title="Lets the local model or an MCP client adjust only this response's selected vector/layer sliders. You can turn it off mid-response.">
              <input
                type="checkbox"
                checked={
                  activeRun
                    ? !!activeRun.self_modification
                    : workspace.selfModification
                }
                disabled={
                  busy.has("set_self_modification") ||
                  (!!activeRun && !activeRun.self_tools_available)
                }
                onChange={async (event) => {
                  const enabled = event.target.checked;
                  if (activeRun)
                    setState((current) => ({
                      ...current,
                      runs: current.runs.map((run) =>
                        run.id === activeRun.id
                          ? { ...run, self_modification: enabled }
                          : run,
                      ),
                    }));
                  if (
                    activeRun &&
                    !(await act("set_self_modification", {
                      run_id: activeRun.id,
                      enabled,
                    }))
                  ) {
                    void refresh();
                    return;
                  }
                  updateWorkspace({ selfModification: enabled });
                }}
              />{" "}
              Allow self-adjustment
              {busy.has("set_self_modification") && <span>· updating…</span>}
            </label>
            <label title="No output-token cap. Stops naturally or when you press Stop. When context fills, replay the prefix and recent tail; older context is dropped, not magically remembered.">
              <input
                type="checkbox"
                checked={!!workspace.sampling.unbounded}
                disabled={frozen}
                onChange={(event) =>
                  updateWorkspace({
                    sampling: {
                      ...workspace.sampling,
                      unbounded: event.target.checked,
                    },
                  })
                }
              />{" "}
              Unbounded output
            </label>
          </div>
          {workspace.mode === "chat" && (
            <div className="chat-selector">
              <select
                aria-label="Conversation"
                value={workspace.conversationId}
                disabled={frozen}
                onChange={(e) =>
                  updateWorkspace({
                    conversationId: e.target.value,
                    prefixMessages: [],
                  })
                }
              >
                <option value="">New conversation</option>
                {[...state.conversations]
                  .filter((conversation) => !conversation.deleted)
                  .sort(recordTime)
                  .map((conversation) => (
                    <option value={conversation.id} key={conversation.id}>
                      {conversation.title || shortId(conversation.id)}
                    </option>
                  ))}
              </select>
              {currentConversation &&
                recordActions(
                  "conversations",
                  currentConversation.id,
                  currentConversation.title,
                )}
              <span>Fresh model state each turn</span>
            </div>
          )}
          {showSettings && (
            <div className="sampling-panel">
              <div className="sampling-fields">
                {(
                  [
                    ["seed", "Seed", 1, 0, 4294967295],
                    ["temperature", "Temperature", 0.05, 0, 5],
                    ["top_p", "Top p", 0.05, 0.01, 1],
                    ["max_tokens", "Max tokens", 1, 1, 32768],
                  ] as const
                ).map(([key, label, step, min, max]) => (
                  <label key={key}>
                    {label}
                    <input
                      type="number"
                      value={workspace.sampling[key]}
                      step={step}
                      min={min}
                      max={max}
                      disabled={
                        frozen ||
                        (key === "max_tokens" && workspace.sampling.unbounded)
                      }
                      onChange={(e) => {
                        const value = finitePercent(e.target.value);
                        if (value !== null)
                          updateWorkspace({
                            sampling: { ...workspace.sampling, [key]: value },
                          });
                      }}
                    />
                  </label>
                ))}
              </div>
              <label className="field">
                System instruction
                <textarea
                  rows={2}
                  placeholder="Optional. Kept on this machine."
                  value={workspace.system}
                  disabled={frozen}
                  onChange={(e) => updateWorkspace({ system: e.target.value })}
                />
              </label>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={workspace.raw}
                  disabled={frozen}
                  onChange={(e) => updateWorkspace({ raw: e.target.checked })}
                />
                Raw-completion mode{" "}
                <span>
                  For template-free models. One user message is literal text;
                  conversations use explicit role labels, not a chat template.
                </span>
              </label>
            </div>
          )}
          <div className="transcript" ref={transcriptRef}>
            {!visibleRuns.length && (
              <div className="welcome-state">
                <div className="signal-graphic" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                </div>
                <h2>What happens if…</h2>
                <p>
                  Start at zero. Change a direction.
                  <br />
                  See what the model actually does.
                </p>
                <div className="prompt-suggestions">
                  {[
                    "Describe a quiet room just before sunrise.",
                    "Explain why people collect things.",
                    "A door appears where there was a wall.",
                  ].map((prompt) => (
                    <button
                      key={prompt}
                      onClick={() => updateWorkspace({ prompt })}
                    >
                      {prompt}
                      <Icon name="arrow" size={14} />
                    </button>
                  ))}
                </div>
              </div>
            )}
            {visibleRuns.map((run) => (
              <RunCard
                key={run.id}
                run={run}
                output={runOutput(run)}
                state={state}
                baseline={runs.find(
                  (candidate) => candidate.baseline_of === run.id,
                )}
                baselineOutput={runOutput}
                frozen={frozen}
                compare={() => void compare(run)}
                duplicate={() => duplicate(run)}
                inspect={() => setInspect({ kind: "run", id: run.id })}
              />
            ))}
          </div>
          <form className="composer" onSubmit={(e) => void generate(e)}>
            <div className="composer-field">
              <textarea
                aria-label="Prompt"
                placeholder={
                  workspace.mode === "chat"
                    ? "Say something. Follow the thread."
                    : "Ask the same question. Try a different direction."
                }
                value={workspace.prompt}
                onChange={(e) => updateWorkspace({ prompt: e.target.value })}
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                    e.preventDefault();
                    void generate();
                  }
                }}
                rows={3}
              />
              <div className="composer-bottom">
                <span>
                  {workspace.mode === "chat"
                    ? "History re-encoded · current mix"
                    : workspace.prefixMessages.length
                      ? `${workspace.prefixMessages.length + 1}-message transcript · fresh state`
                      : "Fresh state · independent prompt"}
                  <kbd>⌘ ↵</kbd>
                </span>
                {activeRun ? (
                  <button
                    className="button stop"
                    type="button"
                    disabled={busy.has("cancel_run")}
                    onClick={() =>
                      void act("cancel_run", { run_id: activeRun.id })
                    }
                  >
                    <Icon name="stop" size={14} />
                    Stop
                  </button>
                ) : (
                  <button
                    className="button primary"
                    type="submit"
                    disabled={
                      !model ||
                      !engineReady ||
                      inferenceBusy ||
                      frozen ||
                      !workspace.prompt.trim() ||
                      connection === "offline"
                    }
                  >
                    {busy.has("generate") ? "Starting…" : "Run prompt"}
                    <Icon name="send" size={15} />
                  </button>
                )}
              </div>
            </div>
            {workspace.mode === "scratchpad" &&
              workspace.prefixMessages.length > 0 && (
                <div className="composer-note">
                  <button
                    type="button"
                    className="quiet-button"
                    onClick={() => updateWorkspace({ prefixMessages: [] })}
                  >
                    Clear duplicated context
                  </button>
                </div>
              )}
          </form>
        </section>

        <aside className="mixer-panel panel" id="mixer-panel">
          <div className="panel-heading">
            <span className="eyebrow">03 / INTERVENTION</span>
            <Icon name="sliders" size={16} />
          </div>
          <div className="mixer-title">
            <h2>The mix</h2>
            <Badge>
              {selectedVectors.length}{" "}
              {selectedVectors.length === 1 ? "concept" : "concepts"}
            </Badge>
          </div>
          <div
            className={`control-status ${pendingRevision !== null ? "pending" : activeRun ? "live" : ""}`}
          >
            <span
              className={`status-dot ${pendingRevision !== null ? "amber" : activeRun ? "green" : ""}`}
            />
            <span>
              {pendingRevision !== null
                ? `Revision ${pendingRevision} pending`
                : activeRun
                  ? ack
                    ? `Rev ${ack.revision} · from token ${ack.firstToken}`
                    : "Initial controls awaiting engine"
                  : "Ready for the next response"}
            </span>
          </div>
          {frozen && (
            <p className="frozen-note">
              Vectors and layers are fixed for this response. Coefficients are
              live.
            </p>
          )}
          <div className="mixer-axes">
            {selectedVectors.map(({ axes, vector }, index) => (
              <ConceptControl
                key={vector.id}
                index={index}
                axes={axes}
                vector={vector}
                frozen={frozen}
                change={(layer, percent) =>
                  changeAxis(vector.id, layer, percent)
                }
                remove={() => toggleVector(vector)}
              />
            ))}
            {!selectedVectors.length && (
              <Empty title="Nothing pulling, yet">
                Add directions from the library. Zero is a perfectly good place
                to start.
              </Empty>
            )}
          </div>
          <div className="mix-actions">
            <button
              className="button secondary"
              disabled={!selectedVectors.length}
              onClick={() =>
                updateWorkspace({
                  axes: workspace.axes.map((axis) => ({ ...axis, percent: 0 })),
                })
              }
            >
              <Icon name="reset" size={14} />
              Zero all
            </button>
            <button
              className="button secondary"
              disabled={!selectedVectors.length}
              onClick={() => setDialog("mix")}
            >
              <Icon name="archive" size={14} />
              Save mix
            </button>
          </div>
          {state.presets.some(
            (preset) => !preset.deleted && preset.model_id === currentModelId,
          ) && (
            <div className="preset-picker">
              <label htmlFor="saved-mix">SAVED MIXES</label>
              <select
                id="saved-mix"
                aria-label="Load saved mix"
                value={currentPreset?.id ?? ""}
                disabled={frozen}
                onChange={(e) => {
                  const preset = state.presets.find(
                    (p) => p.id === e.target.value && !p.deleted,
                  );
                  setSelectedPresetId(preset?.id ?? "");
                  if (preset) updateWorkspace({ axes: preset.axes });
                }}
              >
                <option value="">Load a saved mix…</option>
                {state.presets
                  .filter(
                    (preset) =>
                      !preset.deleted && preset.model_id === currentModelId,
                  )
                  .map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name}
                    </option>
                  ))}
              </select>
              {currentPreset &&
                recordActions("presets", currentPreset.id, currentPreset.name)}
            </div>
          )}
          <details className="mixer-explainer">
            <summary>How the layer controls work</summary>
            <div className="formula">
              Δh<sub>ℓ</sub> = ∑ (p / 100) · s<sub>ℓ</sub> · v̂<sub>ℓ</sub>
            </div>
            <p>
              Each layer has its own coefficient; effects can interact.
              Directions add without renormalization. Later changes affect
              future computation, not tokens already written.
            </p>
            <p>
              ±20% is just the starting range. Type a coefficient directly—even
              200%—or expand the slider. Large values may produce nonsense.
            </p>
          </details>
          {state.mcp && (
            <details className="mcp-connection">
              <summary>MCP connection</summary>
              <p>
                For an external MCP client. Only the active response's mix is
                exposed, and only while self-adjustment is on. This private
                connection token changes each launch.
              </p>
              <textarea
                aria-label="MCP connection configuration"
                readOnly
                rows={8}
                value={JSON.stringify(
                  {
                    url: state.mcp.url,
                    headers: { Authorization: `Bearer ${state.mcp.token}` },
                  },
                  null,
                  2,
                )}
              />
            </details>
          )}
          <div
            className="model-footprint"
            title={
              state.engine.memory_bytes != null
                ? "Peak worker RSS · last sample"
                : engineReady
                  ? "Model file size · not live RAM"
                  : "No model loaded"
            }
          >
            <span>
              {state.engine.memory_bytes != null
                ? "WORKER MEMORY"
                : "MODEL FILE"}
            </span>
            <strong>
              {formatSize(
                state.engine.memory_bytes ??
                  (engineReady ? model?.size_bytes : undefined),
              )}
            </strong>
          </div>
        </aside>
      </main>

      <section className={`jobs-drawer ${showJobs ? "expanded" : ""}`}>
        <button
          className="jobs-toggle"
          onClick={() => setShowJobs(!showJobs)}
          aria-expanded={showJobs}
        >
          <span>
            <Icon name="code" size={16} />
            <strong>Jobs & provenance</strong>
            {activeJobs.length ? (
              <Badge tone="mint">{activeJobs.length} active</Badge>
            ) : (
              <span className="dim">Nothing running in the background</span>
            )}
            {problemJobs.length > 0 && (
              <Badge tone="amber">
                {problemJobs.length}{" "}
                {problemJobs.length === 1 ? "needs" : "need"} attention
              </Badge>
            )}
          </span>
          <span className="drawer-hint">
            {showJobs ? "Collapse" : "Open inspector"}
            <Icon name="chevron" size={14} />
          </span>
        </button>
        {showJobs && (
          <div className="jobs-content">
            {visibleJobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                act={act}
                fork={
                  job.kind === "factory" &&
                  Object.keys(checkpointRefs(jobCheckpoints(job))).length
                    ? () => setForkJobId(job.id)
                    : undefined
                }
                inspect={
                  job.recipe_id
                    ? () => setInspect({ kind: "recipe", id: job.recipe_id! })
                    : job.run_id
                      ? () => setInspect({ kind: "run", id: job.run_id! })
                      : undefined
                }
                duplicate={
                  !frozen &&
                  job.run_id &&
                  state.runs.some((run) => run.id === job.run_id)
                    ? () =>
                        duplicate(
                          state.runs.find((run) => run.id === job.run_id)!,
                        )
                    : undefined
                }
              />
            ))}
            {!state.jobs.length && (
              <p className="muted">
                Downloads, agents, extraction, and inference will appear here.
                Completed stages survive a restart; retries are yours to
                request.
              </p>
            )}
          </div>
        )}
      </section>
      <div className="notices" aria-live="polite">
        {notices.map((item) => (
          <div className={`notice ${item.kind}`} key={item.id}>
            <span>{item.message}</span>
            <button
              onClick={() =>
                setNotices((current) => current.filter((n) => n.id !== item.id))
              }
              aria-label="Dismiss notification"
            >
              <Icon name="close" size={16} />
            </button>
          </div>
        ))}
      </div>

      {dialog === "models" && (
        <ModelDialog
          state={state}
          selected={currentModelId}
          select={selectModel}
          act={act}
          busy={busy}
          frozen={frozen || inferenceBusy}
          close={() => setDialog(null)}
        />
      )}
      {dialog === "concept" && (
        <ConceptDialog
          state={state}
          modelId={currentModelId}
          raw={workspace.raw}
          act={act}
          busy={busy}
          close={() => setDialog(null)}
          created={() => {
            setDialog(null);
            setLibraryTab("recipes");
            setShowJobs(true);
          }}
        />
      )}
      {dialog === "history" && (
        <Modal
          title="Every experiment, kept."
          eyebrow="RUN HISTORY"
          wide
          onClose={() => setDialog(null)}
        >
          <div className="history-list">
            {runs.map((run) => (
              <article className="history-card" key={run.id}>
                <div>
                  <Badge
                    tone={
                      run.status === "completed"
                        ? "mint"
                        : run.status === "failed"
                          ? "amber"
                          : ""
                    }
                  >
                    {run.status}
                  </Badge>
                  <span>{dateLabel(run.created_at)}</span>
                  {run.baseline_of && <Badge>baseline</Badge>}
                  <span>{run.axes.length} axes</span>
                </div>
                <h3>
                  {[...run.messages]
                    .reverse()
                    .find((message) => message.role === "user")?.content ||
                    "Untitled run"}
                </h3>
                <p>
                  {run.output?.slice(0, 220) ||
                    run.error ||
                    "No output recorded yet."}
                </p>
                <div className="history-actions">
                  <button
                    className="quiet-button"
                    disabled={frozen}
                    onClick={() => {
                      updateWorkspace({
                        modelId: run.model_id,
                        mode: run.conversation_id ? "chat" : "scratchpad",
                        conversationId: run.conversation_id ?? "",
                      });
                      setFocusedRunId(run.baseline_of ?? run.id);
                      setDialog(null);
                    }}
                  >
                    Open run
                    <Icon name="arrow" size={14} />
                  </button>
                  <button
                    className="quiet-button"
                    disabled={frozen}
                    onClick={() => duplicate(run)}
                  >
                    <Icon name="copy" size={14} />
                    Fork chat
                  </button>
                  <button
                    className="quiet-button"
                    onClick={() => setInspect({ kind: "run", id: run.id })}
                  >
                    Provenance
                  </button>
                </div>
              </article>
            ))}
            {!runs.length && (
              <Empty title="An unwritten lab notebook">
                Prompts, output, sampling, and requested/applied control events
                will be saved here.
              </Empty>
            )}
          </div>
        </Modal>
      )}
      {dialog === "import" && (
        <ImportDialog act={act} close={() => setDialog(null)} />
      )}
      {dialog === "mix" && (
        <SaveMixDialog
          act={act}
          modelId={currentModelId}
          axes={knownAxes}
          close={() => setDialog(null)}
        />
      )}
      {management && (
        <ManageRecordDialog
          record={management}
          act={act}
          deleteDisabled={frozen || inferenceBusy}
          close={() => setManagement(null)}
          saved={() => {
            if (management.operation === "delete") {
              if (management.kind === "vectors") {
                updateWorkspace({
                  axes: workspace.axes.filter(
                    (axis) => axis.vector_id !== management.id,
                  ),
                });
              }
              if (
                management.kind === "conversations" &&
                workspace.conversationId === management.id
              ) {
                updateWorkspace({ conversationId: "", prefixMessages: [] });
                setFocusedRunId("");
              }
              if (management.kind === "presets") setSelectedPresetId("");
            }
            setManagement(null);
          }}
        />
      )}
      {forkJobId && (
        <Modal
          title="Fork a completed stage."
          eyebrow="FACTORY CHECKPOINT / NEW BRANCH"
          wide
          onClose={() => setForkJobId(null)}
        >
          <p className="intro-text">
            Start a separate recipe and factory job from an edited, completed
            checkpoint. The original job continues unchanged. Only downstream
            work is regenerated.
          </p>
          <StageEditor
            key={forkJobId}
            job={state.jobs.find((job) => job.id === forkJobId)}
            targetModel={model}
            act={act}
            busy={busy}
            onVersion={(id) => {
              setForkJobId(null);
              setInspect({ kind: "recipe", id });
            }}
          />
        </Modal>
      )}
      {inspect && (
        <Inspector
          kind={inspect.kind}
          record={
            inspect.kind === "vector"
              ? state.vectors.find((v) => v.id === inspect.id)
              : inspect.kind === "recipe"
                ? state.recipes.find((r) => r.id === inspect.id)
                : state.runs.find((r) => r.id === inspect.id)
          }
          act={act}
          busy={busy}
          exportVector={exportVector}
          raw={workspace.raw}
          targetModel={model}
          close={() => setInspect(null)}
          onVersion={(id) => setInspect({ kind: "recipe", id })}
        />
      )}
    </div>
  );
}

function ConceptControl({
  axes,
  vector,
  index,
  frozen,
  change,
  remove,
}: {
  axes: Axis[];
  vector: Vector;
  index: number;
  frozen: boolean;
  change: (layer: number, percent: number) => void;
  remove: () => void;
}) {
  const color = ["mint", "lilac", "sand", "sky"][index % 4];
  return (
    <article className={`axis-control axis-${color}`}>
      <div className="axis-heading">
        <span className="axis-dot" />
        <h3 title={vector.name}>{vector.name}</h3>
        <button
          className="icon-button"
          onClick={remove}
          disabled={frozen}
          aria-label={`Remove ${vector.name} from mixer`}
        >
          <Icon name="close" size={14} />
        </button>
      </div>
      <div className="layer-controls">
        {axes.map((axis) => (
          <LayerControl
            key={axis.layer}
            axis={axis}
            vector={vector}
            change={(percent) => change(axis.layer, percent)}
          />
        ))}
      </div>
    </article>
  );
}
function LayerControl({
  axis,
  vector,
  change,
}: {
  axis: Axis;
  vector: Vector;
  change: (percent: number) => void;
}) {
  const [draft, setDraft] = useState(String(axis.percent));
  const [expanded, setExpanded] = useState(20);
  useEffect(() => setDraft(String(axis.percent)), [axis.percent]);
  const extent = Math.max(expanded, Math.abs(axis.percent));
  const label = `${vector.name} at layer ${axis.layer}`;
  const layer = vector.layers.find(
    (candidate) => candidate.layer === axis.layer,
  );
  const norm = layer
    ? ((Math.abs(axis.percent) / 100) * layer.residual_norm).toPrecision(3)
    : "—";
  return (
    <div
      className={`layer-control ${axis.percent !== 0 ? "nonzero" : ""}`}
      data-layer={axis.layer}
    >
      <div className="layer-control-heading">
        <span
          className="layer-label"
          title={`Residual contribution ‖Δh‖ ${norm}`}
        >
          L{axis.layer}
          {axis.layer === vector.selected_layer && (
            <span
              className="selected-layer"
              title="Highest diagnostic separation; not necessarily the best steering behavior"
              aria-label="Selected diagnostic layer"
            >
              ✦
            </span>
          )}
        </span>
        <button
          className="layer-range"
          onClick={() => setExpanded(extent * 2)}
          disabled={!Number.isFinite(extent * 2)}
          aria-label={`Expand range for ${label}`}
          title="Expand slider range; numeric entry is unrestricted"
        >
          ±{extent}%
        </button>
        <button
          className="layer-zero"
          onClick={() => change(0)}
          aria-label={`Zero ${label}`}
          title="Reset this layer to zero"
        >
          0
        </button>
        <label className="coefficient-input">
          <input
            aria-label={`Coefficient for ${label}`}
            title="Any finite percentage, including beyond ±20%"
            type="number"
            step="any"
            value={draft}
            onChange={(event) => {
              setDraft(event.target.value);
              const value = finitePercent(event.target.value);
              if (value !== null) change(value);
            }}
            onBlur={() => {
              if (finitePercent(draft) === null) setDraft(String(axis.percent));
            }}
          />
          <span>%</span>
        </label>
      </div>
      <div className="range-wrap">
        <span className="range-zero" />
        <input
          className="axis-slider"
          type="range"
          min={-extent}
          max={extent}
          step="0.1"
          value={axis.percent}
          aria-label={`Steering slider for ${label}`}
          onChange={(event) => change(Number(event.target.value))}
        />
      </div>
    </div>
  );
}
function RunCard({
  run,
  output,
  state,
  baseline,
  baselineOutput,
  frozen,
  compare,
  duplicate,
  inspect,
}: {
  run: Run;
  output: string;
  state: State;
  baseline?: Run;
  baselineOutput: (run: Run) => string;
  frozen: boolean;
  compare: () => void;
  duplicate: () => void;
  inspect: () => void;
}) {
  const prompt = [...run.messages]
    .reverse()
    .find((message) => message.role === "user")?.content;
  const active = activeStatuses.has(run.status);
  return (
    <article className="run-card">
      <div className="message-label">
        <span>YOU</span>
        <time>{dateLabel(run.created_at)}</time>
      </div>
      <div className="user-message">{prompt}</div>
      <div className="run-response-heading">
        <div className="message-label">
          <span className="response-star">✳</span>
          <span>
            {state.models.find((model) => model.id === run.model_id)?.name ??
              "LOCAL MODEL"}
          </span>
        </div>
        <Badge tone={active ? "mint" : run.status === "failed" ? "amber" : ""}>
          {active ? "generating" : run.status}
        </Badge>
      </div>
      <div className="run-axis-tags">
        {run.axes.length ? (
          [...new Set(run.axes.map((axis) => axis.vector_id))].map((id) => (
            <span
              key={id}
              title={state.vectors.find((vector) => vector.id === id)?.name}
            >
              {state.vectors.find((vector) => vector.id === id)?.name ??
                shortId(id)}{" "}
              <b>
                initial{" "}
                {run.axes
                  .filter((axis) => axis.vector_id === id)
                  .map(
                    (axis) =>
                      `L${axis.layer} ${axis.percent > 0 ? "+" : ""}${axis.percent}%`,
                  )
                  .join(" · ")}
              </b>
            </span>
          ))
        ) : (
          <span>Unsteered baseline</span>
        )}
      </div>
      {!!run.context_rollovers?.length && (
        <p className="rollover-note">
          Context window rolled {run.context_rollovers.length} time
          {run.context_rollovers.length === 1 ? "" : "s"}; older context was
          omitted. Saved output is unchanged.
        </p>
      )}
      {!!run.tool_calls?.length && (
        <details className="run-tool-history">
          <summary>
            {run.tool_calls.length} self-adjustment tool call
            {run.tool_calls.length === 1 ? "" : "s"}
          </summary>
          {run.tool_calls.map((call) => (
            <div key={call.id}>
              <small>After token {call.after_token_index}</small>
              <pre>{call.input}</pre>
              <pre>{JSON.stringify(call.result, null, 2)}</pre>
            </div>
          ))}
        </details>
      )}
      <div className={`responses ${baseline ? "comparing" : ""}`}>
        <div className="response-block">
          {baseline && <span className="comparison-label">WITH MIX</span>}
          <div className={`response-text ${active ? "generating" : ""}`}>
            {output ||
              (active ? "Waiting for the first token…" : "No output recorded.")}
          </div>
          {run.error && <div className="error-box">{run.error}</div>}
        </div>
        {baseline && (
          <div className="response-block baseline">
            <div className="baseline-heading">
              <span className="comparison-label">ZERO / BASELINE</span>
              <Badge>{baseline.status}</Badge>
            </div>
            <div
              className={`response-text ${activeStatuses.has(baseline.status) ? "generating" : ""}`}
            >
              {baselineOutput(baseline) || "Waiting for baseline…"}
            </div>
            {baseline.error && (
              <div className="error-box">{baseline.error}</div>
            )}
          </div>
        )}
      </div>
      {run.mix_diagnostics && (
        <MixtureDiagnostics
          diagnostics={run.mix_diagnostics}
          vectors={state.vectors}
        />
      )}
      {!active && (
        <div className="response-actions">
          <button className="quiet-button" onClick={compare} disabled={frozen}>
            <Icon name="sliders" size={13} />
            {baseline ? "Rerun baseline" : "Compare baseline"}
          </button>
          <button
            className="quiet-button"
            onClick={duplicate}
            disabled={frozen}
            title="Fork a new chat through this reply"
          >
            <Icon name="copy" size={13} />
            Duplicate
          </button>
          <button className="quiet-button" onClick={inspect}>
            <Icon name="code" size={13} />
            Provenance
          </button>
        </div>
      )}
      {baseline && (
        <p className="baseline-note">
          Same transcript, seed, and sampling. Steering disabled. Not a rewrite
          of prior conversation.
        </p>
      )}
    </article>
  );
}
function JobCard({
  job,
  act,
  inspect,
  duplicate,
  fork,
}: {
  job: Job;
  act: Act;
  inspect?: () => void;
  duplicate?: () => void;
  fork?: () => void;
}) {
  const active = activeStatuses.has(job.status);
  const inference =
    !!job.run_id || ["generate", "generation"].includes(job.kind);
  return (
    <article className="job-card">
      <div className="job-main">
        <span
          className={`status-dot ${active ? "green" : ["failed", "interrupted"].includes(job.status) ? "amber" : ""}`}
        />
        <div>
          <strong>{job.kind.replaceAll("_", " ")}</strong>
          <span>{job.stage?.replaceAll("_", " ") ?? shortId(job.id)}</span>
        </div>
        <Badge>{job.status}</Badge>
        <time>{dateLabel(job.created_at)}</time>
      </div>
      {active && (
        <progress
          max="1"
          value={Math.max(0, Math.min(1, job.progress ?? 0))}
          aria-label={`${job.kind} progress`}
        />
      )}
      {job.error && <p className="error-box">{job.error}</p>}
      <div className="job-actions">
        {!job.attention_dismissed &&
          ["failed", "interrupted"].includes(job.status) && (
            <button
              className="quiet-button"
              title="Clear the attention badge; keep the job and its outputs"
              onClick={() =>
                void act("dismiss_job_attention", { job_id: job.id })
              }
            >
              Dismiss
            </button>
          )}
        {active && (
          <button
            className="quiet-button"
            onClick={() => void act("cancel_job", { job_id: job.id })}
          >
            Cancel
          </button>
        )}
        {!inference &&
          ["failed", "interrupted", "cancelled"].includes(job.status) && (
            <button
              className="quiet-button"
              onClick={() =>
                void act(
                  "retry_job",
                  { job_id: job.id },
                  "Retry requested. Completed stages are retained.",
                )
              }
            >
              Resume / retry
            </button>
          )}
        {inference && !active && duplicate && (
          <button className="quiet-button" onClick={duplicate}>
            Fork chat <Icon name="copy" size={13} />
          </button>
        )}
        {inspect && (
          <button className="quiet-button" onClick={inspect}>
            {job.recipe_id ? "Open recipe" : "Open run"}{" "}
            <Icon name="arrow" size={13} />
          </button>
        )}
        {fork && (
          <button className="quiet-button" onClick={fork}>
            Fork completed stage <Icon name="copy" size={13} />
          </button>
        )}
        {job.details !== undefined && (
          <details>
            <summary>Worker details</summary>
            <pre>{jsonText(job.details)}</pre>
          </details>
        )}
      </div>
    </article>
  );
}

function ModelDialog({
  state,
  selected,
  select,
  act,
  busy,
  frozen,
  close,
}: {
  state: State;
  selected: string;
  select: (id: string) => void;
  act: Act;
  busy: Set<string>;
  frozen: boolean;
  close: () => void;
}) {
  const [tab, setTab] = useState<"library" | "local" | "hf">(
    state.models.length ? "library" : "hf",
  );
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [repo, setRepo] = useState("prism-ml/Ternary-Bonsai-2-27B-gguf");
  const [listing, setListing] = useState<HfListing | null>(null);
  const [file, setFile] = useState("");
  const [context, setContext] = useState(8192);
  const [gpuLayers, setGpuLayers] = useState(99);
  const browse = async () => {
    const result = await act<HfListing>("browse_hf", { repo: repo.trim() });
    if (result) {
      setListing(result);
      setFile(
        result.files.find((file) => /PQ2_0\.gguf$/i.test(file.name))?.name ??
          result.files[0]?.name ??
          "",
      );
    }
  };
  const importModel = async (e: FormEvent) => {
    e.preventDefault();
    const result = await act<{ id: string }>(
      "import_model",
      { path: path.trim(), ...(name.trim() ? { name: name.trim() } : {}) },
      "Model fingerprinted and added by reference.",
    );
    if (result) {
      if (!frozen) select(result.id);
      setTab("library");
    }
  };
  return (
    <Modal
      title="Models, on your terms."
      eyebrow="MODEL LIBRARY"
      wide
      onClose={close}
    >
      <div className="segmented dialog-tabs">
        <button
          className={tab === "library" ? "selected" : ""}
          onClick={() => setTab("library")}
        >
          Your models <span>{state.models.length}</span>
        </button>
        <button
          className={tab === "hf" ? "selected" : ""}
          onClick={() => setTab("hf")}
        >
          Hugging Face
        </button>
        <button
          className={tab === "local" ? "selected" : ""}
          onClick={() => setTab("local")}
        >
          Import local
        </button>
      </div>
      {tab === "library" && (
        <>
          <div className="model-list">
            {state.models.map((model) => (
              <article
                className={`model-card ${model.id === selected ? "chosen" : ""}`}
                key={model.id}
              >
                <div className="model-card-top">
                  <div className="model-glyph">◈</div>
                  <div>
                    <h3>{model.name}</h3>
                    <span>
                      {formatSize(model.size_bytes)} ·{" "}
                      {shortId(model.fingerprint)}
                    </span>
                  </div>
                  {state.engine.model_id === model.id && (
                    <Badge tone="mint">{state.engine.status}</Badge>
                  )}
                </div>
                <p className="path-text">{model.path}</p>
                <div className="model-card-actions">
                  <button
                    className="button secondary"
                    disabled={frozen || model.id === selected}
                    onClick={() => select(model.id)}
                  >
                    {model.id === selected ? "Selected" : "Select model"}
                  </button>
                  {state.engine.model_id === model.id ? (
                    <button
                      className="quiet-button"
                      disabled={frozen || busy.has("unload_model")}
                      onClick={() =>
                        void act("unload_model", {}, "Engine unloaded.")
                      }
                    >
                      Unload
                    </button>
                  ) : (
                    <button
                      className="button primary"
                      disabled={frozen || busy.has("load_model")}
                      onClick={async () => {
                        select(model.id);
                        await act(
                          "load_model",
                          {
                            model_id: model.id,
                            context,
                            gpu_layers: gpuLayers,
                          },
                          "Load requested.",
                        );
                      }}
                    >
                      {busy.has("load_model") ? "Loading…" : "Load model"}
                      <Icon name="bolt" size={14} />
                    </button>
                  )}
                </div>
              </article>
            ))}
            {!state.models.length && (
              <Empty title="Bring a brain">
                Download the Bonsai preset or point the app at an existing GGUF
                file.
              </Empty>
            )}
          </div>
          <details className="advanced-settings">
            <summary>Engine settings</summary>
            <div className="two-columns">
              <label className="field">
                Context tokens
                <input
                  type="number"
                  value={context}
                  min="128"
                  max="131072"
                  step="128"
                  onChange={(e) => setContext(Number(e.target.value))}
                />
              </label>
              <label className="field">
                GPU layers
                <input
                  type="number"
                  value={gpuLayers}
                  min="0"
                  max="999"
                  onChange={(e) => setGpuLayers(Number(e.target.value))}
                />
              </label>
            </div>
            <p className="muted">
              Metal · batch 512 · microbatch 128 · F16 cache. One local
              inference job at a time.
            </p>
          </details>
          {state.engine.error && (
            <div className="error-box">{state.engine.error}</div>
          )}
        </>
      )}
      {tab === "local" && (
        <form onSubmit={(e) => void importModel(e)}>
          <p className="intro-text">
            Already have a GGUF? Keep it where it is. The app fingerprints it
            without making a second enormous copy.
          </p>
          <label className="field">
            Absolute file path
            <input
              autoFocus
              placeholder="/Users/you/Models/model.gguf"
              title="In Finder, hold Option and choose Copy as Pathname."
              value={path}
              onChange={(e) => setPath(e.target.value)}
              required
            />
          </label>
          <label className="field">
            Display name <span className="dim">optional</span>
            <input
              placeholder="A recognizable name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <div className="dialog-actions">
            <button
              className="button primary"
              disabled={!path.trim() || busy.has("import_model")}
            >
              {busy.has("import_model")
                ? "Fingerprinting…"
                : "Import by reference"}
              <Icon name="arrow" size={15} />
            </button>
          </div>
        </form>
      )}
      {tab === "hf" && (
        <>
          <div className="preset-feature">
            <span className="eyebrow">START HERE</span>
            <h3>
              Bonsai 2 <span>PQ2_0</span>
            </h3>
            <p>
              Apple Silicon, meet the reference model.
              <br />
              Resolved to an immutable Hugging Face revision before downloading.
            </p>
            <button
              className="button secondary"
              disabled={busy.has("download_model")}
              onClick={() =>
                void act(
                  "download_model",
                  {
                    repo: "prism-ml/Ternary-Bonsai-2-27B-gguf",
                    file: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
                  },
                  "Bonsai download started. Progress and resume controls are in Jobs.",
                )
              }
            >
              {busy.has("download_model")
                ? "Starting download…"
                : "Download Bonsai preset"}
              <Icon name="download" size={15} />
            </button>
          </div>
          <div className="divider">
            <span>OR CHOOSE A REPOSITORY</span>
          </div>
          <form
            className="inline-form"
            onSubmit={(e) => {
              e.preventDefault();
              void browse();
            }}
          >
            <label className="field">
              Hugging Face repository
              <input
                placeholder="owner/repository"
                value={repo}
                onChange={(e) => {
                  setRepo(e.target.value);
                  setListing(null);
                }}
                required
              />
            </label>
            <button
              className="button secondary"
              disabled={!repo.trim() || busy.has("browse_hf")}
            >
              {busy.has("browse_hf") ? "Looking…" : "Browse files"}
            </button>
          </form>
          {listing && (
            <div className="hf-results">
              <div className="hf-revision">
                <span>IMMUTABLE REVISION</span>
                <code>{listing.revision}</code>
              </div>
              <label className="field">
                GGUF file
                <select value={file} onChange={(e) => setFile(e.target.value)}>
                  {listing.files.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name} · {formatSize(file.size_bytes)}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="button primary"
                disabled={!file || busy.has("download_model")}
                onClick={() =>
                  void act(
                    "download_model",
                    { repo: listing.repo, file, revision: listing.revision },
                    "Download started. Partial files are kept for explicit resume.",
                  )
                }
              >
                Download selected file
                <Icon name="download" size={15} />
              </button>
              {!listing.files.length && (
                <p className="muted small">
                  No GGUF files were found in this repository.
                </p>
              )}
            </div>
          )}
          <p className="muted small">
            Model files are separate from the app. Downloads can be resumed;
            completed files are fingerprinted.
          </p>
        </>
      )}
    </Modal>
  );
}

function ConceptDialog({
  state,
  modelId,
  raw,
  act,
  busy,
  close,
  created,
}: {
  state: State;
  modelId: string;
  raw: boolean;
  act: Act;
  busy: Set<string>;
  close: () => void;
  created: () => void;
}) {
  const [concept, setConcept] = useState("");
  const [roles, setRoles] = useState<Roles>({
    designer: "",
    writers: ["", "", ""],
    reviewer: "",
  });
  const [editRoles, setEditRoles] = useState(false);
  const [rawMode, setRawMode] = useState(raw);
  const [extractionMethod, setExtractionMethod] = useState<
    "paper" | "completion"
  >("paper");
  const [readoutSuffix, setReadoutSuffix] = useState("I feel:");
  const [previewMode, setPreviewMode] = useState("none");
  const discovered = useRef(false);
  const manuallyAssigned = useRef(false);
  const modelList = state.codex.models;
  useEffect(() => {
    if (!modelList.length && !discovered.current) {
      discovered.current = true;
      void act("discover_codex");
    }
  }, [act, modelList.length]);
  useEffect(() => {
    if (manuallyAssigned.current || !modelList.length) return;
    const sorted = [...modelList].sort(
      (a, b) =>
        Number(!!(b.isDefault ?? b.is_default)) -
        Number(!!(a.isDefault ?? a.is_default)),
    );
    const ids = [...new Set(sorted.map((model) => model.id))];
    setRoles(
      state.codex.assignments ?? {
        designer: ids[0],
        writers: [ids[0], ids[1 % ids.length], ids[2 % ids.length]],
        reviewer: ids[Math.min(2, ids.length - 1)],
      },
    );
  }, [modelList, state.codex.assignments]);
  const roleSelect = (
    label: string,
    value: string,
    update: (value: string) => void,
  ) => (
    <label className="field" key={label}>
      {label}
      <select
        value={value}
        onChange={(e) => {
          manuallyAssigned.current = true;
          update(e.target.value);
        }}
      >
        {modelList.map((model) => (
          <option value={model.id} key={model.id}>
            {labelModel(model)}
            {model.isDefault || model.is_default ? " · account default" : ""}
          </option>
        ))}
      </select>
    </label>
  );
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const result = await act(
      "create_concept",
      {
        concept: concept.trim(),
        model_id: modelId,
        raw: rawMode,
        extraction: {
          method: extractionMethod,
          readout_suffix: extractionMethod === "paper" ? readoutSuffix : "",
        },
        preview_mode: previewMode,
        ...(roles.designer ? { roles } : {}),
      },
      "Concept factory started. Each stage is saved as it completes.",
    );
    if (result) created();
  };
  return (
    <Modal
      title="A direction begins with a contrast."
      eyebrow="CONCEPT FACTORY"
      wide
      onClose={close}
    >
      <form onSubmit={(e) => void submit(e)}>
        <p className="intro-text">
          Describe the quality you want to explore. Agents design and challenge
          the contrast; your local model supplies the activations.
        </p>
        <label className="field">
          Concept
          <textarea
            autoFocus
            rows={4}
            placeholder="e.g. dry, understated wit rather than earnest literalness; keep factual content and helpfulness matched"
            value={concept}
            onChange={(e) => setConcept(e.target.value)}
            required
          />
        </label>
        <div className="factory-flow">
          <span>Design</span>
          <Icon name="chevron" size={12} />
          <span>3 writers</span>
          <Icon name="chevron" size={12} />
          <span>Review</span>
          <Icon name="chevron" size={12} />
          <span>Extract</span>
          <Icon name="chevron" size={12} />
          <span>Preview</span>
        </div>
        <label className="field">
          Extraction method
          <select
            value={extractionMethod}
            onChange={(e) =>
              setExtractionMethod(e.target.value as "paper" | "completion")
            }
          >
            <option value="paper">
              Paper-style · raw readout + control PCA (default)
            </option>
            <option value="completion">
              Completion contrast · assistant response style
            </option>
          </select>
        </label>
        {extractionMethod === "paper" && (
          <label className="field">
            Fixed readout suffix
            <input
              value={readoutSuffix}
              maxLength={256}
              required
              onChange={(e) => setReadoutSuffix(e.target.value)}
            />
            <span>
              The same suffix follows every raw statement. No chat template. “I
              feel:” matches the pain paper; choose a suitable probe for other
              concepts.
            </span>
          </label>
        )}
        <div className="factory-explanation">
          <div>
            <strong>96 matched pairs</strong>
            <span>Target, not a fiction quota.</span>
          </div>
          <div>
            <strong>5 candidate layers</strong>
            <span>
              {extractionMethod === "paper"
                ? "Grouped cross-validation + 50% control PCA."
                : "Selected by held-out separation."}
            </span>
          </div>
          <div>
            <strong>
              {previewMode === "none"
                ? "Previews disabled"
                : previewMode === "negative_only"
                  ? "0 / −1 / −2% previews"
                  : "− / 0 / + previews"}
            </strong>
            <span>Weak and weird results stay.</span>
          </div>
        </div>
        <div className="role-heading">
          <div>
            <span
              className={`status-dot ${modelList.length ? "green" : "amber"}`}
            />
            <strong>
              {busy.has("discover_codex")
                ? "Discovering your Codex models…"
                : modelList.length
                  ? `${new Set(modelList.map((m) => m.id)).size} Codex models available`
                  : "Codex discovery needed"}
            </strong>
          </div>
          <button
            type="button"
            className="quiet-button"
            onClick={() => setEditRoles(!editRoles)}
          >
            {editRoles ? "Hide roles" : "Edit role assignments"}
          </button>
        </div>
        {state.codex.error && (
          <div className="error-box">{state.codex.error}</div>
        )}
        {!modelList.length && (
          <button
            type="button"
            className="button secondary"
            disabled={busy.has("discover_codex")}
            onClick={() => void act("discover_codex")}
          >
            Discover / retry Codex
          </button>
        )}
        {editRoles && modelList.length > 0 && (
          <div className="role-grid">
            {roleSelect("Designer", roles.designer, (designer) =>
              setRoles((current) => ({ ...current, designer })),
            )}
            {roles.writers.map((writer, index) =>
              roleSelect(`Writer ${index + 1}`, writer, (value) =>
                setRoles((current) => ({
                  ...current,
                  writers: current.writers.map((model, i) =>
                    i === index ? value : model,
                  ),
                })),
              ),
            )}
            {roleSelect("Reviewer", roles.reviewer, (reviewer) =>
              setRoles((current) => ({ ...current, reviewer })),
            )}
          </div>
        )}
        <label className="field">
          Automatic previews
          <select
            value={previewMode}
            onChange={(e) => setPreviewMode(e.target.value)}
          >
            <option value="none">None · extract without generating</option>
            <option value="negative_only">Negative only · 0%, −1%, −2%</option>
            <option value="standard">Standard · −10%, 0%, +10%</option>
          </select>
        </label>
        {extractionMethod === "completion" && (
          <label className="check-label">
            <input
              type="checkbox"
              checked={rawMode}
              onChange={(e) => setRawMode(e.target.checked)}
            />
            Extract in raw-completion mode{" "}
            <span>Only for template-free models.</span>
          </label>
        )}
        <div className="dialog-actions">
          <button
            className="button primary"
            title="Generates examples through Codex; local chats stay local."
            disabled={
              !concept.trim() ||
              !modelId ||
              busy.has("create_concept") ||
              busy.has("discover_codex")
            }
          >
            {busy.has("create_concept") ? "Starting…" : "Create concept"}
            <Icon name="arrow" size={15} />
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ImportDialog({ act, close }: { act: Act; close: () => void }) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    let bundle: unknown;
    try {
      bundle = JSON.parse(text);
    } catch (error) {
      setError(`Invalid JSON: ${errorMessage(error)}`);
      return;
    }
    setPending(true);
    const result = await act(
      "import_vector",
      { bundle },
      "Vector bundle validated and imported.",
    );
    setPending(false);
    if (result) close();
  };
  return (
    <Modal
      title="Bring a direction with receipts."
      eyebrow="IMPORT VECTOR"
      onClose={close}
    >
      <form onSubmit={(e) => void submit(e)}>
        <p className="intro-text">
          Import a Torment Nexus vector bundle. Model binding, tensor shapes,
          and checksums are verified before it joins the library.
        </p>
        <label className="file-picker">
          Choose JSON bundle
          <input
            type="file"
            accept=".json,application/json"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (file) {
                if (file.size > 256 * 1024 * 1024) {
                  setError("Bundle exceeds the 256 MiB browser import limit.");
                  return;
                }
                try {
                  setText(await file.text());
                  setError("");
                } catch (error) {
                  setError(errorMessage(error));
                }
              }
            }}
          />
        </label>
        <label className="field">
          Or paste bundle JSON
          <textarea
            className="code-editor"
            rows={10}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={'{\n  "manifest": …\n}'}
          />
        </label>
        {error && <div className="error-box">{error}</div>}
        <div className="dialog-actions">
          <button className="button primary" disabled={pending || !text.trim()}>
            {pending ? "Validating…" : "Validate & import"}
            <Icon name="arrow" size={15} />
          </button>
        </div>
      </form>
    </Modal>
  );
}
function ManageRecordDialog({
  record,
  act,
  close,
  saved,
  deleteDisabled,
}: {
  record: Management;
  act: Act;
  close: () => void;
  saved: () => void;
  deleteDisabled: boolean;
}) {
  const [name, setName] = useState(record.name);
  const [pending, setPending] = useState(false);
  const deleting = record.operation === "delete";
  const label = recordLabels[record.kind];
  return (
    <Modal
      title={`${deleting ? "Delete" : "Rename"} ${label}`}
      onClose={() => {
        if (!pending) close();
      }}
    >
      <form
        onSubmit={async (event) => {
          event.preventDefault();
          if (pending || (deleting && deleteDisabled)) return;
          setPending(true);
          const result = await act(
            deleting ? "delete_record" : "rename_record",
            {
              kind: record.kind,
              id: record.id,
              ...(!deleting ? { name: name.trim() } : {}),
            },
            `${label[0].toUpperCase() + label.slice(1)} ${deleting ? "deleted" : "renamed"}.`,
          );
          setPending(false);
          if (result) saved();
        }}
      >
        {deleting ? (
          <>
            <p className="intro-text">
              Delete <strong>{record.name}</strong>?
            </p>
            <p className="muted">
              This removes it from the{" "}
              {record.kind === "conversations"
                ? "conversation list and its turns from run history"
                : "library"}
              . Stored artifacts and historical references are retained for
              provenance; this is not a disk wipe.
            </p>
          </>
        ) : (
          <>
            <label className="field">
              New name
              <input
                autoFocus
                required
                maxLength={200}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            {record.kind === "recipes" && (
              <p className="muted">
                Only the display name changes. The original concept and
                generation instructions stay intact.
              </p>
            )}
          </>
        )}
        <div className="dialog-actions">
          <button
            type="button"
            className="button secondary"
            onClick={close}
            disabled={pending}
          >
            Cancel
          </button>
          <button
            className={`button ${deleting ? "delete-button" : "primary"}`}
            disabled={pending || (deleting ? deleteDisabled : !name.trim())}
          >
            {pending ? "Saving…" : deleting ? "Delete" : "Save name"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function SaveMixDialog({
  act,
  modelId,
  axes,
  close,
}: {
  act: Act;
  modelId: string;
  axes: Axis[];
  close: () => void;
}) {
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  return (
    <Modal title="Keep this combination." eyebrow="SAVE MIX" onClose={close}>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          setPending(true);
          const result = await act(
            "save_mix",
            { name: name.trim(), model_id: modelId, axes },
            "Mix saved.",
          );
          setPending(false);
          if (result) close();
        }}
      >
        <label className="field">
          Mix name
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="A recognizable kind of strange"
            required
          />
        </label>
        <p className="muted">
          Saves {axes.length} vector{axes.length !== 1 ? "s" : ""}, their
          layers, and the current coefficients. Vector versions remain
          immutable.
        </p>
        <div className="dialog-actions">
          <button className="button primary" disabled={!name.trim() || pending}>
            Save mix
            <Icon name="archive" size={15} />
          </button>
        </div>
      </form>
    </Modal>
  );
}

function Inspector({
  kind,
  record,
  act,
  busy,
  exportVector,
  raw,
  targetModel,
  close,
  onVersion,
}: {
  kind: "vector" | "recipe" | "run";
  record?: Vector | Recipe | Run;
  act: Act;
  busy: Set<string>;
  exportVector: (vector: Vector) => Promise<void>;
  raw: boolean;
  targetModel?: Model;
  close: () => void;
  onVersion: (id: string) => void;
}) {
  const [tab, setTab] = useState("overview");
  const [artifact, setArtifact] = useState<unknown>(null);
  const [artifactLoading, setArtifactLoading] = useState(false);
  const recipe = kind === "recipe" ? (record as Recipe | undefined) : undefined;
  const vector = kind === "vector" ? (record as Vector | undefined) : undefined;
  const run = kind === "run" ? (record as Run | undefined) : undefined;
  const manifestHash = vector?.manifest_hash ?? run?.manifest_hash;
  useEffect(() => {
    setArtifact(null);
  }, [record?.id]);
  const viewManifest = async () => {
    setTab("manifest");
    if (!manifestHash || artifact !== null) return;
    setArtifactLoading(true);
    const value = await act("artifact", { hash: manifestHash });
    setArtifact(value);
    setArtifactLoading(false);
  };
  return (
    <Modal
      title={
        vector?.name ??
        recipe?.display_name ??
        recipe?.concept ??
        (run ? `Run ${shortId(run.id)}` : "Artifact unavailable")
      }
      eyebrow={`${kind.toUpperCase()} / PROVENANCE`}
      wide
      onClose={close}
    >
      {!record ? (
        <div className="error-box">
          This record is not in the current snapshot. Close this inspector and
          refresh the connection.
        </div>
      ) : (
        <>
          <div className="inspector-id">
            <code>{record.id}</code>
            {recipe && <Badge>v{recipe.version}</Badge>}
            <span>{dateLabel(record.created_at)}</span>
          </div>
          <div className="segmented dialog-tabs">
            <button
              className={tab === "overview" ? "selected" : ""}
              onClick={() => setTab("overview")}
            >
              Overview
            </button>
            {recipe && (
              <button
                className={tab === "edit" ? "selected" : ""}
                onClick={() => setTab("edit")}
              >
                Edit stages
              </button>
            )}
            {vector && (
              <button
                className={tab === "previews" ? "selected" : ""}
                onClick={() => setTab("previews")}
              >
                Previews
              </button>
            )}
            {manifestHash && (
              <button
                className={tab === "manifest" ? "selected" : ""}
                onClick={() => void viewManifest()}
              >
                Manifest
              </button>
            )}
            <button
              className={tab === "json" ? "selected" : ""}
              onClick={() => setTab("json")}
            >
              Record JSON
            </button>
          </div>
          {tab === "overview" && (
            <>
              {vector && (
                <>
                  <div className="stat-grid">
                    <div>
                      <span>MODEL FINGERPRINT</span>
                      <strong className="mono">
                        {shortId(vector.model_fingerprint)}
                      </strong>
                    </div>
                    <div>
                      <span>SELECTED LAYER</span>
                      <strong>{vector.selected_layer}</strong>
                    </div>
                    <div>
                      <span>EXTRACTED LAYERS</span>
                      <strong>{vector.layers.length}</strong>
                    </div>
                  </div>
                  <p className="intro-text">
                    {vector.extraction?.method === "paper"
                      ? `Raw fixed readout (${JSON.stringify(vector.extraction.readout_suffix)}), positive minus pooled controls, with control PCA removing 50% of variance. Family-grouped CV selects the layer, then all pairs fit the final direction. The selection score is not an independent test.`
                      : "Last-assistant-content-token difference of means. Positive completions minus negative completions. Highest held-out projection AUC selects a layer."}{" "}
                    Separation does not establish steering quality.
                  </p>
                  <table className="layer-table">
                    <thead>
                      <tr>
                        <th>Graph layer</th>
                        <th>
                          {vector.extraction?.method === "paper"
                            ? "Grouped CV AUC"
                            : "Separation AUC"}
                        </th>
                        <th>Median residual L2</th>
                        <th>Width</th>
                      </tr>
                    </thead>
                    <tbody>
                      {vector.layers.map((layer) => (
                        <tr key={layer.layer}>
                          <td>
                            {layer.layer}
                            {layer.layer === vector.selected_layer && (
                              <span className="selected-layer">selected</span>
                            )}
                          </td>
                          <td>{layer.auc?.toFixed(4) ?? "—"}</td>
                          <td>{layer.residual_norm?.toPrecision(5) ?? "—"}</td>
                          <td>{layer.width}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <Warnings warnings={vector.warnings} />
                  <div className="dialog-actions">
                    <button
                      className="button secondary"
                      onClick={() => void exportVector(vector)}
                      disabled={busy.has("export_vector")}
                    >
                      <Icon name="download" size={15} />
                      Export complete bundle
                    </button>
                  </div>
                </>
              )}
              {recipe && (
                <>
                  <div className="stat-grid">
                    <div>
                      <span>IMMUTABLE VERSION</span>
                      <strong>v{recipe.version}</strong>
                    </div>
                    <div>
                      <span>PARENT</span>
                      <strong className="mono">
                        {shortId(recipe.parent_id)}
                      </strong>
                    </div>
                    <div>
                      <span>EXTRACTION TARGET</span>
                      <strong>{targetModel?.name ?? "Select a model"}</strong>
                    </div>
                  </div>
                  <Warnings warnings={recipe.warnings} />
                  <h3 className="section-title">Stages</h3>
                  <pre className="json-preview">{jsonText(recipe.stages)}</pre>
                  <h3 className="section-title">Agent roles</h3>
                  <pre className="json-preview">{jsonText(recipe.roles)}</pre>
                  <p className="intro-text">
                    Saved method:{" "}
                    {recipe.extraction?.method === "paper"
                      ? `paper-style, fixed readout ${JSON.stringify(recipe.extraction.readout_suffix)}`
                      : "completion contrast (historical recipes retain this method)"}
                    .
                  </p>
                  <div className="dialog-actions">
                    <button
                      className="button secondary"
                      onClick={() => {
                        setTab("edit");
                      }}
                    >
                      Inspect / edit dataset
                    </button>
                    <button
                      className="button primary"
                      disabled={
                        !targetModel ||
                        recipe.draft === true ||
                        busy.has("extract_recipe")
                      }
                      onClick={() =>
                        void act(
                          "extract_recipe",
                          {
                            recipe_id: recipe.id,
                            model_id: targetModel!.id,
                          },
                          `Extraction requested for ${targetModel!.name}; saved examples are reused.`,
                        )
                      }
                    >
                      Extract for selected model
                      <Icon name="bolt" size={14} />
                    </button>
                  </div>
                  <RecipeExtractionSettings
                    key={recipe.id}
                    recipe={recipe}
                    act={act}
                    busy={busy}
                    fallbackRaw={raw}
                    targetModel={targetModel}
                  />
                </>
              )}
              {run && (
                <>
                  <div className="stat-grid">
                    <div>
                      <span>STATUS</span>
                      <strong>{run.status}</strong>
                    </div>
                    <div>
                      <span>SEED</span>
                      <strong>{run.sampling.seed}</strong>
                    </div>
                    <div>
                      <span>INITIAL AXES</span>
                      <strong>{run.axes.length}</strong>
                    </div>
                  </div>
                  {run.error && <div className="error-box">{run.error}</div>}
                  <h3 className="section-title">
                    Requested vs. applied controls
                  </h3>
                  <p className="muted">
                    The engine acknowledges the first affected output-token
                    index. Requested changes are not treated as applied until
                    acknowledged.
                  </p>
                  <div className="two-columns">
                    <div>
                      <span className="eyebrow">REQUESTED</span>
                      <pre className="json-preview">
                        {jsonText(run.requested_controls)}
                      </pre>
                    </div>
                    <div>
                      <span className="eyebrow">APPLIED</span>
                      <pre className="json-preview">
                        {jsonText(run.applied_controls)}
                      </pre>
                    </div>
                  </div>
                  {run.mix_diagnostics && (
                    <MixtureDiagnostics diagnostics={run.mix_diagnostics} />
                  )}
                  <h3 className="section-title">Sampling</h3>
                  <pre className="json-preview">{jsonText(run.sampling)}</pre>
                  <div className="dialog-actions">
                    <button
                      className="button secondary"
                      onClick={() => downloadJson(`run-${run.id}.json`, run)}
                    >
                      <Icon name="download" size={15} />
                      Export run record
                    </button>
                  </div>
                </>
              )}
            </>
          )}
          {tab === "edit" && recipe && (
            <StageEditor
              key={recipe.id}
              recipe={recipe}
              targetModel={targetModel}
              act={act}
              busy={busy}
              onVersion={onVersion}
            />
          )}
          {tab === "previews" && vector && (
            <>
              <p className="intro-text">
                {vector.preview_mode === "none"
                  ? "Automatic previews were disabled for this recipe. Extraction did not generate preview responses."
                  : vector.preview_mode === "negative_only"
                    ? "Three neutral prompts at 0%, −1%, and −2%. Degeneration and weak effects are retained."
                    : "Three neutral prompts, negative / zero / positive intervention. Degeneration and weak effects are retained, not filtered away."}
              </p>
              {vector.preview_mode !== "none" && (
                <PreviewGallery previews={vector.previews} />
              )}
            </>
          )}
          {tab === "manifest" && (
            <pre className="json-preview">
              {artifactLoading
                ? "Loading verified artifact…"
                : jsonText(artifact)}
            </pre>
          )}
          {tab === "json" && (
            <pre className="json-preview">{jsonText(record)}</pre>
          )}
        </>
      )}
    </Modal>
  );
}
function RecipeExtractionSettings({
  recipe,
  act,
  busy,
  fallbackRaw,
  targetModel,
}: {
  recipe: Recipe;
  act: Act;
  busy: Set<string>;
  fallbackRaw: boolean;
  targetModel?: Model;
}) {
  const [method, setMethod] = useState<"paper" | "completion">(
    recipe.extraction?.method ?? "completion",
  );
  const [suffix, setSuffix] = useState(
    recipe.extraction?.readout_suffix || "I feel:",
  );
  const [raw, setRaw] = useState(recipe.raw ?? fallbackRaw);
  return (
    <details className="advanced-settings">
      <summary>Re-extract with different settings</summary>
      <p className="muted">
        Changed settings create a new recipe version. Saved examples are reused.
      </p>
      <label className="field">
        Extraction method for new version
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value as "paper" | "completion")}
        >
          <option value="paper">Paper-style · raw readout + control PCA</option>
          <option value="completion">
            Completion contrast · assistant response style
          </option>
        </select>
      </label>
      {method === "paper" ? (
        <label className="field">
          Readout suffix for new version
          <input
            value={suffix}
            maxLength={256}
            onChange={(e) => setSuffix(e.target.value)}
          />
        </label>
      ) : (
        <label className="check-row">
          <input
            type="checkbox"
            checked={raw}
            onChange={(e) => setRaw(e.target.checked)}
          />
          Raw completion mode (no chat template)
        </label>
      )}
      <button
        className="button secondary"
        disabled={
          !targetModel ||
          recipe.draft === true ||
          busy.has("extract_recipe") ||
          (method === "paper" && !suffix.trim())
        }
        onClick={() =>
          void act(
            "extract_recipe",
            {
              recipe_id: recipe.id,
              model_id: targetModel!.id,
              raw,
              extraction: {
                method,
                readout_suffix: method === "paper" ? suffix : "",
              },
            },
            "Extraction requested; changed settings create a new version, not an overwrite.",
          )
        }
      >
        Extract with these settings
      </button>
    </details>
  );
}
function Warnings({
  warnings,
  collapsible = true,
}: {
  warnings?: string[];
  collapsible?: boolean;
}) {
  if (!warnings?.length) return null;
  const notes = (
    <ul>
      {warnings.map((warning, index) => (
        <li key={index}>{warning}</li>
      ))}
    </ul>
  );
  return collapsible ? (
    <details className="warnings">
      <summary>Diagnostics ({warnings.length})</summary>
      {notes}
    </details>
  ) : (
    <div className="warnings">{notes}</div>
  );
}

function PreviewGallery({ previews }: { previews?: Json }) {
  const valid = Array.isArray(previews)
    ? previews.filter(
        (
          preview,
        ): preview is {
          prompt: string;
          percent: number;
          output: string;
          [key: string]: Json;
        } =>
          typeof preview === "object" &&
          preview !== null &&
          !Array.isArray(preview) &&
          typeof preview.prompt === "string" &&
          typeof preview.percent === "number" &&
          typeof preview.output === "string",
      )
    : [];
  if (!valid.length)
    return (
      <Empty title="No completed previews yet">
        Completed outputs will appear here with their signed coefficients.
        Nothing is silently replaced.
      </Empty>
    );
  const prompts = [...new Set(valid.map((preview) => preview.prompt))];
  return (
    <div className="preview-gallery">
      {prompts.map((prompt) => (
        <section className="preview-group" key={prompt}>
          <h3>{prompt}</h3>
          <div className="preview-columns">
            {valid
              .filter((preview) => preview.prompt === prompt)
              .sort((a, b) => a.percent - b.percent)
              .map((preview, index) => (
                <article
                  key={index}
                  className={preview.percent === 0 ? "zero-preview" : ""}
                >
                  <span className="eyebrow">
                    {preview.percent > 0 ? "+" : ""}
                    {preview.percent}%{" "}
                    {preview.percent === 0 ? "/ BASELINE" : "/ PERTURBATION"}
                  </span>
                  <p>{preview.output || "No output recorded."}</p>
                </article>
              ))}
          </div>
        </section>
      ))}
      <details className="advanced-settings">
        <summary>Preview records JSON</summary>
        <pre className="json-preview">{jsonText(previews)}</pre>
      </details>
    </div>
  );
}

function StageEditor({
  recipe,
  job,
  act,
  busy,
  targetModel,
  onVersion,
}: {
  recipe?: Recipe;
  job?: Job;
  act: Act;
  busy: Set<string>;
  targetModel?: Model;
  onVersion: (id: string) => void;
}) {
  const refs = checkpointRefs(
    recipe?.stages ?? (job ? jobCheckpoints(job) : undefined),
  );
  const refSignature = JSON.stringify(refs);
  const [outputs, setOutputs] = useState<Partial<Record<StageName, Json>>>({});
  const [artifacts, setArtifacts] = useState<Partial<Record<StageName, Json>>>(
    {},
  );
  const [stage, setStage] = useState<StageName>(
    recipe?.dataset ? "dataset" : "design",
  );
  const [drafts, setDrafts] = useState<Partial<Record<StageName, string>>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshCount, setRefreshCount] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const load = async () => {
      const entries = await Promise.all(
        Object.entries(refs).map(
          async ([name, hash]) =>
            [name as StageName, await act<Json>("artifact", { hash })] as const,
        ),
      );
      if (cancelled) return;
      const direct = directRecipeStages(recipe);
      const available = { ...direct };
      const originals: Partial<Record<StageName, Json>> = {};
      for (const [name, artifact] of entries) {
        if (artifact === null) {
          delete available[name];
          continue;
        }
        originals[name] = artifact;
        const output = completedOutput(artifact);
        if (output !== undefined)
          available[name] = Object.hasOwn(direct, name)
            ? direct[name]!
            : output;
        else delete available[name];
      }
      setOutputs(available);
      setArtifacts(originals);
      setLoading(false);
      setStage((current) =>
        Object.hasOwn(available, current)
          ? current
          : (stageNames.find((name) => Object.hasOwn(available, name)) ??
            "design"),
      );
    };
    void load();
    return () => {
      cancelled = true;
    };
    // Checkpoint identities, not polling object identities, define this read.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refSignature, recipe?.id, job?.id, act, refreshCount]);
  const availableStages = stageNames.filter((name) =>
    Object.hasOwn(outputs, name),
  );
  const stageValue = outputs[stage];
  const draft = drafts[stage] ?? jsonText(stageValue);
  const save = async () => {
    if (!targetModel) return;
    setError("");
    let value: unknown;
    try {
      value = JSON.parse(draft);
    } catch (error) {
      setError(`Invalid JSON: ${errorMessage(error)}`);
      return;
    }
    const action = job ? "edit_job_stage" : "edit_recipe";
    const identity = job ? { job_id: job.id } : { recipe_id: recipe?.id };
    const result = await act<{ id?: string; recipe_id?: string }>(
      action,
      { ...identity, stage, value, model_id: targetModel.id },
      job
        ? "Checkpoint forked. Original job is unchanged; downstream work runs in a separate job."
        : "New recipe version saved. Only downstream products are regenerated.",
    );
    if (result?.id || result?.recipe_id)
      onVersion(result.id ?? result.recipe_id!);
  };
  if (!recipe && !job)
    return (
      <div className="error-box">
        This source is not present in the current snapshot.
      </div>
    );
  return (
    <>
      <div className="editor-toolbar">
        <label className="field">
          Completed stage
          <select
            aria-label="Completed stage"
            value={availableStages.length ? stage : ""}
            disabled={!availableStages.length}
            onChange={(e) => {
              setStage(e.target.value as StageName);
              setError("");
            }}
          >
            {!availableStages.length && (
              <option value="">
                {loading ? "Reading checkpoints…" : "No completed stages yet"}
              </option>
            )}
            {availableStages.map((name) => (
              <option key={name} value={name}>
                {stageLabels[name]}
              </option>
            ))}
          </select>
        </label>
        <button
          className="quiet-button"
          onClick={() => setRefreshCount((count) => count + 1)}
          disabled={loading}
        >
          {loading ? "Reading…" : "Refresh stages"}
        </button>
        {availableStages.length > 0 && (
          <button
            className="quiet-button"
            onClick={() => {
              try {
                setDrafts((current) => ({
                  ...current,
                  [stage]: jsonText(JSON.parse(draft)),
                }));
                setError("");
              } catch (error) {
                setError(errorMessage(error));
              }
            }}
          >
            Format JSON
          </button>
        )}
      </div>
      {!availableStages.length ? (
        <p className="muted small">No completed stages to edit yet.</p>
      ) : (
        <>
          <p className="muted small">
            Edit the completed {stage.replaceAll("_", " ")} output, not its
            metadata wrapper. Matched pairs keep explicit IDs, scenario
            families, messages, and positive/negative completions.
          </p>
          {refs[stage] && (
            <div className="stage-source">
              <span className="eyebrow">SOURCE CHECKPOINT</span>
              <code>{refs[stage]}</code>
            </div>
          )}
          <textarea
            className="code-editor full"
            aria-label={`${stage} JSON editor`}
            spellCheck={false}
            rows={22}
            value={draft}
            onChange={(e) =>
              setDrafts((current) => ({ ...current, [stage]: e.target.value }))
            }
          />
          {error && <div className="error-box">{error}</div>}
          {artifacts[stage] && (
            <details className="advanced-settings">
              <summary>Original checkpoint and generation provenance</summary>
              <pre className="json-preview">{jsonText(artifacts[stage])}</pre>
            </details>
          )}
          <div className="dialog-actions">
            <span className="muted">
              {job
                ? "Forks a new recipe and job. Original work keeps running. Downstream stages may call Codex."
                : "Creates a new version and resumes downstream jobs, including Codex review. Originals stay intact."}{" "}
              {targetModel
                ? `Extraction targets ${targetModel.name}.`
                : "Select a model for downstream extraction."}
            </span>
            <button
              className="button primary"
              disabled={
                !targetModel ||
                loading ||
                busy.has(job ? "edit_job_stage" : "edit_recipe") ||
                draft === jsonText(stageValue)
              }
              onClick={() => void save()}
            >
              {job ? "Save fork & resume" : "Save new version"}
              <Icon name="arrow" size={15} />
            </button>
          </div>
        </>
      )}
    </>
  );
}

function MixtureDiagnostics({
  diagnostics,
  vectors = [],
}: {
  diagnostics: AppliedMixture;
  vectors?: Vector[];
}) {
  const { revision, first_token_index, geometry } = diagnostics;
  const layers = geometry.layers.filter(
    (layer) => layer.axis_count > 0 || layer.injected_norm !== 0,
  );
  const vectorLabel = (id: string) => {
    const name = vectors.find((vector) => vector.id === id)?.name;
    return name
      ? `${name.length > 38 ? name.slice(0, 35) + "…" : name} (${shortId(id)})`
      : shortId(id);
  };
  const metric = (value: number | null | undefined) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toPrecision(4)
      : "different calibrations";
  return (
    <details className="mixture-diagnostics">
      <summary>Applied mixture geometry · revision {revision}</summary>
      <div className="mixture-diagnostics-body">
        <p className="applied-geometry-label">
          Applied revision {revision} · first affected output token{" "}
          {first_token_index}
        </p>
        <Warnings warnings={geometry.warnings} collapsible={false} />
        {layers.length ? (
          <table className="layer-table">
            <caption>Applied residual-space contributions</caption>
            <thead>
              <tr>
                <th>Layer</th>
                <th>Axes</th>
                <th>Injected L2 norm</th>
                <th>% of shared calibration</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((layer) => (
                <tr key={layer.layer}>
                  <td>{layer.layer}</td>
                  <td>{layer.axis_count}</td>
                  <td>{metric(layer.injected_norm)}</td>
                  <td>{metric(layer.percent_of_shared_calibration)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted small">
            No active layer contributions. The applied engine buffer is zero.
          </p>
        )}
        {geometry.cosines.length > 0 && (
          <>
            <h4>Direction overlap at the same layer</h4>
            <p className="muted small">
              Unweighted direction cosine. Signed coefficients determine whether
              contributions add or cancel; the mixture is never renormalized.
            </p>
            <ul className="cosine-list">
              {geometry.cosines.map((pair, index) => (
                <li key={index}>
                  <span>
                    {vectorLabel(pair.left_vector_id)}
                    <br />
                    {vectorLabel(pair.right_vector_id)}
                  </span>
                  <span>
                    L{pair.layer}{" "}
                    <b>
                      {pair.cosine >= 0 ? "+" : ""}
                      {pair.cosine.toFixed(3)}
                    </b>
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </details>
  );
}
