import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Database,
  Loader2,
  LogOut,
  MessageSquare,
  Plus,
  Trash2,
  Upload,
  User,
} from "lucide-react";

// CORS reminder: ensure your Python backend enables CORS middleware, e.g. allow_origins=["*"].
const BACKEND_URL = "http://localhost:8000";

const STORAGE_KEYS = {
  users: "rag_poc_users",
  sessionUserId: "rag_poc_active_user",
  resources: "rag_poc_resources",
  chatbots: "rag_poc_chatbots",
};

const UPLOAD_STEPS = [
  "Uploading Text...",
  "Chunking...",
  "Generating OpenAI Embeddings...",
  "Storing in Qdrant...",
  "Success!",
];

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function uploadResource(userId, title, text) {
  const payload = {
    userId,
    title,
    text,
  };

  const response = await fetch(`${BACKEND_URL}/upload`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Upload failed");
  }

  return response.json();
}

async function sendMessage(userId, message, filterResourceIds) {
  const response = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      userId,
      message,
      filterResourceIds: filterResourceIds?.length ? filterResourceIds : null,
    }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Chat request failed");
  }
  return response.json();
}

function readJsonStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function RagPocApp() {
  const [authMode, setAuthMode] = useState("login");
  const [usernameInput, setUsernameInput] = useState("");
  const [passwordInput, setPasswordInput] = useState("");

  const [users, setUsers] = useState(() => readJsonStorage(STORAGE_KEYS.users, []));
  const [activeUserId, setActiveUserId] = useState(() =>
    localStorage.getItem(STORAGE_KEYS.sessionUserId) || ""
  );

  const [resources, setResources] = useState(() =>
    readJsonStorage(STORAGE_KEYS.resources, [])
  );
  const [titleInput, setTitleInput] = useState("");
  const [textInput, setTextInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadStepIndex, setUploadStepIndex] = useState(-1);
  const [uploadError, setUploadError] = useState("");

  // ── Chatbots
  const [chatbots, setChatbots] = useState(() => readJsonStorage(STORAGE_KEYS.chatbots, []));
  const [botNameInput, setBotNameInput] = useState("");
  const [botDescInput, setBotDescInput] = useState("");
  const [expandedBotId, setExpandedBotId] = useState(null);

  // ── Layout
  const [leftTab, setLeftTab] = useState("resources");

  // ── Chat
  const [activeChatbotId, setActiveChatbotId] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  const [backendConnected, setBackendConnected] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.users, JSON.stringify(users));
  }, [users]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.resources, JSON.stringify(resources));
  }, [resources]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.chatbots, JSON.stringify(chatbots));
  }, [chatbots]);

  useEffect(() => {
    if (activeUserId) {
      localStorage.setItem(STORAGE_KEYS.sessionUserId, activeUserId);
    } else {
      localStorage.removeItem(STORAGE_KEYS.sessionUserId);
    }
  }, [activeUserId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  useEffect(() => {
    let cancelled = false;

    async function pingBackend() {
      try {
        const res = await fetch(BACKEND_URL, { method: "GET" });
        if (!cancelled) {
          setBackendConnected(res.ok);
        }
      } catch {
        if (!cancelled) {
          setBackendConnected(false);
        }
      }
    }

    pingBackend();
    const intervalId = setInterval(pingBackend, 15000);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  const activeUser = useMemo(
    () => users.find((u) => u.userId === activeUserId) || null,
    [users, activeUserId]
  );

  const activeUserResources = useMemo(
    () => resources.filter((r) => r.userId === activeUserId),
    [resources, activeUserId]
  );

  const activeUserChatbots = useMemo(
    () => chatbots.filter((b) => b.userId === activeUserId),
    [chatbots, activeUserId]
  );

  const activeChatbot = useMemo(
    () => activeUserChatbots.find((b) => b.id === activeChatbotId) || null,
    [activeUserChatbots, activeChatbotId]
  );

  function handleAuthSubmit(event) {
    event.preventDefault();
    const username = usernameInput.trim();
    const password = passwordInput.trim();

    if (!username || !password) {
      return;
    }

    if (authMode === "signup") {
      const exists = users.some((u) => u.username.toLowerCase() === username.toLowerCase());
      if (exists) {
        alert("Username already exists. Please log in.");
        return;
      }

      const userId = username.toLowerCase().replace(/\s+/g, "_");
      const newUser = { userId, username, password };
      const updatedUsers = [...users, newUser];
      setUsers(updatedUsers);
      setActiveUserId(userId);
      setUsernameInput("");
      setPasswordInput("");
      return;
    }

    const matched = users.find(
      (u) => u.username.toLowerCase() === username.toLowerCase() && u.password === password
    );

    if (!matched) {
      alert("Invalid credentials.");
      return;
    }

    setActiveUserId(matched.userId);
    setUsernameInput("");
    setPasswordInput("");
  }

  function handleLogout() {
    setActiveUserId("");
    setMessages([]);
    setChatInput("");
    setChatError("");
    setActiveChatbotId("");
  }

  // ── Chatbot handlers
  function handleCreateBot(event) {
    event.preventDefault();
    const name = botNameInput.trim();
    if (!name) return;
    const newBot = {
      id: `bot_${Date.now().toString(36)}`,
      name,
      description: botDescInput.trim(),
      resourceIds: [],
      userId: activeUserId,
      createdAt: new Date().toISOString(),
    };
    setChatbots((prev) => [newBot, ...prev]);
    setBotNameInput("");
    setBotDescInput("");
    setExpandedBotId(newBot.id);
  }

  function handleDeleteBot(botId) {
    setChatbots((prev) => prev.filter((b) => b.id !== botId));
    if (activeChatbotId === botId) {
      setActiveChatbotId("");
      setMessages([]);
    }
    if (expandedBotId === botId) setExpandedBotId(null);
  }

  function toggleBotResource(botId, resourceId) {
    setChatbots((prev) =>
      prev.map((b) => {
        if (b.id !== botId) return b;
        const already = b.resourceIds.includes(resourceId);
        return {
          ...b,
          resourceIds: already
            ? b.resourceIds.filter((id) => id !== resourceId)
            : [...b.resourceIds, resourceId],
        };
      })
    );
  }

  async function handleUpload(event) {
    event.preventDefault();

    const title = titleInput.trim();
    const text = textInput.trim();

    if (!title || !text || uploading || !activeUserId) {
      return;
    }

    setUploadError("");
    setUploading(true);

    try {
      for (let i = 0; i < UPLOAD_STEPS.length - 1; i += 1) {
        setUploadStepIndex(i);
        await wait(700);
      }

      const apiResponse = await uploadResource(activeUserId, title, text);

      setUploadStepIndex(UPLOAD_STEPS.length - 1);
      await wait(600);

      const resourceId =
        apiResponse?.resourceId ||
        apiResponse?.id ||
        `${activeUserId}_${Date.now().toString(36)}`;

      setResources((prev) => [
        {
          id: String(resourceId),
          title,
          textPreview: text.slice(0, 120),
          userId: activeUserId,
          tenantId: activeUserId,
          uploadedAt: new Date().toISOString(),
        },
        ...prev,
      ]);
      setTitleInput("");
      setTextInput("");
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStepIndex(-1), 500);
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault();

    const text = chatInput.trim();
    if (!text || chatLoading || !activeUserId) {
      return;
    }

    const userMessage = {
      id: `u_${Date.now()}`,
      role: "user",
      content: text,
      createdAt: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setChatInput("");
    setChatError("");
    setChatLoading(true);

    try {
      const filterResourceIds = activeChatbot?.resourceIds?.length
        ? activeChatbot.resourceIds
        : null;

      const response = await sendMessage(activeUserId, text, filterResourceIds);

      const assistantText =
        response?.answer || response?.message || response?.response || "No response text returned.";

      const assistantMessage = {
        id: `a_${Date.now()}`,
        role: "assistant",
        content: assistantText,
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Failed to reach backend");
    } finally {
      setChatLoading(false);
    }
  }

  if (!activeUserId || !activeUser) {
    return (
      <div className="min-h-screen bg-linear-to-br from-slate-100 via-indigo-50 to-slate-200 p-6 md:p-10">
        <div className="mx-auto max-w-md rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-2xl backdrop-blur">
          <div className="mb-6 flex items-center gap-3">
            <div className="rounded-xl bg-indigo-600 p-2 text-white">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900">RAG PoC Control Panel</h1>
              <p className="text-sm text-slate-500">Client-side auth for rapid prototyping</p>
            </div>
          </div>

          <div className="mb-6 grid grid-cols-2 rounded-xl bg-slate-100 p-1">
            {["login", "signup"].map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setAuthMode(mode)}
                className={`rounded-lg py-2 text-sm font-medium transition ${
                  authMode === mode ? "bg-white text-indigo-700 shadow" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {mode === "login" ? "Log In" : "Sign Up"}
              </button>
            ))}
          </div>

          <form onSubmit={handleAuthSubmit} className="space-y-4">
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-slate-700">Username</span>
              <input
                type="text"
                value={usernameInput}
                onChange={(e) => setUsernameInput(e.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-slate-900 outline-none ring-indigo-500 transition focus:ring"
                placeholder="e.g. acme_analyst"
                required
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-slate-700">Password</span>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-slate-900 outline-none ring-indigo-500 transition focus:ring"
                placeholder="Any password (local only)"
                required
              />
            </label>
            <button
              type="submit"
              className="w-full rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white shadow-lg shadow-indigo-300 transition hover:bg-indigo-500"
            >
              {authMode === "login" ? "Log In" : "Create Account"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800">
      {/* Nav */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:px-6">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-indigo-600 p-2 text-white">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">RAG PoC Dashboard</h2>
              <p className="text-xs text-slate-500">Tenant-aware resources + chatbots</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                backendConnected
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-amber-200 bg-amber-50 text-amber-700"
              }`}
            >
              {backendConnected ? "Connected to Backend" : "Backend Unreachable"}
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm">
              <User className="h-4 w-4 text-slate-500" />
              <span className="font-medium">{activeUser.username}</span>
            </div>
            <button
              type="button"
              onClick={handleLogout}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium hover:bg-slate-50"
            >
              <LogOut className="h-4 w-4" />
              Log Out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-4 p-4 md:grid-cols-2 md:gap-6 md:p-6">

        {/* ── Left column: tabbed ──────────────────────────────────── */}
        <section className="flex flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">

          {/* Tab bar */}
          <div className="grid grid-cols-2 rounded-t-2xl border-b border-slate-200 bg-slate-50 p-1">
            <button
              type="button"
              onClick={() => setLeftTab("resources")}
              className={`inline-flex items-center justify-center gap-2 rounded-xl py-2 text-sm font-medium transition ${
                leftTab === "resources" ? "bg-white text-indigo-700 shadow-sm" : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <Upload className="h-4 w-4" />
              Resources
              {activeUserResources.length > 0 && (
                <span className="rounded-full bg-indigo-100 px-1.5 py-0.5 text-xs text-indigo-700">
                  {activeUserResources.length}
                </span>
              )}
            </button>
            <button
              type="button"
              onClick={() => setLeftTab("chatbots")}
              className={`inline-flex items-center justify-center gap-2 rounded-xl py-2 text-sm font-medium transition ${
                leftTab === "chatbots" ? "bg-white text-indigo-700 shadow-sm" : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <Bot className="h-4 w-4" />
              Chatbots
              {activeUserChatbots.length > 0 && (
                <span className="rounded-full bg-indigo-100 px-1.5 py-0.5 text-xs text-indigo-700">
                  {activeUserChatbots.length}
                </span>
              )}
            </button>
          </div>

          <div className="flex-1 overflow-auto p-4 md:p-5">

            {/* Resources tab */}
            {leftTab === "resources" && (
              <div className="space-y-4">
                <form onSubmit={handleUpload} className="space-y-3">
                  <input
                    type="text"
                    value={titleInput}
                    onChange={(e) => setTitleInput(e.target.value)}
                    className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring"
                    placeholder="Resource title"
                    required
                  />
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    rows={5}
                    className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring"
                    placeholder="Paste document text or knowledge snippet here..."
                    required
                  />
                  <button
                    type="submit"
                    disabled={uploading}
                    className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                    Upload to Qdrant
                  </button>
                </form>

                {(uploadStepIndex >= 0 || uploadError) && (
                  <div className="rounded-xl bg-slate-50 p-3">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Pipeline Status</p>
                    <ul className="space-y-1 text-sm">
                      {UPLOAD_STEPS.map((step, index) => {
                        const active = uploadStepIndex === index;
                        const done = uploadStepIndex > index;
                        return (
                          <li
                            key={step}
                            className={`flex items-center gap-2 rounded-md px-2 py-1 ${
                              active ? "bg-indigo-50 text-indigo-700" : done ? "text-emerald-700" : "text-slate-400"
                            }`}
                          >
                            {active ? <Loader2 className="h-4 w-4 animate-spin" /> : <span className="inline-block h-2 w-2 rounded-full bg-current" />}
                            {step}
                          </li>
                        );
                      })}
                    </ul>
                    {uploadError && <p className="mt-2 text-sm text-red-600">{uploadError}</p>}
                  </div>
                )}

                <div>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Uploaded Resources</p>
                  <div className="space-y-2">
                    {activeUserResources.length === 0 && (
                      <p className="rounded-lg border border-dashed border-slate-300 p-3 text-sm text-slate-500">No resources uploaded yet.</p>
                    )}
                    {activeUserResources.map((resource) => (
                      <div key={resource.id} className="rounded-xl border border-slate-200 p-3">
                        <p className="text-sm font-semibold text-slate-800">{resource.title}</p>
                        <p className="mt-1 text-xs text-slate-500">{resource.textPreview}…</p>
                        <div className="mt-2 flex flex-wrap gap-2 text-xs">
                          <span className="rounded bg-emerald-50 px-2 py-0.5 text-emerald-700">user_id: {resource.userId}</span>
                          <span className="rounded bg-indigo-50 px-2 py-0.5 text-indigo-700">id: {resource.id.slice(0, 8)}…</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Chatbots tab */}
            {leftTab === "chatbots" && (
              <div className="space-y-4">
                <form onSubmit={handleCreateBot} className="space-y-2 rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">New Chatbot</p>
                  <input
                    type="text"
                    value={botNameInput}
                    onChange={(e) => setBotNameInput(e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring"
                    placeholder="Chatbot name"
                    required
                  />
                  <input
                    type="text"
                    value={botDescInput}
                    onChange={(e) => setBotDescInput(e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring"
                    placeholder="Short description (optional)"
                  />
                  <button
                    type="submit"
                    className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500"
                  >
                    <Plus className="h-4 w-4" />
                    Create Chatbot
                  </button>
                </form>

                <div className="space-y-2">
                  {activeUserChatbots.length === 0 && (
                    <p className="rounded-lg border border-dashed border-slate-300 p-3 text-sm text-slate-500">No chatbots yet. Create one above.</p>
                  )}
                  {activeUserChatbots.map((bot) => {
                    const isExpanded = expandedBotId === bot.id;
                    const isActive = activeChatbotId === bot.id;
                    return (
                      <div
                        key={bot.id}
                        className={`rounded-xl border transition ${
                          isActive ? "border-indigo-300 bg-indigo-50" : "border-slate-200 bg-white"
                        }`}
                      >
                        {/* Bot header */}
                        <div className="flex items-center gap-2 p-3">
                          <div className={`rounded-lg p-1.5 ${ isActive ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-500" }`}>
                            <Bot className="h-4 w-4" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-semibold text-slate-800">{bot.name}</p>
                            {bot.description && <p className="truncate text-xs text-slate-500">{bot.description}</p>}
                          </div>
                          <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">
                            {bot.resourceIds.length} resource{bot.resourceIds.length !== 1 ? "s" : ""}
                          </span>
                          <button
                            type="button"
                            onClick={() => setExpandedBotId(isExpanded ? null : bot.id)}
                            className="shrink-0 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                          >
                            {isExpanded ? "Close" : "Edit"}
                          </button>
                          <button
                            type="button"
                            title="Delete chatbot"
                            onClick={() => handleDeleteBot(bot.id)}
                            className="shrink-0 rounded-md border border-slate-200 bg-white p-1.5 text-slate-400 hover:border-red-200 hover:bg-red-50 hover:text-red-600"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>

                        {/* Resource assignment panel */}
                        {isExpanded && (
                          <div className="border-t border-slate-200 p-3 pt-2">
                            <p className="mb-2 text-xs font-medium text-slate-500">Assign knowledge resources</p>
                            {activeUserResources.length === 0 ? (
                              <p className="text-xs text-slate-400">Upload resources first in the Resources tab.</p>
                            ) : (
                              <div className="space-y-1.5">
                                {activeUserResources.map((resource) => {
                                  const checked = bot.resourceIds.includes(resource.id);
                                  return (
                                    <label
                                      key={resource.id}
                                      className={`flex cursor-pointer items-center gap-2.5 rounded-lg border px-3 py-2 text-sm transition ${
                                        checked
                                          ? "border-indigo-300 bg-indigo-50 text-indigo-800"
                                          : "border-slate-200 text-slate-700 hover:border-slate-300"
                                      }`}
                                    >
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        onChange={() => toggleBotResource(bot.id, resource.id)}
                                        className="accent-indigo-600"
                                      />
                                      <span className="min-w-0 flex-1 truncate font-medium">{resource.title}</span>
                                    </label>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ── Right column: Chat ───────────────────────────────────── */}
        <section className="flex h-[75vh] min-h-140 flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">

          {/* Chat header with bot selector */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-4">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-indigo-600" />
              <h3 className="text-lg font-semibold text-slate-900">Chat</h3>
            </div>
            <div className="flex min-w-0 flex-1 items-center justify-end gap-2">
              <Bot className="h-4 w-4 shrink-0 text-slate-400" />
              <select
                value={activeChatbotId}
                onChange={(e) => {
                  setActiveChatbotId(e.target.value);
                  setMessages([]);
                  setChatError("");
                }}
                className="min-w-0 max-w-xs flex-1 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring"
              >
                <option value="">— Select a chatbot —</option>
                {activeUserChatbots.map((bot) => (
                  <option key={bot.id} value={bot.id}>
                    {bot.name} ({bot.resourceIds.length} resource{bot.resourceIds.length !== 1 ? "s" : ""})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Active chatbot context pill */}
          {activeChatbot && (
            <div className="border-b border-slate-100 bg-indigo-50 px-4 py-2">
              <p className="text-xs text-indigo-700">
                <span className="font-semibold">{activeChatbot.name}</span>
                {activeChatbot.description ? ` — ${activeChatbot.description}` : ""}
                {" · "}
                {activeChatbot.resourceIds.length === 0
                  ? "No resources assigned — will search all your data"
                  : `Filtering ${activeChatbot.resourceIds.length} assigned resource${activeChatbot.resourceIds.length !== 1 ? "s" : ""}`}
              </p>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 space-y-3 overflow-auto bg-slate-50 p-4">
            {messages.length === 0 && !chatLoading && (
              <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-sm text-slate-500">
                {activeChatbot
                  ? `Ask anything — ${activeChatbot.name} will retrieve context from its assigned resources.`
                  : "Select a chatbot above to start, or create one in the Chatbots tab."}
              </div>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm shadow-sm ${
                    msg.role === "user"
                      ? "bg-indigo-600 text-white"
                      : "border border-slate-200 bg-white text-slate-800"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {chatLoading && (
              <div className="flex justify-start">
                <div className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 shadow-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {activeChatbot ? `${activeChatbot.name} is thinking…` : "Thinking…"}
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {chatError && (
            <p className="mx-4 mt-1 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600">
              {chatError}
            </p>
          )}

          <form onSubmit={handleSendMessage} className="p-4 pt-3">
            <div className="flex gap-2">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder={activeChatbot ? `Message ${activeChatbot.name}…` : "Select a chatbot first…"}
                disabled={!activeChatbotId}
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring disabled:cursor-not-allowed disabled:bg-slate-100"
              />
              <button
                type="submit"
                disabled={chatLoading || !activeChatbotId}
                className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Send
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}

export default RagPocApp;