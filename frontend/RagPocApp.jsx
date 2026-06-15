import { useEffect, useMemo, useRef, useState } from "react";
import {
  Database,
  Loader2,
  LogOut,
  MessageSquare,
  Upload,
  User,
} from "lucide-react";

// CORS reminder: ensure your Python backend enables CORS middleware, e.g. allow_origins=["*"].
const BACKEND_URL = "http://localhost:8000";

const STORAGE_KEYS = {
  users: "rag_poc_users",
  sessionUserId: "rag_poc_active_user",
  resources: "rag_poc_resources",
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

async function sendMessage(userId, message, activeResourceId) {
  const payload = {
    userId,
    message,
    filterResourceId: activeResourceId || null,
  };

  const response = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
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

  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [chatScope, setChatScope] = useState("all");
  const [selectedResourceId, setSelectedResourceId] = useState("");

  const [backendConnected, setBackendConnected] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.users, JSON.stringify(users));
  }, [users]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.resources, JSON.stringify(resources));
  }, [resources]);

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

  const effectiveSelectedResourceId = useMemo(() => {
    if (!activeUserResources.length) {
      return "";
    }

    const stillExists = activeUserResources.some((r) => r.id === selectedResourceId);
    return stillExists ? selectedResourceId : activeUserResources[0].id;
  }, [activeUserResources, selectedResourceId]);

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

      const newResource = {
        id: String(resourceId),
        title,
        textPreview: text.slice(0, 100),
        userId: activeUserId,
        tenantId: activeUserId,
        uploadedAt: new Date().toISOString(),
      };

      setResources((prev) => [newResource, ...prev]);
      setSelectedResourceId(String(resourceId));
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
      const filterResourceId =
        chatScope === "resource" && effectiveSelectedResourceId
          ? effectiveSelectedResourceId
          : null;

      const response = await sendMessage(activeUserId, text, filterResourceId);

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
            <button
              type="button"
              onClick={() => setAuthMode("login")}
              className={`rounded-lg py-2 text-sm font-medium transition ${
                authMode === "login"
                  ? "bg-white text-indigo-700 shadow"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Log In
            </button>
            <button
              type="button"
              onClick={() => setAuthMode("signup")}
              className={`rounded-lg py-2 text-sm font-medium transition ${
                authMode === "signup"
                  ? "bg-white text-indigo-700 shadow"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Sign Up
            </button>
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
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:px-6">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-indigo-600 p-2 text-white">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">RAG PoC Dashboard</h2>
              <p className="text-xs text-slate-500">Tenant-aware resources + retrieval chat</p>
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
        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm md:p-5">
          <div className="mb-4 flex items-center gap-2">
            <Upload className="h-5 w-5 text-indigo-600" />
            <h3 className="text-lg font-semibold text-slate-900">Resource Management</h3>
          </div>

          <form onSubmit={handleUpload} className="space-y-3">
            <input
              type="text"
              value={titleInput}
              onChange={(e) => setTitleInput(e.target.value)}
              className="w-full rounded-xl border border-slate-300 px-3 py-2 outline-none ring-indigo-500 transition focus:ring"
              placeholder="Resource title"
              required
            />
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              rows={6}
              className="w-full rounded-xl border border-slate-300 px-3 py-2 outline-none ring-indigo-500 transition focus:ring"
              placeholder="Paste document text or knowledge snippet here..."
              required
            />

            <button
              type="submit"
              disabled={uploading}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              Upload to Qdrant
            </button>
          </form>

          <div className="mt-4 rounded-xl bg-slate-50 p-3">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Pipeline Status
            </p>
            <ul className="space-y-1 text-sm">
              {UPLOAD_STEPS.map((step, index) => {
                const active = uploadStepIndex === index;
                const done = uploadStepIndex > index;
                return (
                  <li
                    key={step}
                    className={`flex items-center gap-2 rounded-md px-2 py-1 ${
                      active
                        ? "bg-indigo-50 text-indigo-700"
                        : done
                          ? "text-emerald-700"
                          : "text-slate-400"
                    }`}
                  >
                    {active ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <span className="inline-block h-2 w-2 rounded-full bg-current" />
                    )}
                    {step}
                  </li>
                );
              })}
            </ul>
            {uploadError && <p className="mt-2 text-sm text-red-600">{uploadError}</p>}
          </div>

          <div className="mt-5">
            <h4 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Uploaded Resources
            </h4>
            <div className="max-h-72 space-y-2 overflow-auto pr-1">
              {activeUserResources.length === 0 && (
                <p className="rounded-lg border border-dashed border-slate-300 p-3 text-sm text-slate-500">
                  No resources uploaded for this user yet.
                </p>
              )}

              {activeUserResources.map((resource) => (
                <button
                  key={resource.id}
                  type="button"
                  onClick={() => {
                    setSelectedResourceId(resource.id);
                    setChatScope("resource");
                  }}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    effectiveSelectedResourceId === resource.id
                      ? "border-indigo-300 bg-indigo-50"
                      : "border-slate-200 hover:border-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-semibold text-slate-800">{resource.title}</p>
                    <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                      id: {resource.id}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{resource.textPreview}</p>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                    <span className="rounded bg-emerald-50 px-2 py-0.5 text-emerald-700">
                      user_id: {resource.userId}
                    </span>
                    <span className="rounded bg-indigo-50 px-2 py-0.5 text-indigo-700">
                      tenant_id: {resource.tenantId}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="flex h-[75vh] min-h-140 flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-sm md:p-5">
          <div className="mb-3 flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-indigo-600" />
            <h3 className="text-lg font-semibold text-slate-900">Chatbot Window</h3>
          </div>

          <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <select
              value={chatScope}
              onChange={(e) => setChatScope(e.target.value)}
              className="rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-indigo-500 transition focus:ring"
            >
              <option value="all">All My Resources</option>
              <option value="resource" disabled={activeUserResources.length === 0}>
                Filter by Active Resource
              </option>
            </select>

            <select
              value={effectiveSelectedResourceId}
              onChange={(e) => setSelectedResourceId(e.target.value)}
              disabled={chatScope !== "resource" || activeUserResources.length === 0}
              className="rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-indigo-500 transition disabled:cursor-not-allowed disabled:bg-slate-100"
            >
              {activeUserResources.length === 0 && <option value="">No resources</option>}
              {activeUserResources.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.title}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 space-y-3 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3">
            {messages.length === 0 && !chatLoading && (
              <p className="text-sm text-slate-500">
                Start the conversation. Messages are routed with user-level context for
                multitenant retrieval.
              </p>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm shadow-sm ${
                  msg.role === "user"
                    ? "ml-auto bg-indigo-600 text-white"
                    : "mr-auto bg-white text-slate-800"
                }`}
              >
                {msg.content}
              </div>
            ))}

            {chatLoading && (
              <div className="mr-auto inline-flex items-center gap-2 rounded-2xl bg-white px-3 py-2 text-sm text-slate-600 shadow-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                Assistant is thinking...
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {chatError && <p className="mt-2 text-sm text-red-600">{chatError}</p>}

          <form onSubmit={handleSendMessage} className="mt-3 flex gap-2">
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 rounded-xl border border-slate-300 px-3 py-2 outline-none ring-indigo-500 transition focus:ring"
              required
            />
            <button
              type="submit"
              disabled={chatLoading}
              className="rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Send
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}

export default RagPocApp;