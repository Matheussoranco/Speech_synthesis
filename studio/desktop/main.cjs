// Minimal desktop window: starts the local server (unless one is already running) and shows its UI.
const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const PORT = process.env.STUDIO_PORT || "7860";
const URL = `http://127.0.0.1:${PORT}`;
const STUDIO = path.resolve(__dirname, "..");
let server = null;

async function isUp() {
  try { return (await fetch(`${URL}/api/status`)).ok; } catch { return false; }
}

function startServer() {
  const venvPython = process.platform === "win32"
    ? path.join(STUDIO, ".venv", "Scripts", "python.exe")
    : path.join(STUDIO, ".venv", "bin", "python");
  const [cmd, args] = fs.existsSync(venvPython)
    ? [venvPython, ["-m", "server", "--no-browser"]]
    : ["uv", ["run", "python", "-m", "server", "--no-browser"]];
  server = spawn(cmd, args, { cwd: STUDIO, env: { ...process.env, STUDIO_PORT: PORT }, windowsHide: true, stdio: "inherit" });
  server.on("exit", (code) => { server = null; if (code) console.error(`server exited with ${code}`); });
}

async function waitForServer(ms = 120000) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    if (await isUp()) return true;
    if (!server) return false;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: 1100, height: 800, title: "Estúdio de Voz", autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, sandbox: true },
  });
  win.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(
    "<body style='font-family:system-ui;display:grid;place-items:center;height:90vh;color:#666'>Iniciando…</body>"));
  if (!(await isUp())) startServer();
  if (await waitForServer()) win.loadURL(URL);
  else win.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(
    "<body style='font-family:system-ui;padding:40px'>O servidor não iniciou. Rode <code>uv sync</code> na pasta studio e tente de novo.</body>"));
  // Downloads and external links open in the system browser.
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  win.webContents.session.setPermissionRequestHandler((_wc, permission, cb) => cb(permission === "media"));
});

app.on("window-all-closed", () => app.quit());
app.on("quit", () => { if (server) server.kill(); });
