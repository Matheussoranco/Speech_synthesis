# Electron desktop contracts

## Runtime and transport

The renderer uses relative /api requests. Desktop production serves it at
voicestudio://voicestudio/; only that authority is accepted. The main process
forwards /api/* to http://127.0.0.1:<OMNIVOICE_PORT>/api/*, preserving real
multipart headers, response status, streaming bodies and Range requests.
The HTML base is / so direct navigation and reloads of nested routes resolve assets.
Development uses the configured loopback Vite renderer origin and proxy.

Sandboxing and context isolation remain enabled, with no Node integration in
the renderer. The CommonJS preload exposes the bridge only to the main frame.
IPC validates the sender webContents, main frame and renderer URL. HTTP/HTTPS
external links are opened through the main process; other external protocols,
credentials, control characters and arbitrary navigation are rejected.
Static files require both lexical containment and resolved filesystem containment.

## Backend lifetime

The desktop probes /api/system/info, accepting the configured app version.
An existing service is attached without taking ownership. A newly spawned child
is owned by the desktop, with restarts and quit serialized. Windows terminates
only that owned PID tree; POSIX terminates its own spawned process group.
No Python stdin watchdog or Rust bridge is supplied by this checkout.

Development resolves backend/ as the actual uv project and passes an absolute
--app-dir. Installed applications include backend source and project metadata.
Python/uv and backend dependencies must already be usable; no bundled Python,
automatic runtime installer or clean-machine setup guarantee is provided.
OMNIVOICE_BACKEND_CMD can select an explicit JSON argv array. Backend version,
port and startup budget remain validated. A skipped backend is attach-only.

## Actual capabilities

On Windows, the independent desktop backend can synthesize standard speech using
the installed System.Speech voices. Discovery lists actual voice names/languages.
It produces measured WAV audio and session history. Speed maps approximately to
the operating system voice rate; arbitrary pitch control is rejected.
The compatibility speech endpoint supports WAV and, when FFmpeg is installed,
actual MP3, Opus/Ogg, AAC/ADTS, FLAC and mono 24 kHz PCM conversion.

The independent backend has no configured cloning, voice-design, transcription,
diarization, dubbing or audiobook model adapter. These requests return explicit
unavailable errors instead of fabricated audio, text or completed jobs. The
separate legacy Python package remains separate, with its own dependencies.

Profiles accept multipart audio with independent upload and stored display names.
Stored audio names must be portable leaf names. Uploads have a configurable
64 MiB default cap and clean up incomplete owned files. Reads/deletes validate
stored identifiers and containment, including unsafe in-memory legacy entries.
GET audio preserves the supported file media type. Profile edits use multipart.

Profile/history metadata is currently held in process memory. Restart loses that
metadata; audio files remain on disk. Settings disclose this and export the
current metadata, including paginated history, rather than claiming persistence.
Storage totals measure files and free disk space. Theme selection is saved locally.
Controls without implementations are not advertised as functional settings.

## Tooling and validation

Use Bun 1.4.2, the committed root bun.lock and the Node version declared by the
Electron package. Type checking uses TypeScript 6 tsc. Tests use Vitest and the
backend pytest suites; lint uses source-scoped oxlint. Explicit electron-vite
configuration builds dist/main/index.cjs and sandboxed dist/preload/index.cjs.
Bun trusts the official Electron dependency's install script.

- bun install --frozen-lockfile
- bun run check:electron
- python -m pytest backend/tests -q
- bun run dist:dir
- bun run smoke-test

Distribution scripts check actual main/preload syntax, renderer assets and Python
resources before packaging. The smoke test requires the unpacked executable and
rendered UI without JavaScript errors; it deliberately skips backend startup.
This is distinct from native real-backend speech validation and does not prove
fresh-machine setup. Linux/macOS packages, physical microphones, model quality,
remote deployment and certificate signing require their own applicable checks.
