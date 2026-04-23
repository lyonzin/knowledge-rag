#!/usr/bin/env node

"use strict";

const { execFileSync, spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PACKAGE_VERSION = require("../package.json").version;
const BASE_DIR = path.join(os.homedir(), ".knowledge-rag");
const VENV_DIR = path.join(BASE_DIR, "venv");
const VERSION_FILE = path.join(BASE_DIR, ".npm-version");
const PYPI_PACKAGE = "knowledge-rag";
const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 11;

// ---------------------------------------------------------------------------
// Helpers — all user-facing output goes to STDERR (MCP uses stdout)
// ---------------------------------------------------------------------------

function log(msg) {
  process.stderr.write(`[knowledge-rag] ${msg}\n`);
}

function fatal(msg) {
  process.stderr.write(`[knowledge-rag] ERROR: ${msg}\n`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Python discovery (cross-platform)
// ---------------------------------------------------------------------------

function tryPython(cmd, args) {
  try {
    const out = execFileSync(cmd, args, {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 10000,
    }).trim();
    return out;
  } catch {
    return null;
  }
}

function parsePythonVersion(versionStr) {
  const match = versionStr && versionStr.match(/(\d+)\.(\d+)\.(\d+)/);
  if (!match) return null;
  return {
    major: parseInt(match[1], 10),
    minor: parseInt(match[2], 10),
    patch: parseInt(match[3], 10),
    raw: match[0],
  };
}

function findPython() {
  const isWindows = process.platform === "win32";

  const candidates = isWindows
    ? [
        { cmd: "python", args: ["--version"] },
        { cmd: "py", args: ["-3", "--version"] },
      ]
    : [
        { cmd: "python3", args: ["--version"] },
        { cmd: "python", args: ["--version"] },
      ];

  for (const { cmd, args } of candidates) {
    const raw = tryPython(cmd, args);
    const ver = parsePythonVersion(raw);
    if (!ver) continue;
    if (ver.major < MIN_PYTHON_MAJOR) continue;
    if (ver.major === MIN_PYTHON_MAJOR && ver.minor < MIN_PYTHON_MINOR) continue;

    const execCmd = cmd === "py" ? "py" : cmd;
    const execArgs = cmd === "py" ? ["-3"] : [];
    return { cmd: execCmd, args: execArgs, version: ver };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Venv management
// ---------------------------------------------------------------------------

function getVenvPython() {
  if (process.platform === "win32") {
    return path.join(VENV_DIR, "Scripts", "python.exe");
  }
  return path.join(VENV_DIR, "bin", "python");
}

function venvExists() {
  return fs.existsSync(getVenvPython());
}

function createVenv(python) {
  log(`Creating virtual environment at ${VENV_DIR}`);
  fs.mkdirSync(BASE_DIR, { recursive: true });

  const args = [...python.args, "-m", "venv", VENV_DIR];
  try {
    execFileSync(python.cmd, args, {
      stdio: ["ignore", "ignore", "pipe"],
      timeout: 120000,
    });
  } catch (err) {
    fatal(
      `Failed to create virtual environment.\n` +
        `  Command: ${python.cmd} ${args.join(" ")}\n` +
        `  ${err.stderr ? err.stderr.toString().trim() : err.message}`
    );
  }
}

function installedVersionMatches() {
  try {
    const stored = fs.readFileSync(VERSION_FILE, "utf-8").trim();
    return stored === PACKAGE_VERSION;
  } catch {
    return false;
  }
}

function writeVersionFile() {
  fs.writeFileSync(VERSION_FILE, PACKAGE_VERSION, "utf-8");
}

function installPackage() {
  const venvPython = getVenvPython();
  log(`Installing ${PYPI_PACKAGE}==${PACKAGE_VERSION} from PyPI`);

  const args = [
    "-m",
    "pip",
    "install",
    "--upgrade",
    "--quiet",
    `${PYPI_PACKAGE}==${PACKAGE_VERSION}`,
  ];

  try {
    execFileSync(venvPython, args, {
      stdio: ["ignore", "ignore", "pipe"],
      timeout: 300000,
    });
  } catch (err) {
    fatal(
      `Failed to install ${PYPI_PACKAGE}.\n` +
        `  ${err.stderr ? err.stderr.toString().trim() : err.message}`
    );
  }

  writeVersionFile();
  log("Installation complete");
}

// ---------------------------------------------------------------------------
// Ensure environment is ready
// ---------------------------------------------------------------------------

function ensureInstalled(python) {
  if (!venvExists()) {
    createVenv(python);
    installPackage();
    return;
  }

  if (!installedVersionMatches()) {
    log("Version changed, updating package");
    installPackage();
    return;
  }
}

// ---------------------------------------------------------------------------
// Run the MCP server
// ---------------------------------------------------------------------------

function runServer() {
  const venvPython = getVenvPython();

  const child = spawn(venvPython, ["-m", "mcp_server.server"], {
    stdio: "inherit",
    windowsHide: true,
  });

  const forwardSignal = (sig) => {
    if (child.pid) {
      try {
        child.kill(sig);
      } catch {
        // child may have already exited
      }
    }
  };

  process.on("SIGINT", () => forwardSignal("SIGINT"));
  process.on("SIGTERM", () => forwardSignal("SIGTERM"));

  child.on("error", (err) => {
    fatal(`Failed to start MCP server: ${err.message}`);
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.exit(1);
    }
    process.exit(code || 0);
  });
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

function main() {
  const args = process.argv.slice(2);

  if (args.includes("--version")) {
    process.stderr.write(`knowledge-rag ${PACKAGE_VERSION}\n`);
    process.exit(0);
  }

  const python = findPython();
  if (!python) {
    fatal(
      `Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required but was not found.\n` +
        `  Searched for: ${process.platform === "win32" ? "python, py -3" : "python3, python"}\n` +
        `  Install Python from https://www.python.org/downloads/`
    );
  }

  log(`Found Python ${python.version.raw}`);
  ensureInstalled(python);

  if (args.includes("--install-only")) {
    log("Install complete (--install-only). Exiting.");
    process.exit(0);
  }

  runServer();
}

main();
