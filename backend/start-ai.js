// Starts the Python API with the project's .venv, so it can't pick up a newer system
// Python that lacks the packages or can't load the saved model.
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const venvPython = path.join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const hasVenv = fs.existsSync(venvPython);

if (!hasVenv) {
    console.warn("No .venv found, so trying the system Python. See \"Run it\" in README.md.");
}

const result = spawnSync(hasVenv ? venvPython : "python", ["-m", "uvicorn", "app:app", "--port", "8000"], {
    cwd: root,
    stdio: "inherit"
});

if (result.error) {
    console.error("Could not start Python:", result.error.message);
}

process.exit(result.status ?? 1);
