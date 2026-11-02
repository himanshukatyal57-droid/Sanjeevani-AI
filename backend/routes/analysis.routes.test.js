// Run with: npm test
const { test, before, after, beforeEach } = require("node:test");
const assert = require("node:assert");
const express = require("express");
const analysisRoutes = require("./analysis.routes");

const ROUTE = { hospital: { name: "Test Hospital" }, minutes: 12.5, km: 8 };

function pythonResponse(overrides = {}) {
    return {
        status: "success",
        extracted_symptoms: ["cough"],
        negated_symptoms: [],
        clinical_flags: [],
        predictions: [{ disease: "flu", confidence: 0.5 }],
        route: null,
        route_error: null,
        triage: { level: "ROUTINE", emergency: false, primary_care: null, care_needed: [], rules_fired: [] },
        ...overrides
    };
}

let appServer;
let pythonServer;
let appUrl;
let pythonUrl;
let pythonHandler;
let lastPythonBody;

function listen(app) {
    return new Promise(resolve => {
        const server = app.listen(0, "127.0.0.1", () => resolve(server));
    });
}

// Opens a port and closes it again, so connecting to it is really refused
async function closedPortUrl() {
    const server = await listen(express());
    const { port } = server.address();
    await new Promise(resolve => server.close(resolve));
    return `http://127.0.0.1:${port}`;
}

function post(body) {
    return fetch(`${appUrl}/api/analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });
}

const VALID = { symptoms: "I have a cough", duration: "1–3 days" };

before(async () => {
    // A stand-in for the Python API, so these tests need no trained model
    const python = express();
    python.use(express.json());
    python.use((req, res) => {
        lastPythonBody = req.body;
        pythonHandler(req, res);
    });
    pythonServer = await listen(python);
    pythonUrl = `http://127.0.0.1:${pythonServer.address().port}`;

    const app = express();
    app.use(express.json());
    app.use("/api/analysis", analysisRoutes);
    appServer = await listen(app);
    appUrl = `http://127.0.0.1:${appServer.address().port}`;
});

after(() => {
    appServer.close();
    pythonServer.close();
});

beforeEach(() => {
    process.env.PYTHON_API_URL = pythonUrl;
    lastPythonBody = null;
    pythonHandler = (req, res) => res.json(pythonResponse());
});

test("sends the text, duration and location to Python", async () => {
    await post({ ...VALID, location: "saket" });
    assert.deepStrictEqual(lastPythonBody, { user_text: "I have a cough", duration: "1–3 days", location: "saket" });
});

test("an empty location is sent as no location", async () => {
    await post({ ...VALID, location: "" });
    assert.strictEqual(lastPythonBody.location, null);
});

test("translates a routine result for the dashboard", async () => {
    const body = await (await post(VALID)).json();
    const { pipeline } = body.data;
    assert.strictEqual(body.success, true);
    assert.deepStrictEqual(pipeline.nlp.symptoms, ["cough"]);
    assert.strictEqual(pipeline.triage.level, "ROUTINE");
    assert.strictEqual(pipeline.triage.emergency, false);
    assert.strictEqual(pipeline.prediction.status, "complete");
    assert.strictEqual(pipeline.prediction.predictions[0].disease, "flu");
    assert.strictEqual(pipeline.route.status, "no_location");
    assert.ok(body.data.disclaimer);
});

test("an emergency is marked as skipping the prediction", async () => {
    pythonHandler = (req, res) => res.json(pythonResponse({
        status: "emergency",
        predictions: [],
        route: ROUTE,
        triage: { level: "EMERGENCY", emergency: true, primary_care: "cardiac", care_needed: ["cardiac"], rules_fired: [{ rule: "E1" }] }
    }));
    const { pipeline } = (await (await post({ ...VALID, location: "saket" })).json()).data;
    assert.strictEqual(pipeline.triage.emergency, true);
    assert.strictEqual(pipeline.triage.primaryCare, "cardiac");
    assert.strictEqual(pipeline.triage.rulesFired[0].rule, "E1");
    assert.strictEqual(pipeline.prediction.status, "skipped_emergency");
    assert.strictEqual(pipeline.route.status, "complete");
    assert.strictEqual(pipeline.route.hospital.name, "Test Hospital");
});

test("no recognized symptoms means the prediction did not run", async () => {
    pythonHandler = (req, res) => res.json(pythonResponse({
        status: "no_symptoms_recognized", extracted_symptoms: [], predictions: []
    }));
    const { pipeline } = (await (await post(VALID)).json()).data;
    assert.strictEqual(pipeline.prediction.status, "not_run");
});

test("a routing problem keeps the rest of the result", async () => {
    pythonHandler = (req, res) => res.json(pythonResponse({
        status: "emergency",
        predictions: [],
        route_error: "Unknown location: atlantis",
        triage: { level: "EMERGENCY", emergency: true, primary_care: "trauma", care_needed: ["trauma"], rules_fired: [] }
    }));
    const response = await post({ ...VALID, location: "atlantis" });
    const { pipeline } = (await response.json()).data;
    assert.strictEqual(response.status, 200);
    assert.strictEqual(pipeline.triage.emergency, true);
    assert.strictEqual(pipeline.route.status, "unavailable");
    assert.strictEqual(pipeline.route.message, "Unknown location: atlantis");
});

test("invalid input is rejected without calling Python", async () => {
    const cases = [
        {},
        { symptoms: "I have a cough" },
        { duration: "1–3 days" },
        { symptoms: "abc", duration: "1–3 days" },
        { symptoms: "x".repeat(1001), duration: "1–3 days" },
        { symptoms: 12345, duration: "1–3 days" },
        { symptoms: "I have a cough", duration: 3 },
        { symptoms: "I have a cough", duration: "1–3 days", location: 7 }
    ];
    for (const body of cases) {
        const response = await post(body);
        assert.strictEqual(response.status, 400, JSON.stringify(body));
        assert.strictEqual((await response.json()).success, false);
    }
    assert.strictEqual(lastPythonBody, null);
});

test("Python being down gives a clear 503", async () => {
    process.env.PYTHON_API_URL = await closedPortUrl();
    const response = await post(VALID);
    const body = await response.json();
    assert.strictEqual(response.status, 503);
    assert.strictEqual(body.success, false);
    assert.strictEqual(body.message, "The AI service is not available right now.");
});

test("a Python error gives a clear 503 without its details", async () => {
    pythonHandler = (req, res) => res.status(500).json({ detail: "secret stack trace" });
    const response = await post(VALID);
    const body = await response.json();
    assert.strictEqual(response.status, 503);
    assert.ok(!JSON.stringify(body).includes("secret"));
});

test("locations are passed through", async () => {
    pythonHandler = (req, res) => res.json([{ id: "saket", name: "Saket" }]);
    const body = await (await fetch(`${appUrl}/api/analysis/locations`)).json();
    assert.deepStrictEqual(body.locations, [{ id: "saket", name: "Saket" }]);
});

test("locations report a failure when Python is down", async () => {
    process.env.PYTHON_API_URL = await closedPortUrl();
    const response = await fetch(`${appUrl}/api/analysis/locations`);
    assert.strictEqual(response.status, 503);
});

test("status reports each part separately", async () => {
    pythonHandler = (req, res) => res.json({ status: "running", triage_ready: true, model_loaded: false, map_loaded: true });
    const body = await (await fetch(`${appUrl}/api/analysis/status`)).json();
    assert.deepStrictEqual(
        { online: body.online, triage: body.triage, prediction: body.prediction, routing: body.routing },
        { online: true, triage: true, prediction: false, routing: true }
    );
});

test("status reports offline when Python is down", async () => {
    process.env.PYTHON_API_URL = await closedPortUrl();
    const body = await (await fetch(`${appUrl}/api/analysis/status`)).json();
    assert.strictEqual(body.online, false);
    assert.strictEqual(body.routing, false);
});
