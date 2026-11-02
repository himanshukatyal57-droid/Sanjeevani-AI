
const express = require("express");

const router = express.Router();

const PYTHON_TIMEOUT_MS = 15000;
const DISCLAIMER = "This is an academic demonstration, not a medical assessment. Do not use it to make healthcare decisions.";

// Read on each call, so a value from .env or a test is always picked up
function pythonUrl(path) {
    return (process.env.PYTHON_API_URL || "http://127.0.0.1:8000") + path;
}

async function callPython(path, options = {}) {
    const response = await fetch(pythonUrl(path), {
        ...options,
        signal: AbortSignal.timeout(PYTHON_TIMEOUT_MS)
    });

    if (!response.ok) {
        throw new Error(`Python API responded with status ${response.status}`);
    }

    return response.json();
}

function buildPrediction(result) {
    if (result.triage.emergency) {
        return { status: "skipped_emergency", predictions: [], decisionPath: null };
    }

    if (result.status === "model_unavailable") {
        return { status: "unavailable", predictions: [], decisionPath: null };
    }

    if (result.predictions.length === 0) {
        return { status: "not_run", predictions: [], decisionPath: null };
    }

    return {
        status: "complete",
        predictions: result.predictions,
        lowConfidence: result.low_confidence,
        followUp: result.follow_up,
        decisionPath: result.decision_path
    };
}

function buildRoute(result, location) {
    if (!location) {
        return { status: "no_location" };
    }

    if (!result.route) {
        return {
            status: "unavailable",
            message: result.route_error || "No route could be found."
        };
    }

    return { status: "complete", ...result.route, basis: result.route_basis };
}

// Translates the Python response into the shape the dashboard displays
function buildPipeline(result, location) {
    return {
        nlp: {
            status: "complete",
            symptoms: result.extracted_symptoms,
            negatedSymptoms: result.negated_symptoms,
            flags: result.clinical_flags
        },

        triage: {
            status: "complete",
            level: result.triage.level,
            emergency: result.triage.emergency,
            primaryCare: result.triage.primary_care,
            careNeeded: result.triage.care_needed,
            initialFacts: result.triage.initial_facts,
            rulesFired: result.triage.rules_fired,
            rulesTotal: result.triage.rules_total
        },

        prediction: buildPrediction(result),

        route: buildRoute(result, location)
    };
}

// POST /api/analysis
router.post("/", async (req, res) => {
    try {
        // Express leaves req.body undefined when the request isn't JSON
        const { symptoms, duration, location } = req.body || {};

        // Validate input
        if (!symptoms || !duration) {
            return res.status(400).json({
                success: false,
                message: "Please provide symptoms and duration."
            });
        }

        if (typeof symptoms !== "string" ||
            symptoms.trim().length < 3 ||
            symptoms.trim().length > 1000) {
            return res.status(400).json({
                success: false,
                message: "Symptoms must contain between 3 and 1000 characters."
            });
        }

        if (typeof duration !== "string") {
            return res.status(400).json({
                success: false,
                message: "Invalid duration."
            });
        }

        // Location is optional, so choosing one can never delay an emergency result
        if (location && typeof location !== "string") {
            return res.status(400).json({
                success: false,
                message: "Invalid location."
            });
        }

        const result = await callPython("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                user_text: symptoms.trim(),
                duration: duration,
                location: location || null
            })
        });

        res.status(200).json({
            success: true,
            message: "Analysis complete.",

            data: {
                patientInput: {
                    symptoms: symptoms.trim(),
                    duration: duration,
                    location: location || null
                },

                pipeline: buildPipeline(result, location),

                disclaimer: DISCLAIMER
            }
        });

    } catch (error) {
        console.error("Analysis API error:", error);

        res.status(503).json({
            success: false,
            message: "The AI service is not available right now."
        });
    }
});

// GET /api/analysis/locations
router.get("/locations", async (req, res) => {
    try {
        const locations = await callPython("/locations");

        res.status(200).json({
            success: true,
            locations: locations
        });

    } catch (error) {
        console.error("Locations API error:", error);

        res.status(503).json({
            success: false,
            message: "Locations are not available right now."
        });
    }
});

// GET /api/analysis/map
router.get("/map", async (req, res) => {
    try {
        const map = await callPython("/map");

        res.status(200).json({
            success: true,
            map: map
        });

    } catch (error) {
        console.error("Map API error:", error);

        res.status(503).json({
            success: false,
            message: "The map is not available right now."
        });
    }
});

// GET /api/analysis/status
router.get("/status", async (req, res) => {
    try {
        const health = await callPython("/");

        res.status(200).json({
            success: true,
            online: true,
            triage: health.triage_ready,
            prediction: health.model_loaded,
            routing: health.map_loaded
        });

    } catch (error) {
        // Not an error for the caller: "offline" is a valid answer to this question
        res.status(200).json({
            success: true,
            online: false,
            triage: false,
            prediction: false,
            routing: false
        });
    }
});

module.exports = router;
