
const express = require("express");

const router = express.Router();

// POST /api/analysis
router.post("/", (req, res) => {
    try {
        const { symptoms, duration } = req.body;

        // Validate input
        if (!symptoms || !duration) {
            return res.status(400).json({
                success: false,
                message: "Please provide symptoms and duration."
            });
        }

        if (typeof symptoms !== "string" ||
            symptoms.trim().length < 5 ||
            symptoms.trim().length > 1000) {
            return res.status(400).json({
                success: false,
                message: "Symptoms must contain between 5 and 1000 characters."
            });
        }

        if (typeof duration !== "string") {
            return res.status(400).json({
                success: false,
                message: "Invalid duration."
            });
        }

        // Temporary structured demonstration response
        res.status(200).json({
            success: true,
            message: "Analysis request processed successfully!",

            data: {
                patientInput: {
                    symptoms: symptoms.trim(),
                    duration: duration
                },

                pipeline: {
                    nlp: {
                        status: "not_connected",
                        message: "Python NLP service is not connected yet."
                    },

                    triage: {
                        status: "not_connected",
                        message: "Java emergency triage service is not connected yet.",
                        emergency: null
                    },

                    prediction: {
                        status: "not_connected",
                        message: "Python Decision Tree service is not connected yet.",
                        disease: null,
                        confidence: null
                    },

                    route: {
                        status: "not_connected",
                        message: "Java hospital routing service is not connected yet.",
                        hospital: null,
                        distance: null
                    }
                },

                disclaimer: "This is a demonstration response, not a medical assessment. Do not use it to make healthcare decisions."
            }
        });

    } catch (error) {
        console.error("Analysis API error:", error);

        res.status(500).json({
            success: false,
            message: "Internal server error."
        });
    }
});

module.exports = router;
