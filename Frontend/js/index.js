
// ===================================
// 1. SANJEEVANI AI - FRONTEND SCRIPT
// ===================================

console.log("Sanjeevani AI frontend loaded successfully!");

// ===================================
// 2. NAVIGATION SCROLL EFFECT
// ===================================

const navbar = document.querySelector(".navbar");

window.addEventListener("scroll", () => {
    if (window.scrollY > 50) {
        navbar.classList.add("scrolled");
    } else {
        navbar.classList.remove("scrolled");
    }
});

// ===================================
// 3. SMOOTH SCROLL FOR NAVIGATION
// ===================================

const navigationLinks = document.querySelectorAll(
    '.nav-links a, .nav-button, .hero-actions a, .logo'
);

navigationLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
        const targetId = link.getAttribute("href");

        // A bare "#" is not a valid selector
        if (targetId && targetId.length > 1 && targetId.startsWith("#")) {
            const target = document.querySelector(targetId);

            if (target) {
                event.preventDefault();

                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        }
    });
});


// ===================================
// 4. SYMPTOM FORM SUBMISSION
// ===================================

const symptomForm = document.getElementById("symptomForm");
const resultContainer = document.getElementById("resultContainer");
const resultMessage = document.getElementById("resultMessage");
const resultTitle = document.getElementById("resultTitle");


// Dashboard elements
const resultsDashboard = document.getElementById("results");
const nlpResult = document.getElementById("nlpResult");
const triageResult = document.getElementById("triageResult");
const predictionResult = document.getElementById("predictionResult");
const routeResult = document.getElementById("routeResult");
const emergencyNotice = document.getElementById("emergencyNotice");
const emergencyMessage = document.getElementById("emergencyMessage");

const REQUEST_TIMEOUT_MS = 20000;
const SAFETY_ADVICE = "If this could be an emergency, call 112 now. Do not wait for this tool.";

// Results from an earlier request must never be mistaken for this one
function clearPreviousResults() {
    resultsDashboard.hidden = true;
    emergencyNotice.hidden = true;
    emergencyMessage.textContent = "";
    resultContainer.classList.remove("is-error");
    resultTitle.textContent = "Analysis Result";
}

function showFailure(message) {
    resultContainer.classList.add("is-error");
    resultTitle.textContent = "Analysis unavailable";
    resultMessage.textContent = message;
}

function describeFailure(error) {
    if (error.name === "TimeoutError") {
        return `The analysis is taking too long, so it was stopped. Please try again. ${SAFETY_ADVICE}`;
    }

    if (error.name === "TypeError") {
        return `We could not reach the Sanjeevani AI server. Please check your connection and try again. ${SAFETY_ADVICE}`;
    }

    return `The server could not complete the analysis. Please try again. ${SAFETY_ADVICE}`;
}


symptomForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    // Collect form information
    const symptoms = document.getElementById("symptoms").value.trim();
    const duration = document.getElementById("duration").value;

    // Show loading state
    const submitButton = symptomForm.querySelector(".submit-button");

    submitButton.disabled = true;
    submitButton.textContent = "Sending...";

    clearPreviousResults();
    resultContainer.hidden = false;
    resultMessage.textContent = "Submitting your information to the server...";

    try {
        // Send data to Express backend
        const response = await fetch("/api/analysis", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                symptoms: symptoms,
                duration: duration
            }),
            signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
        });

        // Error pages from a proxy or a crash may not be JSON
        const data = await response.json().catch(() => null);

        // The backend writes its 400 messages for users, so they can be shown as they are
        if (response.status === 400 && data && data.message) {
            showFailure(data.message);
            return;
        }

        if (!response.ok || !data || !data.success) {
            throw new Error(`Analysis API responded with status ${response.status}`);
        }

        // Display backend response
        resultMessage.textContent = data.message;

        // Get structured backend data
        const analysisData = data.data;
        const pipeline = analysisData.pipeline;

        // Display the results dashboard
        resultsDashboard.hidden = false;

        // Update NLP result
        nlpResult.textContent = pipeline.nlp.message;

        // Update emergency triage result
        triageResult.textContent = pipeline.triage.message;

        // Update disease prediction result
        predictionResult.textContent = pipeline.prediction.message;

        // Update hospital route result
        routeResult.textContent = pipeline.route.message;

        // Display emergency notice only if emergency is confirmed.
        // Text is set after revealing it, so screen readers announce the alert.
        if (pipeline.triage.emergency === true) {
            emergencyNotice.hidden = false;
            emergencyMessage.textContent =
                "This may be a medical emergency. Call 112 or get to a hospital now.";
        }

        // Bring dashboard into view
        resultsDashboard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

        console.log("Backend response:", data);


    } catch (error) {
        console.error("Analysis request failed:", error);

        showFailure(describeFailure(error));

    } finally {
        // Restore button
        submitButton.disabled = false;
        submitButton.textContent = "Analyze Symptoms →";
    }
});


// ===================================
// 5. SMART SYMPTOM SUGGESTIONS
// ===================================

const symptomChips = document.getElementById("symptomChips");
const symptomsInput = document.getElementById("symptoms");
const relatedSuggestions = document.getElementById("relatedSuggestions");
const relatedChips = document.getElementById("relatedChips");

const symptomSuggestions = {
    headache: [
        "Throbbing pain",
        "Dizziness",
        "Neck pain",
        "Nausea",
        "Pain around forehead"
    ],
    fever: [
        "Chills",
        "Sweating",
        "Feeling unusually hot",
        "Weakness",
        "Body aches"
    ],
    cough: [
        "Dry cough",
        "Wet cough",
        "Sore throat",
        "Chest discomfort",
        "Mucus while coughing"
    ],
    fatigue: [
        "Low energy",
        "Feeling exhausted",
        "Muscle weakness",
        "Sleepiness"
    ],
    nausea: [
        "Feeling like vomiting",
        "Loss of appetite",
        "Stomach discomfort",
        "Dizziness"
    ],
    dizziness: [
        "Feeling lightheaded",
        "Loss of balance",
        "Spinning sensation",
        "Blurred vision"
    ],
    "body pain": [
        "Muscle aches",
        "Joint pain",
        "Back pain",
        "Neck pain"
    ],
    "sore throat": [
        "Pain while swallowing",
        "Throat irritation",
        "Dry throat",
        "Hoarse voice"
    ]
};

// Get individual comma-separated entries
function getSymptomEntries() {
    return symptomsInput.value
        .split(",")
        .map(item => item.trim())
        .filter(Boolean);
}

// Add or remove a symptom
function toggleSymptom(symptom) {
    let entries = getSymptomEntries();

    const existingIndex = entries.findIndex(
        item => item.toLowerCase() === symptom.toLowerCase()
    );

    if (existingIndex !== -1) {
        entries.splice(existingIndex, 1);
    } else {
        entries.push(symptom);
    }

    symptomsInput.value = entries.join(", ");

    // Update all suggestion states
    updateSymptomInterface();
}

function chipLabel(symptom, selected) {
    return `${selected ? "✓" : "+"} ${symptom}`;
}

// Update common symptom button appearance
function updateCommonChips() {
    const entries = getSymptomEntries().map(
        item => item.toLowerCase()
    );

    document.querySelectorAll(".symptom-chip").forEach(button => {
        const selected = entries.includes(
            button.dataset.symptom.toLowerCase()
        );

        button.textContent = chipLabel(button.dataset.symptom, selected);
        button.classList.toggle("selected", selected);
        button.setAttribute("aria-pressed", String(selected));
    });
}

// Display related suggestions
function updateRelatedSuggestions() {
    const text = symptomsInput.value.toLowerCase();

    relatedChips.innerHTML = "";

    // Find all matching keywords
    const matchedKeywords = Object.keys(symptomSuggestions)
        .filter(keyword => text.includes(keyword));

    const suggestions = [
        ...new Set(
            matchedKeywords.flatMap(
                keyword => symptomSuggestions[keyword]
            )
        )
    ];

    if (suggestions.length === 0) {
        relatedSuggestions.hidden = true;
        return;
    }

    const existingEntries = getSymptomEntries().map(
        item => item.toLowerCase()
    );

    suggestions.forEach(suggestion => {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "related-chip";

        const selected = existingEntries.includes(
            suggestion.toLowerCase()
        );

        button.textContent = chipLabel(suggestion, selected);
        button.classList.toggle("selected", selected);
        button.setAttribute("aria-pressed", String(selected));

        button.addEventListener("click", () => {
            toggleSymptom(suggestion);
        });

        relatedChips.appendChild(button);
    });

    relatedSuggestions.hidden = false;
}

// Refresh everything together
function updateSymptomInterface() {
    updateCommonChips();
    updateRelatedSuggestions();
}

// Common symptom buttons
symptomChips.addEventListener("click", event => {
    const button = event.target.closest(".symptom-chip");

    if (!button) return;

    toggleSymptom(button.dataset.symptom);
});

// Listen to manually typed symptoms
symptomsInput.addEventListener("input", updateSymptomInterface);

// Initialize interface
updateSymptomInterface();


