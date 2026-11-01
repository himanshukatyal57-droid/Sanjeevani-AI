
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

        if (targetId && targetId.startsWith("#")) {
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


// Dashboard elements
const resultsDashboard = document.getElementById("results");
const nlpResult = document.getElementById("nlpResult");
const triageResult = document.getElementById("triageResult");
const predictionResult = document.getElementById("predictionResult");
const routeResult = document.getElementById("routeResult");
const emergencyNotice = document.getElementById("emergencyNotice");
const emergencyMessage = document.getElementById("emergencyMessage");


symptomForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    // Collect form information
    const symptoms = document.getElementById("symptoms").value.trim();
    const duration = document.getElementById("duration").value;

    // Show loading state
    const submitButton = symptomForm.querySelector(".submit-button");

    submitButton.disabled = true;
    submitButton.textContent = "Sending...";

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
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Something went wrong.");
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

        // Display emergency notice only if emergency is confirmed
        if (pipeline.triage.emergency === true) {
            emergencyNotice.hidden = false;
            emergencyMessage.textContent =
                "An emergency warning has been returned. Seek immediate medical assistance.";
        } else {
            emergencyNotice.hidden = true;
        }

        // Bring dashboard into view
        resultsDashboard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

        console.log("Backend response:", data);


    } catch (error) {
        console.error("Analysis request failed:", error);

        resultMessage.textContent =
            error.message || "Unable to connect to the server.";

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
        "Sensitivity to light",
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
        "Difficulty concentrating"
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

// Update common symptom button appearance
function updateCommonChips() {
    const entries = getSymptomEntries().map(
        item => item.toLowerCase()
    );

    document.querySelectorAll(".symptom-chip").forEach(button => {
        const selected = entries.includes(
            button.dataset.symptom.toLowerCase()
        );

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
        button.textContent = "+ " + suggestion;

        const selected = existingEntries.includes(
            suggestion.toLowerCase()
        );

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


