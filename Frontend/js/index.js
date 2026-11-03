
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
const nlpStatus = document.getElementById("nlpStatus");
const triageStatus = document.getElementById("triageStatus");
const predictionStatus = document.getElementById("predictionStatus");
const routeStatus = document.getElementById("routeStatus");
const resultDisclaimer = document.getElementById("resultDisclaimer");
const emergencyNotice = document.getElementById("emergencyNotice");
const emergencyMessage = document.getElementById("emergencyMessage");

const routeMapCard = document.getElementById("routeMapCard");
const routeMap = document.getElementById("routeMap");
const routeMapCaption = document.getElementById("routeMapCaption");
const replayRouteMap = document.getElementById("replayRouteMap");

let mapData = null;
let activeRouteMap = null;
let routeMapObserver = null;

const REQUEST_TIMEOUT_MS = 20000;
const SAFETY_ADVICE = "If this could be an emergency, call 112 now. Do not wait for this tool.";

// Results from an earlier request must never be mistaken for this one
function clearPreviousResults() {
    resultsDashboard.hidden = true;
    emergencyNotice.hidden = true;
    emergencyMessage.textContent = "";
    [nlpResult, triageResult, predictionResult, routeResult].forEach(card => card.replaceChildren());
    clearRouteMap();
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

    if (error.isConnectionFailure) {
        return `We could not reach the Sanjeevani AI server. Please check your connection and try again. ${SAFETY_ADVICE}`;
    }

    return `The server could not complete the analysis. Please try again. ${SAFETY_ADVICE}`;
}


symptomForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    // Collect form information
    const symptoms = document.getElementById("symptoms").value.trim();
    const duration = document.getElementById("duration").value;
    const location = document.getElementById("location").value;

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
                duration: duration,
                location: location
            }),
            signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
        }).catch(error => {
            // Marked here because a bug while drawing the results is also a TypeError
            error.isConnectionFailure = true;
            throw error;
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

        // Display emergency notice only if emergency is confirmed. This comes first, so a
        // problem drawing a card below can never hide it.
        // Text is set after revealing it, so screen readers announce the alert.
        if (pipeline.triage.emergency === true) {
            emergencyNotice.hidden = false;
            emergencyMessage.textContent = buildEmergencyMessage(pipeline.route);
        }

        renderTriage(pipeline.triage);
        renderSymptoms(pipeline.nlp);
        renderPrediction(pipeline.prediction);
        renderRoute(pipeline.route);
        resultDisclaimer.textContent = analysisData.disclaimer;

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
// 5. RESULTS RENDERING
// ===================================

const LEVEL_ADVICE = {
    EMERGENCY: "Call 112 or get to a hospital now.",
    URGENT: "This should not wait. Please see a doctor today.",
    ROUTINE: "No emergency signs were found in what you described.",
    UNKNOWN: "We could not recognize any symptoms, so this is not an all-clear. Try describing them differently."
};

const CARE_NAMES = {
    cardiac: "cardiac care",
    neuro: "neurological care",
    respiratory: "respiratory care",
    trauma: "trauma care",
    toxicology: "poison care",
    mental_health: "mental health care",
    general: "general care"
};

// All text goes through textContent, so user input can never become HTML
function createElement(tag, className, text) {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    if (text !== undefined) {
        element.textContent = text;
    }

    return element;
}

// "low_back_pain" -> "Low back pain"
function readableName(name) {
    const words = name.replace(/\(s\)|\.\d+$/g, "").replace(/,/g, "").replace(/_/g, " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
}

function createTagList(names, className) {
    const list = createElement("div", "tag-list");

    // Duplicate dataset columns such as "regurgitation.1" read the same
    [...new Set(names.map(readableName))].forEach(name => {
        list.appendChild(createElement("span", `tag ${className}`, name));
    });

    return list;
}

function renderSymptoms(nlp) {
    nlpStatus.textContent = "Complete";

    if (nlp.symptoms.length > 0) {
        nlpResult.appendChild(createElement("p", "result-label", "Symptoms found"));
        nlpResult.appendChild(createTagList(nlp.symptoms, "tag-found"));
    } else {
        nlpResult.appendChild(createElement("p", "", "No symptoms the model knows were found."));
    }

    if (nlp.negatedSymptoms.length > 0) {
        nlpResult.appendChild(createElement("p", "result-label", "Not counted (denied or in the past)"));
        nlpResult.appendChild(createTagList(nlp.negatedSymptoms, "tag-negated"));
    }

    if (nlp.flags.length > 0) {
        nlpResult.appendChild(createElement("p", "result-label", "Also noted"));
        nlpResult.appendChild(createTagList(nlp.flags, "tag-flag"));
    }
}

// The reasons for the level given. Rule names show the stage: E for emergency, U for urgent.
function triageReasons(triage) {
    const prefix = { EMERGENCY: "E", URGENT: "U" }[triage.level];

    if (!prefix) {
        return [];
    }

    return [...new Set(triage.rulesFired
        .filter(step => step.rule.startsWith(prefix))
        .map(step => step.description))];
}

function renderTriage(triage) {
    triageStatus.textContent = "Complete";

    triageResult.appendChild(createElement("span", `level-badge level-${triage.level.toLowerCase()}`, triage.level));
    triageResult.appendChild(createElement("p", "", LEVEL_ADVICE[triage.level]));

    const reasons = triageReasons(triage);

    if (reasons.length > 0) {
        triageResult.appendChild(createElement("p", "result-label", "Why"));

        const list = createElement("ul", "result-list");
        reasons.forEach(reason => list.appendChild(createElement("li", "", reason)));
        triageResult.appendChild(list);
    }

    if (triage.primaryCare) {
        triageResult.appendChild(createElement("p", "result-label", "Care needed"));
        triageResult.appendChild(createElement("p", "", readableName(CARE_NAMES[triage.primaryCare])));
    }

    triageResult.appendChild(createRuleTrace(triage));
}

// The answer goes into the symptom text and the form is sent again, so a "yes" also
// goes through the emergency rules, and the text always shows what was said
function answerFollowUp(answer) {
    symptomsInput.value = `${symptomsInput.value.trim()}, ${answer}`;
    updateSymptomInterface();
    symptomForm.requestSubmit();
}

function createFollowUp(symptom) {
    const name = readableName(symptom);
    const box = createElement("div", "follow-up");
    const buttons = createElement("div", "follow-up-buttons");
    const yes = createElement("button", "primary-button", "Yes");
    const no = createElement("button", "secondary-button", "No");

    yes.type = "button";
    no.type = "button";
    yes.addEventListener("click", () => answerFollowUp(name));
    no.addEventListener("click", () => answerFollowUp(`no ${name}`));
    buttons.append(yes, no);

    box.appendChild(createElement("p", "follow-up-question", `Do you also have ${name.toLowerCase()}?`));
    box.appendChild(buttons);
    box.appendChild(createElement("p", "result-note", "The Decision Tree chose this as its next most useful question."));

    return box;
}

function renderPrediction(prediction) {
    if (prediction.status === "skipped_emergency") {
        predictionStatus.textContent = "Skipped";
        predictionResult.appendChild(createElement("p", "", "Skipped because this may be an emergency. Getting help comes first."));

        const bypass = createDebugBlock("Decision Tree trace");
        bypass.appendChild(createElement("p", "", "The tree was never run: the triage rules found an emergency first."));
        predictionResult.appendChild(bypass);
        return;
    }

    if (prediction.status === "unavailable") {
        predictionStatus.textContent = "Unavailable";
        predictionResult.appendChild(createElement("p", "", "The prediction model is not available right now. The emergency rules above still ran."));
        return;
    }

    if (prediction.status === "not_run") {
        predictionStatus.textContent = "Not run";
        predictionResult.appendChild(createElement("p", "", "No prediction was made, because the model needs at least one symptom it knows."));
        return;
    }

    predictionStatus.textContent = "Complete";

    if (prediction.lowConfidence) {
        predictionResult.appendChild(createElement("p", "weak-notice", "Not enough information yet. Please add more symptoms, or answer the question below."));
    }

    if (prediction.followUp) {
        predictionResult.appendChild(createFollowUp(prediction.followUp));
    }

    if (prediction.lowConfidence) {
        predictionResult.appendChild(createElement("p", "result-label", "Possible conditions so far"));
    }

    const list = createElement("div", "prediction-list");

    prediction.predictions.forEach(item => {
        const percent = Math.round(item.confidence * 100);
        const row = createElement("div", "prediction-row");
        const heading = createElement("div", "prediction-heading");
        const track = createElement("div", "prediction-track");
        const bar = createElement("div", "prediction-bar");

        heading.appendChild(createElement("span", "", readableName(item.disease)));
        heading.appendChild(createElement("span", "prediction-score", `${percent}%`));
        bar.style.width = `${percent}%`;
        track.appendChild(bar);
        row.append(heading, track);
        list.appendChild(row);
    });

    predictionResult.appendChild(list);
    predictionResult.appendChild(createElement("p", "result-note", "Possible conditions only, not a diagnosis. Percentages are model confidence, not medical certainty."));

    if (prediction.decisionPath) {
        predictionResult.appendChild(createTreeTrace(prediction.decisionPath));
    }
}

function describeRouteReason(route) {
    const careName = CARE_NAMES[route.care];

    if (route.basis.source === "triage") {
        return `Going to ${careName} because of the triage rules.`;
    }

    if (route.basis.source === "prediction") {
        return `Going to ${careName} because the most likely condition is ${readableName(route.basis.disease).toLowerCase()}.`;
    }

    return "Going to general care because the system isn't confident enough to choose a specialty.";
}

function renderRoute(route) {
    if (route.status === "no_location") {
        routeStatus.textContent = "Not run";
        routeResult.appendChild(createElement("p", "", "Select your area in the form above to find the fastest suitable hospital."));
        return;
    }

    if (route.status === "unavailable") {
        routeStatus.textContent = "Unavailable";
        routeResult.appendChild(createElement("p", "", `We could not find a route. ${route.message}`));
        return;
    }

    routeStatus.textContent = "Complete";

    routeResult.appendChild(createElement("p", "route-hospital", route.hospital.name));

    if (route.path.length > 1) {
        routeResult.appendChild(createElement("p", "route-stats", `${route.minutes} min · ${route.km} km`));
        routeResult.appendChild(createElement("p", "", route.path.map(stop => stop.name).join(" → ")));
    } else {
        routeResult.appendChild(createElement("p", "route-stats", "In your area"));
    }

    // Only an explanation, so a server without this field still shows the route
    if (route.basis) {
        routeResult.appendChild(createElement("p", "route-reason", describeRouteReason(route)));
    }

    const comparison = route.comparison;
    const notes = createElement("ul", "result-list");
    const careName = CARE_NAMES[route.care];

    if (!comparison.closest_hospital.can_treat) {
        notes.appendChild(createElement("li", "", `The closest hospital, ${comparison.closest_hospital.name}, does not offer ${careName}.`));
    }

    if (comparison.minutes_saved > 0) {
        notes.appendChild(createElement("li", "", `${comparison.minutes_saved} min faster than ${comparison.closest_suitable_hospital.name}, which is closer by distance.`));
    }

    if (notes.children.length > 0) {
        routeResult.appendChild(notes);
    }

    routeResult.appendChild(createElement("p", "result-note", "Demo map: hospitals are fictional, and travel times are estimates."));
    routeResult.appendChild(createSearchTrace(route));

    showRouteMap(route);
}

// ===================================
// ACADEMIC DEBUG MODE
// Each trace sits in the card it explains, shown only while the toggle is on.
// ===================================

function createDebugBlock(title) {
    const block = createElement("div", "debug-only");
    block.appendChild(createElement("p", "debug-title", title));
    return block;
}

function createFactList(facts) {
    const list = createElement("span", "debug-facts");

    facts.forEach(fact => {
        list.appendChild(createElement("code", "", fact));
    });

    return list;
}

function createFactLine(label, facts) {
    const line = createElement("p", "debug-line", `${label}: `);
    line.appendChild(createFactList(facts));
    return line;
}

function createRuleTrace(triage) {
    const block = createDebugBlock("Forward chaining trace");

    block.appendChild(createFactLine("Starting facts", triage.initialFacts));

    if (triage.rulesFired.length > 0) {
        const list = createElement("ol", "debug-rules");

        triage.rulesFired.forEach(step => {
            const item = createElement("li");

            item.appendChild(createElement("p", "debug-rule-name", `${step.rule}: ${step.description}`));
            item.appendChild(createFactLine("IF", step.because));

            if (step.required_absent.length > 0) {
                item.appendChild(createFactLine("AND NOT", step.required_absent));
            }

            item.appendChild(createFactLine("THEN", step.concluded));
            list.appendChild(item);
        });

        block.appendChild(list);
    }

    block.appendChild(createElement("p", "result-note",
        `${triage.rulesFired.length} of ${triage.rulesTotal} rules fired, in this order. Chaining stops when no rule can add a new fact.`));

    return block;
}

// Most checks answer "no", so runs of them collapse and the "yes" splits stand out
function createTreeTrace(path) {
    const block = createDebugBlock("Decision Tree trace");
    const list = createElement("ol", "debug-tree");
    let absentRun = [];

    function flushAbsentRun() {
        if (absentRun.length === 0) {
            return;
        }

        const item = createElement("li", "debug-tree-absent");
        const details = createElement("details");
        const count = absentRun.length;

        details.appendChild(createElement("summary", "", `${count} ${count === 1 ? "check" : "checks"}: symptom not present`));
        details.appendChild(createElement("p", "debug-absent-names", absentRun.map(step => readableName(step.symptom)).join(", ")));
        item.appendChild(details);
        list.appendChild(item);
        absentRun = [];
    }

    path.steps.forEach(step => {
        if (!step.present) {
            absentRun.push(step);
            return;
        }

        flushAbsentRun();
        list.appendChild(createElement("li", "debug-tree-present",
            `${readableName(step.symptom)}? Yes (${step.samples.toLocaleString("en-IN")} training rows reach this split)`));
    });

    flushAbsentRun();

    block.appendChild(list);
    block.appendChild(createElement("p", "result-note",
        `${path.steps.length} splits to reach a leaf with ${path.leaf_samples.toLocaleString("en-IN")} training rows, ` +
        `in a tree of ${path.tree_nodes.toLocaleString("en-IN")} nodes. Percentages come from the diseases in that leaf. ` +
        "Training rows include copies with symptoms removed, which teach the tree to handle short descriptions."));

    return block;
}

function createSearchTrace(route) {
    const block = createDebugBlock("A* search trace");
    const wrapper = createElement("div", "debug-table-wrapper");
    const table = createElement("table", "debug-table");
    const headerRow = createElement("tr");
    const routeIds = new Set(route.path.map(stop => stop.id));
    const names = new Map((mapData ? mapData.locations : []).map(location => [location.id, location.name]));

    ["Step", "Expanded", "g", "h", "f", "Frontier"].forEach(heading => {
        const cell = createElement("th", "", heading);
        cell.scope = "col";
        headerRow.appendChild(cell);
    });

    const head = createElement("thead");
    const body = createElement("tbody");
    head.appendChild(headerRow);

    route.search.expansion_steps.forEach((step, index) => {
        const row = createElement("tr", routeIds.has(step.node) ? "is-on-route" : "");

        [
            index + 1,
            names.get(step.node) || readableName(step.node),
            step.g_minutes,
            step.h_minutes,
            step.f_minutes,
            step.frontier_size
        ].forEach(value => row.appendChild(createElement("td", "", String(value))));

        body.appendChild(row);
    });

    table.append(head, body);
    wrapper.appendChild(table);
    block.appendChild(wrapper);

    block.appendChild(createElement("p", "result-note",
        "g = minutes travelled so far. h = straight-line estimate of the minutes left. A* always expands the lowest f = g + h. " +
        `Bold rows are on the final route. A* expanded ${route.search.nodes_expanded} locations; Dijkstra would expand ${route.search.dijkstra_nodes_expanded}.`));

    return block;
}

const debugToggle = document.getElementById("debugToggle");

function setDebugMode(enabled) {
    resultsDashboard.classList.toggle("debug-on", enabled);
    debugToggle.checked = enabled;
}

debugToggle.addEventListener("change", () => {
    setDebugMode(debugToggle.checked);

    // Only a convenience: the page works the same if storage is blocked
    try {
        localStorage.setItem("sanjeevaniDebugMode", String(debugToggle.checked));
    } catch (error) {
        console.warn("Could not remember Debug Mode:", error);
    }
});

try {
    setDebugMode(localStorage.getItem("sanjeevaniDebugMode") === "true");
} catch (error) {
    setDebugMode(false);
}


function clearRouteMap() {
    if (activeRouteMap) {
        activeRouteMap.stop();
        activeRouteMap = null;
    }

    if (routeMapObserver) {
        routeMapObserver.disconnect();
        routeMapObserver = null;
    }

    routeMap.replaceChildren();
    routeMapCard.hidden = true;
}

// The route card above still gives the full answer if the map could not be loaded
function showRouteMap(route) {
    if (!mapData) {
        return;
    }

    routeMapCard.hidden = false;
    activeRouteMap = renderRouteMap(routeMap, mapData, route);

    const explored = route.search.nodes_expanded;
    const dijkstra = route.search.dijkstra_nodes_expanded;
    const comparison = dijkstra > explored ? ` Dijkstra's search, with no heuristic, would explore ${dijkstra}.` : "";

    routeMapCaption.textContent =
        `A* explored ${explored} of ${mapData.locations.length} locations to find this route.${comparison} ` +
        "Lines show which areas connect, not exact road shapes. On a computer, hover over a hospital to see its specialties.";

    // Start the search animation when the map scrolls into view, so it isn't missed
    routeMapObserver = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) {
            routeMapObserver.disconnect();
            routeMapObserver = null;
            activeRouteMap.play();
        }
    }, { threshold: 0.3 });

    routeMapObserver.observe(routeMapCard);
}

replayRouteMap.addEventListener("click", () => {
    if (activeRouteMap) {
        activeRouteMap.play();
    }
});

function buildEmergencyMessage(route) {
    const advice = `This may be a medical emergency. ${LEVEL_ADVICE.EMERGENCY}`;

    if (route.status !== "complete") {
        return advice;
    }

    return `${advice} Fastest suitable hospital on the demo map: ${route.hospital.name}.`;
}


// ===================================
// 6. LIVE SYSTEM STATUS AND LOCATIONS
// ===================================

function showServiceStatus(id, ready) {
    const element = document.getElementById(id);

    element.textContent = ready ? "Ready" : "Unavailable";
    element.className = ready ? "status-active" : "status-down";
}

function showSystemStatus(status) {
    const indicator = document.getElementById("systemStatus");

    indicator.textContent = status.online ? "● SYSTEM ONLINE" : "● SYSTEM OFFLINE";
    indicator.classList.toggle("is-offline", !status.online);

    showServiceStatus("statusTriage", status.triage);
    showServiceStatus("statusPrediction", status.prediction);
    showServiceStatus("statusRouting", status.routing);
}

async function loadSystemStatus() {
    try {
        const response = await fetch("/api/analysis/status");
        showSystemStatus(await response.json());
    } catch (error) {
        console.error("Status check failed:", error);
        showSystemStatus({ online: false });
    }
}

async function loadLocations() {
    const select = document.getElementById("location");

    try {
        const response = await fetch("/api/analysis/locations");
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message);
        }

        data.locations
            .sort((a, b) => a.name.localeCompare(b.name))
            .forEach(location => {
                const option = createElement("option", "", location.name);
                option.value = location.id;
                select.appendChild(option);
            });

    } catch (error) {
        console.error("Could not load locations:", error);
        document.getElementById("locationHint").textContent =
            "Areas could not be loaded, so hospital routes are unavailable. You can still submit your symptoms.";
    }
}

async function loadMapData() {
    try {
        const response = await fetch("/api/analysis/map");
        const data = await response.json();

        if (data.success) {
            mapData = data.map;
        }
    } catch (error) {
        console.error("Could not load the map:", error);
    }
}

loadSystemStatus();
loadLocations();
loadMapData();


// ===================================
// 7. SMART SYMPTOM SUGGESTIONS
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


