// Browser controller: planning → draft review → revision or final response.
// The backend owns execution/checkpoints; this file renders responses and polls progress.
// Session storage survives refreshes; Open Trip recovers database state after a restart.
let currentThreadId = sessionStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let pendingReview = null;
let requestBusy = false;
let progressTimer = null;
let progressGeneration = 0;
let tripGeneration = 0;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function toStr(val) {
    // Guard against non-string API values (objects, arrays, null, undefined)
    if (val === null || val === undefined) return "";
    if (typeof val === "string") return val;

    // LangChain MCP text responses: [{type:"text", text:"..."}]
    if (Array.isArray(val)) {
        const parts = val.map(item => {
            if (!item) return "";
            if (typeof item === "string") return item;
            // Try to parse inner JSON text (Tavily wraps results in a JSON string)
            if (item.text) {
                try {
                    const parsed = JSON.parse(item.text);
                    return formatTavilyResults(parsed);
                } catch (_) {
                    return item.text;
                }
            }
            return JSON.stringify(item);
        });
        return parts.filter(Boolean).join("\n\n");
    }

    // Plain object — try Tavily shape first
    if (typeof val === "object") {
        if (val.results || val.answer) return formatTavilyResults(val);
        return JSON.stringify(val, null, 2);
    }

    return String(val);
}

function formatTavilyResults(data) {
    // Tavily response: { answer, results: [{title, url, content, score}] }
    const lines = [];

    if (data.answer && data.answer.trim()) {
        lines.push("### Summary\n" + data.answer.trim());
    }

    if (Array.isArray(data.results) && data.results.length) {
        lines.push("\n### Top Results\n");
        data.results.forEach((r, i) => {
            if (!r) return;
            const title   = r.title   || "Result " + (i + 1);
            const url     = r.url     || "";
            const content = r.content ? r.content.slice(0, 400) + (r.content.length > 400 ? "…" : "") : "";
            lines.push(`**${i + 1}. [${title}](${url})**\n${content}\n`);
        });
    }

    return lines.length ? lines.join("\n") : JSON.stringify(data, null, 2);
}

function md(text) {
    const s = toStr(text).trim();
    if (!s) return "<p><em>No data available for this section.</em></p>";
    return typeof marked !== "undefined" ? marked.parse(s) : s;
}

// ─── Prompt ───────────────────────────────────────────────────────────────────

function setPrompt(text) {
    const el = document.getElementById("userInput");
    el.value = text;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 300) + "px";
    el.focus();
}

function newTrip() {
    // Ignore late responses from a trip that was still generating.
    tripGeneration++;
    stopProgress();
    currentThreadId = "user_" + crypto.randomUUID();
    sessionStorage.setItem("travel_thread_id", currentThreadId);
    sessionStorage.removeItem("travel_pending_review");
    localStorage.removeItem("travel_thread_id");
    pendingReview = null;
    latestAnswerMarkdown = "";
    const input = document.getElementById("userInput");
    input.value = "";
    input.style.height = "auto";
    document.getElementById("savedThreadId").value = "";
    document.getElementById("reviewFeedback").value = "";
    document.getElementById("approvalMessage").textContent = "";
    document.getElementById("agentSummary").textContent = "";
    document.getElementById("threadInfo").textContent = "—";
    document.getElementById("resultTitle").textContent = "Your AI Travel Plan";
    document.getElementById("agentProgress").replaceChildren();
    for (const id of ["resultBox", "flightBox", "hotelBox", "itineraryBox"]) {
        document.getElementById(id).replaceChildren();
    }
    document.querySelectorAll(".tab").forEach(tab => {
        tab.classList.remove("hidden");
        tab.classList.toggle("active", tab.dataset.tab === "full");
    });
    document.querySelectorAll(".panel").forEach(panel => {
        panel.classList.toggle("active", panel.dataset.panel === "full");
    });
    hideError();
    setLoading(false);
    document.getElementById("progressMessage").textContent = "";
    for (const id of ["resultSection", "thinkingBox", "reviewBox"]) {
        document.getElementById(id).classList.add("hidden");
    }
    input.focus();
}

// ─── Loading ──────────────────────────────────────────────────────────────────

function setLoading(on, reviewing = false) {
    requestBusy = on;
    document.getElementById("openTripBtn").disabled = on;
    const btn      = document.getElementById("sendBtn");
    const btnText  = document.getElementById("btnText");
    const btnSpin  = document.getElementById("btnLoader");
    const thinking = document.getElementById("thinkingBox");

    btn.disabled = on || Boolean(pendingReview);
    document.getElementById("approveBtn").disabled = on;
    document.getElementById("reviseBtn").disabled = on;
    btnText.classList.toggle("hidden", on);
    btnSpin.classList.toggle("hidden", !on);

    if (on) thinking.classList.remove("hidden");
    if (!on) {
        document.getElementById("progressMessage").textContent = pendingReview ? "Draft ready — review it below." : "Request finished.";
        return;
    }
    document.getElementById("progressMessage").textContent = reviewing
        ? "Processing your review — revisions return to research and draft review…"
        : "The supervisor is selecting research agents and preparing your draft…";
}

const AGENT_LABELS = {
    revision: "Updating trip requirements", supervisor: "Supervisor & guardrail", flight_agent: "Flights",
    hotel_agent: "Hotels", weather_agent: "Weather", budget_agent: "Budget",
    itinerary_agent: "Itinerary", human_approval: "Your review",
    final_agent: "Final response", guardrail_blocked: "Request blocked"
};

function renderProgress(data) {
    const container = document.getElementById("agentProgress");
    container.replaceChildren();
    for (const agent of data.agents || []) {
        const card = document.createElement("div");
        card.className = "workflow-step " + agent.status;
        const title = document.createElement("strong");
        title.textContent = AGENT_LABELS[agent.name] || agent.name;
        const status = document.createElement("span");
        status.textContent = agent.status === "waiting" ? "Awaiting your review" : agent.status;
        card.append(title, status);
        container.append(card);
    }
}

// Each polling cycle captures a generation number. Stale replies cannot repaint a new trip.
function stopProgress() {
    progressGeneration++;
    clearTimeout(progressTimer);
}

function startProgress(threadId, runId = null) {
    stopProgress();
    const generation = progressGeneration;
    const poll = async () => {
        try {
            const suffix = runId ? "?run_id=" + encodeURIComponent(runId) : "";
            const response = await fetch("/api/travel/progress/" + encodeURIComponent(threadId) + suffix);
            if (response.ok) {
                const data = await response.json();
                if (generation === progressGeneration) renderProgress(data);
            }
        } catch (_) { /* Progress polling does not interrupt the main request. */ }
        if (generation === progressGeneration) progressTimer = setTimeout(poll, 1500);
    };
    poll();
}

function renderResultProgress(data) {
    stopProgress();
    document.getElementById("thinkingBox").classList.remove("hidden");
    const blocked = data.guardrail_allowed === false;
    const agents = [{name: "supervisor", status: "completed"}];
    if (blocked) agents.push({name: "guardrail_blocked", status: "blocked"});
    else {
        for (const name of data.selected_agents || []) agents.push({name, status: "completed"});
        agents.push({name: "human_approval", status: data.requires_approval ? "waiting" : "completed"});
        agents.push({name: "final_agent", status: data.requires_approval ? "queued" : "completed"});
    }
    renderProgress({agents});
}

// ─── Error ────────────────────────────────────────────────────────────────────

function showError(msg) {
    document.getElementById("errorMsg").textContent = msg;
    document.getElementById("errorBox").classList.remove("hidden");
}

function hideError() {
    document.getElementById("errorMsg").textContent = "";
    document.getElementById("errorBox").classList.add("hidden");
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

function initTabs() {
    document.querySelectorAll(".tab").forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            btn.classList.add("active");
            document.querySelectorAll(".panel").forEach(p => {
                p.classList.toggle("active", p.dataset.panel === target);
            });
        });
    });
}

// ─── Show result ──────────────────────────────────────────────────────────────

function showResult(data) {
    renderResultProgress(data);
    latestAnswerMarkdown = toStr(data.answer);
    pendingReview = data.requires_approval ? data : null;
    if (pendingReview) {
        sessionStorage.setItem("travel_pending_review", JSON.stringify(data));
    } else {
        sessionStorage.removeItem("travel_pending_review");
    }
    const blocked = data.guardrail_allowed === false;
    document.getElementById("resultTitle").textContent = blocked
        ? "Request needs attention" : pendingReview ? "Draft — awaiting your review" : "Your final travel plan";
    document.getElementById("reviewBox").classList.toggle("hidden", !pendingReview);
    document.getElementById("approvalMessage").textContent = data.approval_request || "Review the draft below before continuing.";
    document.getElementById("reviewFeedback").value = "";
    document.getElementById("agentSummary").textContent = blocked ? "Research was not started."
        : "Selected agents: " + (data.selected_agents || []).map(name => name.replace(/_agent$/, "")).join(", ");
    for (const [tab, value] of [["flights", data.flight_results], ["hotels", data.hotel_results], ["itinerary", data.itinerary]]) {
        document.querySelector(`.tab[data-tab="${tab}"]`).classList.toggle("hidden", !toStr(value).trim());
    }
    document.getElementById("sendBtn").disabled = requestBusy || Boolean(pendingReview);

    document.getElementById("resultBox").innerHTML    = md(data.answer);
    document.getElementById("flightBox").innerHTML    = md(data.flight_results);
    document.getElementById("hotelBox").innerHTML     = md(data.hotel_results);
    document.getElementById("itineraryBox").innerHTML = md(data.itinerary);

    document.getElementById("threadInfo").textContent = "Thread: " + data.thread_id;

    // Reset to full tab
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelector('.tab[data-tab="full"]').classList.add("active");
    document.querySelectorAll(".panel").forEach(p => {
        p.classList.toggle("active", p.dataset.panel === "full");
    });

    const section = document.getElementById("resultSection");
    section.classList.remove("hidden");
    const generation = tripGeneration;
    setTimeout(() => {
        if (generation === tripGeneration) section.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 80);
}

// ─── Send ─────────────────────────────────────────────────────────────────────

async function sendMessage() {
    if (requestBusy || pendingReview) return;
    hideError();

    const input   = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    // A trip reset increments tripGeneration; ignore responses from the old request.
    const generation = tripGeneration;
    const runId = crypto.randomUUID();
    if (!currentThreadId) currentThreadId = "user_" + crypto.randomUUID();
    setLoading(true);
    renderProgress({agents: [{name: "supervisor", status: "running"}]});
    startProgress(currentThreadId, runId);

    try {
        const res  = await fetch("/api/travel", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ message, thread_id: currentThreadId, run_id: runId })
        });

        const data = await res.json();
        if (generation !== tripGeneration) return;

        if (!res.ok || !data.success) {
            throw new Error(data.error || "Something went wrong. Please try again.");
        }

        currentThreadId = data.thread_id;
        sessionStorage.setItem("travel_thread_id", currentThreadId);
        showResult(data);

    } catch (err) {
        if (generation === tripGeneration) showError(err.message);
    } finally {
        if (generation === tripGeneration) {
            stopProgress();
            setLoading(false);
        }
    }
}

// Loading is a read-only GET: do not send a saved ID through Generate Plan.
async function openSavedTrip() {
    if (requestBusy) return;
    hideError();
    const threadId = document.getElementById("savedThreadId").value.trim();
    if (!threadId) {
        showError("Paste the saved trip's thread ID in Open a saved trip.");
        return;
    }
    const generation = tripGeneration;
    stopProgress();
    setLoading(true);
    document.getElementById("progressMessage").textContent = "Opening saved trip…";
    try {
        const response = await fetch("/api/travel/threads/" + encodeURIComponent(threadId));
        const data = await response.json();
        if (generation !== tripGeneration) return;
        if (!response.ok || !data.success) throw new Error(data.error || "Could not open saved trip.");
        currentThreadId = data.thread_id;
        sessionStorage.setItem("travel_thread_id", currentThreadId);
        document.getElementById("userInput").value = data.user_query || "";
        showResult(data);
    } catch (error) {
        if (generation === tripGeneration) showError(error.message);
    } finally {
        if (generation === tripGeneration) setLoading(false);
    }
}

// Resume the saved interrupt: false + feedback returns another draft for review.
async function submitReview(approved) {
    if (requestBusy || !pendingReview) return;
    hideError();
    const feedback = document.getElementById("reviewFeedback").value.trim();
    if (!approved && !feedback) {
        showError("Describe the changes you want before requesting a revision.");
        document.getElementById("reviewFeedback").focus();
        return;
    }
    const generation = tripGeneration;
    const reviewThread = pendingReview.thread_id;
    setLoading(true, true);
    startProgress(reviewThread);
    try {
        const res = await fetch("/api/travel/approve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ thread_id: reviewThread, approved, feedback })
        });
        const data = await res.json();
        if (generation !== tripGeneration) return;
        if (!res.ok || !data.success) throw new Error(data.error || "Review could not be submitted.");
        currentThreadId = data.thread_id;
        sessionStorage.setItem("travel_thread_id", currentThreadId);
        showResult(data);
    } catch (err) {
        if (generation === tripGeneration) showError(err.message);
    } finally {
        if (generation === tripGeneration) {
            stopProgress();
            setLoading(false);
        }
    }
}

// ─── Copy ─────────────────────────────────────────────────────────────────────

function copyResult() {
    const panel = document.querySelector(".panel.active");
    const text  = panel ? panel.innerText : "";

    if (!text || text.includes("No data available")) {
        showError("Nothing to copy in the current tab.");
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        const btn  = document.querySelector(".btn-copy");
        const orig = btn.innerHTML;
        btn.innerHTML = "✅ Copied!";
        setTimeout(() => { btn.innerHTML = orig; }, 1500);
    }).catch(() => showError("Could not copy to clipboard."));
}

// ─── PDF ─────────────────────────────────────────────────────────────────────

function downloadPDF() {
    if (!latestAnswerMarkdown) {
        showError("No travel plan available to download.");
        return;
    }

    const btn  = document.querySelector(".btn-dl");
    const orig = btn.innerHTML;
    btn.innerHTML = "⏳ Preparing…";
    btn.disabled  = true;

    // Reveal all panels for capture
    document.querySelectorAll(".panel").forEach(p => { p.style.display = "block"; });

    html2pdf()
        .set({
            margin:      0.6,
            filename:    "tripwise360-plan.pdf",
            image:       { type: "jpeg", quality: 0.97 },
            html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
            jsPDF:       { unit: "in", format: "a4", orientation: "portrait" },
            pagebreak:   { mode: ["avoid-all", "css", "legacy"] }
        })
        .from(document.getElementById("pdfContent"))
        .save()
        .finally(() => {
            btn.innerHTML = orig;
            btn.disabled  = false;
            document.querySelectorAll(".panel").forEach(p => { p.style.display = ""; });
            const active = document.querySelector(".tab.active");
            if (active) active.click();
        });
}

// ─── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    document.getElementById("savedThreadId").addEventListener("keydown", event => {
        if (event.key === "Enter") { event.preventDefault(); openSavedTrip(); }
    });
    try {
        const saved = JSON.parse(sessionStorage.getItem("travel_pending_review") || "null");
        if (saved && saved.requires_approval && saved.thread_id) {
            currentThreadId = saved.thread_id;
            showResult(saved);
        }
    } catch (_) {
        sessionStorage.removeItem("travel_pending_review");
    }

    const ta = document.getElementById("userInput");
    ta.addEventListener("input", () => {
        ta.style.height = "auto";
        ta.style.height = Math.min(ta.scrollHeight, 300) + "px";
    });
});

document.addEventListener("keydown", e => {
    if (e.ctrlKey && e.key === "Enter" && e.target.id === "userInput") sendMessage();
});
