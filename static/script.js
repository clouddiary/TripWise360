let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";

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

// ─── Loading ──────────────────────────────────────────────────────────────────

function setLoading(on) {
    const btn      = document.getElementById("sendBtn");
    const btnText  = document.getElementById("btnText");
    const btnSpin  = document.getElementById("btnLoader");
    const thinking = document.getElementById("thinkingBox");

    btn.disabled = on;
    btnText.classList.toggle("hidden", on);
    btnSpin.classList.toggle("hidden", !on);

    if (on) {
        thinking.classList.remove("hidden");
        startAgents();
    } else {
        thinking.classList.add("hidden");
        stopAgents();
    }
}

// ─── Agent animation ──────────────────────────────────────────────────────────

const STEP_IDS = ["step-flight", "step-hotel", "step-weather", "step-itinerary"];
let agentTimer = null;

function startAgents() {
    STEP_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove("active", "done");
    });
    runAgent(0);
}

function runAgent(i) {
    if (i >= STEP_IDS.length) return;
    const el = document.getElementById(STEP_IDS[i]);
    if (el) el.classList.add("active");
    agentTimer = setTimeout(() => {
        if (el) { el.classList.remove("active"); el.classList.add("done"); }
        runAgent(i + 1);
    }, 2800);
}

function stopAgents() {
    clearTimeout(agentTimer);
    STEP_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.classList.remove("active"); el.classList.add("done"); }
    });
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
    latestAnswerMarkdown = toStr(data.answer);

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
    setTimeout(() => section.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
}

// ─── Send ─────────────────────────────────────────────────────────────────────

async function sendMessage() {
    hideError();

    const input   = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);

    try {
        const res  = await fetch("/api/travel", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ message, thread_id: currentThreadId })
        });

        const data = await res.json();

        if (!res.ok || !data.success) {
            throw new Error(data.error || "Something went wrong. Please try again.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);
        showResult(data);

    } catch (err) {
        showError(err.message);
    } finally {
        setLoading(false);
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

    const ta = document.getElementById("userInput");
    ta.addEventListener("input", () => {
        ta.style.height = "auto";
        ta.style.height = Math.min(ta.scrollHeight, 300) + "px";
    });
});

document.addEventListener("keydown", e => {
    if (e.ctrlKey && e.key === "Enter") sendMessage();
});
