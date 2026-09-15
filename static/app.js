// DataMind frontend — plain JS, no framework needed for something this size.

const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");

const askPanel = document.getElementById("ask-panel");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const askStatus = document.getElementById("ask-status");

const log = document.getElementById("log");

const auditToggle = document.getElementById("audit-toggle");
const auditLogEl = document.getElementById("audit-log");

// ----- Upload flow -----

fileInput.addEventListener("change", () => {
  uploadBtn.disabled = !fileInput.files.length;
});

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadStatus.textContent = "Uploading...";
  uploadStatus.classList.remove("error");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      uploadStatus.textContent = data.detail || "Upload failed.";
      uploadStatus.classList.add("error");
      return;
    }

    uploadStatus.textContent = `Loaded "${data.filename}" — ${data.rows} rows, ${data.columns.length} columns.`;
    askPanel.hidden = false;
    questionInput.focus();
  } catch (err) {
    uploadStatus.textContent = "Could not reach the server. Is it running?";
    uploadStatus.classList.add("error");
  }
});

// ----- Ask flow -----

askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  askBtn.disabled = true;
  askStatus.textContent = "Thinking...";
  askStatus.classList.remove("error");

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    addLogEntry(data);
    questionInput.value = "";
    askStatus.textContent = "";
  } catch (err) {
    askStatus.textContent = "Could not reach the server.";
    askStatus.classList.add("error");
  } finally {
    askBtn.disabled = false;
    questionInput.focus();
  }
}

function addLogEntry(data) {
  const entry = document.createElement("article");
  entry.className = "entry";

  const isError = data.status !== "success";

  let html = `<p class="entry-question">${escapeHTML(data.question)}</p>`;

  if (data.status === "success") {
    html += `<p class="entry-answer">${escapeHTML(formatResult(data.result))}</p>`;
    if (data.chart_url) {
      html += `<img class="entry-chart" src="${data.chart_url}" alt="Generated chart">`;
    }
  } else if (data.status === "ai_unavailable") {
    html += `<p class="entry-answer error">${escapeHTML(data.message)}</p>`;
  } else {
    html += `<p class="entry-answer error">${escapeHTML(data.message || "Could not answer this question.")}</p>`;
  }

  if (data.attempts && data.attempts.length) {
    const toggleId = `reasoning-${Date.now()}`;
    html += `<button class="reasoning-toggle" data-target="${toggleId}">Show reasoning (${data.attempts.length} attempt${data.attempts.length > 1 ? "s" : ""})</button>`;
    html += `<div class="reasoning-body" id="${toggleId}" hidden>`;
    data.attempts.forEach((a) => {
      html += `<div class="attempt">`;
      html += `<div class="attempt-label ${a.status === "error" ? "error" : ""}">Attempt ${a.attempt} — ${a.status}</div>`;
      html += `<pre>${escapeHTML(a.code)}</pre>`;
      if (a.error) html += `<p class="attempt-error-msg">${escapeHTML(a.error)}</p>`;
      html += `</div>`;
    });
    html += `</div>`;
  }

  entry.innerHTML = html;
  log.prepend(entry);

  const toggleBtn = entry.querySelector(".reasoning-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const body = entry.querySelector(`#${toggleBtn.dataset.target}`);
      body.hidden = !body.hidden;
    });
  }
}

function formatResult(result) {
  if (typeof result === "string") return result;
  return JSON.stringify(result, null, 2);
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ----- Audit log -----

auditToggle.addEventListener("click", async () => {
  if (!auditLogEl.hidden) {
    auditLogEl.hidden = true;
    return;
  }

  auditLogEl.hidden = false;
  auditLogEl.innerHTML = "Loading...";

  try {
    const res = await fetch("/audit-log?limit=50");
    const entries = await res.json();

    if (!entries.length) {
      auditLogEl.innerHTML = "<p>No questions logged yet.</p>";
      return;
    }

    auditLogEl.innerHTML = entries.map((e) => `
      <div class="audit-row">
        <span class="timestamp">${new Date(e.timestamp).toLocaleString()}</span>
        <span class="audit-status ${e.status}">${e.status}</span>
        <div>${escapeHTML(e.question)}</div>
      </div>
    `).join("");
  } catch (err) {
    auditLogEl.innerHTML = "<p>Could not load audit log.</p>";
  }
});