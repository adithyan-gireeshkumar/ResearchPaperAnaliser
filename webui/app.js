const $ = (id) => document.getElementById(id);
let paperId = null;
let chatHistory = [];

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4500);
}

// Fallback markdown formatter in case marked.js CDN is unavailable
function fallbackMarkdown(text) {
  if (!text) return "";
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  // Headings
  html = html.replace(/^### (.*$)/gim, "<h4>$1</h4>");
  html = html.replace(/^## (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^# (.*$)/gim, "<h2>$1</h2>");
  // Bold & italic
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
  // Bullet lists
  html = html.replace(/^\s*[-*]\s+(.*$)/gim, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");
  // Paragraphs
  html = html.replace(/\n\n+/g, "<br><br>");
  return html;
}

function renderMarkdown(text) {
  if (!text) return "";
  if (typeof window.marked !== "undefined" && typeof window.marked.parse === "function") {
    try {
      return window.marked.parse(text, { breaks: true, gfm: true });
    } catch (e) {
      console.warn("marked.parse error, using fallback", e);
    }
  }
  return fallbackMarkdown(text);
}

function renderText(target, text) {
  const element = $(target);
  if (!element) return;
  element.innerHTML = renderMarkdown(text || "No details were returned.");
}

function setFile(file) {
  if (!file) return;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Please choose a PDF file.");
    return;
  }
  $("file-input").files = (() => {
    const data = new DataTransfer();
    data.items.add(file);
    return data.files;
  })();
  $("file-name").textContent = file.name;
  $("selected-file").hidden = false;
  $("dropzone").style.display = "none";
  $("upload-button").disabled = false;
}

function clearFile() {
  $("file-input").value = "";
  $("selected-file").hidden = true;
  $("dropzone").style.display = "flex";
  $("upload-button").disabled = true;
}

// Load saved API key from localStorage
const storedKey = localStorage.getItem("groq_api_key");
if (storedKey && $("api-key")) {
  $("api-key").value = storedKey;
}

$("api-key").addEventListener("input", (e) => {
  const val = e.target.value.trim();
  if (val) {
    localStorage.setItem("groq_api_key", val);
  } else {
    localStorage.removeItem("groq_api_key");
  }
});

$("toggle-key").addEventListener("click", () => {
  const input = $("api-key");
  const isPassword = input.type === "password";
  input.type = isPassword ? "text" : "password";
  $("toggle-key").textContent = isPassword ? "🔒" : "👁";
  $("toggle-key").title = isPassword ? "Hide API key" : "Show API key";
});

async function uploadPaper() {
  const file = $("file-input").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const keyVal = $("api-key").value.trim();
  if (keyVal) {
    form.append("api_key", keyVal);
    localStorage.setItem("groq_api_key", keyVal);
  }

  $("empty-state").hidden = true;
  $("brief").hidden = true;
  $("loading-state").hidden = false;
  $("upload-button").disabled = true;

  try {
    const response = await fetch("/upload", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Upload failed.");
    paperId = data.paper_id;
    chatHistory = [];

    $("paper-title").textContent = data.title;
    $("paper-meta").textContent = `${data.page_count} pages · Ready to explore`;
    renderText("summary", data.summary);
    renderText("contributions", data.contributions);
    renderText("technical-terms", data.technical_terms);

    $("loading-state").hidden = true;
    $("brief").hidden = false;
  } catch (error) {
    $("loading-state").hidden = true;
    $("empty-state").hidden = false;
    showToast(error.message);
  } finally {
    $("upload-button").disabled = false;
  }
}

async function askQuestion(event, customQuery = null) {
  if (event) event.preventDefault();
  const input = $("question");
  const askButton = $("ask-button");
  const query = (customQuery || input.value).trim();
  if (!query || !paperId) return;

  // Prevent double submissions
  input.disabled = true;
  askButton.disabled = true;

  const messages = $("messages");
  const userMessage = document.createElement("div");
  userMessage.className = "message user-message";
  userMessage.textContent = query;
  messages.appendChild(userMessage);

  if (!customQuery) input.value = "";

  const pending = document.createElement("div");
  pending.className = "message assistant-message";
  pending.innerHTML = `
    <div class="message-loading">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      <span class="loading-label">Analyzing paper...</span>
    </div>
  `;
  messages.appendChild(pending);
  messages.scrollTop = messages.scrollHeight;

  try {
    const payload = {
      query,
      paper_id: paperId,
      api_key: $("api-key").value.trim() || null,
      history: chatHistory.slice(-6),
    };

    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not answer that question.");

    const answer = data.answer || "I couldn't find an answer in the paper.";
    chatHistory.push({ role: "user", content: query });
    chatHistory.push({ role: "assistant", content: answer });

    let bodyHtml = `<div class="message-body">${renderMarkdown(answer)}</div>`;

    // Build citation pills
    const pills = [];
    if (data.has_overview) {
      pills.push(`<span class="citation-pill overview-pill">Overview</span>`);
    }
    if (Array.isArray(data.pages) && data.pages.length > 0) {
      data.pages.forEach((p) => {
        pills.push(`<span class="citation-pill">Page ${p}</span>`);
      });
    }

    let metaHtml = "";
    if (pills.length > 0) {
      metaHtml = `<div class="message-meta"><span class="meta-label">Sources:</span>${pills.join("")}</div>`;
    }

    const copyBtnHtml = `<button type="button" class="copy-btn" title="Copy answer" aria-label="Copy answer">Copy</button>`;

    pending.innerHTML = `
      ${bodyHtml}
      <div class="message-footer">
        ${metaHtml}
        ${copyBtnHtml}
      </div>
    `;

    // Attach copy handler
    const copyBtn = pending.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(answer);
          copyBtn.textContent = "Copied!";
          setTimeout(() => (copyBtn.textContent = "Copy"), 2000);
        } catch (e) {
          showToast("Failed to copy to clipboard.");
        }
      });
    }
  } catch (error) {
    pending.className = "message assistant-message message-error";
    pending.innerHTML = `
      <div class="message-body error-body">
        <strong>Error:</strong> ${error.message}
      </div>
    `;
    showToast(error.message);
  } finally {
    input.disabled = false;
    askButton.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
  }
}

$("file-input").addEventListener("change", (event) => setFile(event.target.files[0]));
$("clear-file").addEventListener("click", clearFile);
$("upload-button").addEventListener("click", uploadPaper);
$("question-form").addEventListener("submit", askQuestion);

// Suggested quick prompt chips
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const q = btn.getAttribute("data-q");
    if (q && paperId) {
      askQuestion(null, q);
    }
  });
});

$("new-paper").addEventListener("click", () => {
  paperId = null;
  chatHistory = [];
  clearFile();
  $("brief").hidden = true;
  $("empty-state").hidden = false;
  $("messages").innerHTML = `
    <div class="message assistant-message">
      <div class="message-body">Ask me about the methods, findings, authors, or anything else in the paper.</div>
    </div>
  `;
});

["dragenter", "dragover"].forEach((eventName) =>
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("dropzone").classList.add("dragging");
  })
);

["dragleave", "drop"].forEach((eventName) =>
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("dropzone").classList.remove("dragging");
  })
);

$("dropzone").addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
