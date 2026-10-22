const $ = (id) => document.getElementById(id);
let paperId = null;

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4500);
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

function renderText(target, text) {
  const element = $(target);
  element.textContent = text || "No details were returned.";
}

async function uploadPaper() {
  const file = $("file-input").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  if ($("api-key").value.trim()) form.append("api_key", $("api-key").value.trim());

  $("empty-state").hidden = true;
  $("brief").hidden = true;
  $("loading-state").hidden = false;
  $("upload-button").disabled = true;
  try {
    const response = await fetch("/upload", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Upload failed.");
    paperId = data.paper_id;
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

async function askQuestion(event) {
  event.preventDefault();
  const input = $("question");
  const query = input.value.trim();
  if (!query || !paperId) return;
  const messages = $("messages");
  const userMessage = document.createElement("div");
  userMessage.className = "message user-message";
  userMessage.textContent = query;
  messages.appendChild(userMessage);
  input.value = "";
  const pending = document.createElement("div");
  pending.className = "message assistant-message";
  pending.textContent = "Thinking...";
  messages.appendChild(pending);
  messages.scrollTop = messages.scrollHeight;
  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, paper_id: paperId, api_key: $("api-key").value.trim() || null }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not answer that question.");
    pending.textContent = data.answer || "I couldn't find an answer in the paper.";
  } catch (error) {
    pending.remove();
    showToast(error.message);
  }
  messages.scrollTop = messages.scrollHeight;
}

$("file-input").addEventListener("change", (event) => setFile(event.target.files[0]));
$("clear-file").addEventListener("click", clearFile);
$("upload-button").addEventListener("click", uploadPaper);
$("question-form").addEventListener("submit", askQuestion);
$("new-paper").addEventListener("click", () => {
  paperId = null;
  clearFile();
  $("brief").hidden = true;
  $("empty-state").hidden = false;
  $("messages").innerHTML = '<div class="message assistant-message">Ask me about the methods, findings, authors, or anything else in the paper.</div>';
});
["dragenter", "dragover"].forEach((eventName) => $("dropzone").addEventListener(eventName, (event) => {
  event.preventDefault();
  $("dropzone").classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => $("dropzone").addEventListener(eventName, (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("dragging");
}));
$("dropzone").addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
