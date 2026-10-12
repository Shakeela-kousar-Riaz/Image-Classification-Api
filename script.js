
const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const statusText = document.getElementById("status");
const dropzone = document.getElementById("dropzone");
const predictButton = document.getElementById("predict");
const results = document.getElementById("results");

let selectedFile = null;
let previewUrl = null;

function selectFile(file) {
  if (!file) return;

  if (!["image/jpeg", "image/png"].includes(file.type)) {
    statusText.textContent = "Please select a JPEG or PNG image.";
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    statusText.textContent = "Image size must be below 5 MB.";
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);

  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  results.hidden = true;
  statusText.textContent = "";
}

fileInput.addEventListener("change", () => {
  selectFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", event => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", event => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  selectFile(event.dataTransfer.files[0]);
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) {
    statusText.textContent = "Please upload an image first.";
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "Analyzing...";
  statusText.textContent = "⏳ Please wait while AI analyzes your image.";
  results.hidden = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    document.getElementById("label").textContent =
      data.predicted_class;

    document.getElementById("confidence").textContent =
      `Confidence: ${data.confidence}%`;

    document.getElementById("inferenceTime").textContent =
      `Inference time: ${data.inference_time_seconds} seconds`;

    const top5 = document.getElementById("top5");
    top5.replaceChildren();

    data.top5.forEach(item => {
      const row = document.createElement("div");
      row.className = "result-row";

      const label = document.createElement("p");
      label.textContent =
        `${item.class_name}: ${item.confidence}%`;

      const bar = document.createElement("div");
      bar.className = "bar";

      const fill = document.createElement("span");
      fill.style.width = `${item.confidence}%`;

      bar.appendChild(fill);
      row.append(label, bar);
      top5.appendChild(row);
    });

    results.hidden = false;
    statusText.textContent = "Prediction completed successfully.";

    try {
      const history = JSON.parse(
        localStorage.getItem("predictions") || "[]"
      );

      history.unshift({
        name: data.predicted_class,
        confidence: data.confidence,
        time: new Date().toLocaleString()
      });

      localStorage.setItem(
        "predictions",
        JSON.stringify(history.slice(0, 10))
      );

      renderHistory();
    } catch {
      statusText.textContent =
        "Prediction completed. Browser history storage is unavailable.";
    }
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Analyze Image";
  }
});

function renderHistory() {
  const list = document.getElementById("history");
  list.replaceChildren();

  try {
    const history = JSON.parse(
      localStorage.getItem("predictions") || "[]"
    );

    history.forEach(item => {
      const li = document.createElement("li");
      li.textContent =
        `${item.name} — ${item.confidence}% — ${item.time}`;
      list.appendChild(li);
    });
  } catch {
    list.textContent = "History is unavailable.";
  }
}

document.getElementById("clearHistory").addEventListener("click", () => {
  localStorage.removeItem("predictions");
  renderHistory();
});

renderHistory();
