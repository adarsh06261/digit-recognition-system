const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const modelSelect = document.getElementById("modelSelect");
const resultBox = document.getElementById("result");
const predDigit = document.getElementById("predDigit");
const probsPre = document.getElementById("probs");

// drawing setup
let drawing = false;
ctx.lineWidth = 20;          // thick stroke helps MNIST-like digits
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#000000"; // black on white canvas

function start(e) {
  drawing = true;
  const {x, y} = pos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!drawing) return;
  const {x, y} = pos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function end() { drawing = false; }

function pos(e) {
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches?.[0];
  const clientX = touch ? touch.clientX : e.clientX;
  const clientY = touch ? touch.clientY : e.clientY;
  return { x: clientX - rect.left, y: clientY - rect.top };
}

// mouse & touch events
canvas.addEventListener("mousedown", start);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", end);

canvas.addEventListener("touchstart", (e) => { e.preventDefault(); start(e); }, {passive:false});
canvas.addEventListener("touchmove", (e) => { e.preventDefault(); draw(e); }, {passive:false});
canvas.addEventListener("touchend", (e) => { e.preventDefault(); end(e); }, {passive:false});

// clear canvas
function clearCanvas() {
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}
clearBtn.addEventListener("click", () => {
  clearCanvas();
  resultBox.classList.add("hidden");
});
clearCanvas();

// predict
predictBtn.addEventListener("click", async () => {
  // Grab 280x280, backend will center-crop + resize to 28x28
  const dataURL = canvas.toDataURL("image/png");
  const model = modelSelect.value;

  predDigit.textContent = "…";
  probsPre.textContent = "";
  resultBox.classList.remove("hidden");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ image: dataURL, model })
    });
    const json = await res.json();
    if (json.error) throw new Error(json.error);

    predDigit.textContent = json.prediction;
    if (json.probabilities) {
      // Pretty-print top-3
      const probs = json.probabilities.map((p, i) => ({ digit: i, p }));
      probs.sort((a,b) => b.p - a.p);
      const top = probs.slice(0, 3)
        .map(o => `${o.digit}: ${(o.p*100).toFixed(2)}%`)
        .join("\n");
      probsPre.textContent = `Top-3:\n${top}`;
    } else {
      probsPre.textContent = "(No probability available for RF)";
    }
  } catch (err) {
    predDigit.textContent = "Error";
    probsPre.textContent = err.message || String(err);
  }
});
