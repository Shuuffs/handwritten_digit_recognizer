const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Initial canvas setup
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

// Make background white (not transparent)
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

// Drawing events
canvas.addEventListener("mousedown", () => {
  drawing = true;
});
canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath(); // reset path after each stroke
});
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

// Clear canvas
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white"; // ensure background stays white
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById("result").innerText = "Prediction: ";
}

// Send to backend
async function predict() {
  const imgData = canvas.toDataURL("image/png");

  const res = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-cache", // avoid cached responses
    },
    body: JSON.stringify({ image: imgData }),
  });

  const data = await res.json();
  document.getElementById("result").innerText =
    "Prediction: " + data.prediction;
}
