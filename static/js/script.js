function updatePredictions() {
    fetch("/predictions")
      .then((response) => response.json())
      .then((data) => {
        // parameter
        const container = document.getElementById("predictionsContainer");
        container.innerHTML = "";

        // generate data.predictions using doms
        data.predictions.forEach((pred) => {
          const bar = document.createElement("div");
          bar.className = "prediction-bar";

          const fill = document.createElement("div");
          fill.className = "bar-fill";
          fill.style.width = `${pred.confidence * 100}%`;

          const label = document.createElement("div");
          label.className = "bar-label";
          label.innerHTML = `
                <span>${pred.action}</span>
                <span>${(pred.confidence * 100).toFixed(1)}%</span>
            `;

          bar.appendChild(fill);
          bar.appendChild(label);
          container.appendChild(bar);
        });
      })
      .catch((error) => console.error("Error:", error));
  }

  setInterval(updatePredictions, 500);

  // function switchModel() {
  //   const model = document.getElementById("modelSelect").value;
  //   fetch("/switch_model", {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({ model: model }),
  //   });
  // }

  function switchModel() {
    const isGRU = document.getElementById('modelToggle').checked;
    const model = isGRU ? 'gru' : 'lstm';
    
    fetch("/switch_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: model })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Model switched:", data.message);
    });
}

  function updateThreshold() {
    const threshold = document.getElementById("thresholdSlider").value;
    document.getElementById("thresholdValue").textContent = threshold;
    fetch("/update_threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: threshold }),
    });
  }

  function startCamera() {
    fetch("/start_camera", { method: "POST" })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          // Refresh video feed with timestamp to prevent caching
          const videoElement = document.getElementById("videoFeed");
          videoElement.src = "{{ url_for('video_feed') }}?t=" + Date.now();
        }
      });
  }

  function stopCamera() {
    fetch("/stop_camera", { method: "POST" })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          document.getElementById("videoFeed").src = "";
        }
      });
  }