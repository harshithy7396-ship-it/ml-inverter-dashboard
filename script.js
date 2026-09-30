// ========== UTILITY FUNCTIONS ==========

function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function standardize(arr) {
  const m = mean(arr);
  const s = std(arr) || 1;
  return arr.map(v => (v - m) / s);
}

function sampleWithReplacement(arr, n) {
  const result = [];
  for (let i = 0; i < n; i++) {
    result.push(arr[Math.floor(Math.random() * arr.length)]);
  }
  return result;
}

// ========== RANDOM FOREST REGRESSOR ==========

class DecisionTreeRegressor {
  constructor(maxDepth = 12, minSamplesSplit = 5, maxFeatures = null) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxFeatures = maxFeatures;
    this.tree = null;
  }

  fit(X, y) {
    this.tree = this._buildTree(X, y, 0);
  }

  _buildTree(X, y, depth) {
    if (y.length < this.minSamplesSplit || depth >= this.maxDepth) {
      return { value: mean(y) };
    }

    const nFeatures = X[0].length;
    const featureIndices = this._selectFeatures(nFeatures);

    let bestFeature = -1;
    let bestThreshold = 0;
    let bestScore = Infinity;
    let bestLeftIdx = [];
    let bestRightIdx = [];

    for (const fIdx of featureIndices) {
      const values = X.map(row => row[fIdx]);
      const sorted = [...new Set(values)].sort((a, b) => a - b);

      // Sample thresholds to speed up
      const step = Math.max(1, Math.floor(sorted.length / 20));
      for (let t = 0; t < sorted.length; t += step) {
        const threshold = sorted[t];
        const leftIdx = [];
        const rightIdx = [];

        for (let i = 0; i < X.length; i++) {
          if (X[i][fIdx] <= threshold) leftIdx.push(i);
          else rightIdx.push(i);
        }

        if (leftIdx.length === 0 || rightIdx.length === 0) continue;

        const leftY = leftIdx.map(i => y[i]);
        const rightY = rightIdx.map(i => y[i]);
        const score = this._mse(leftY) * leftY.length + this._mse(rightY) * rightY.length;

        if (score < bestScore) {
          bestScore = score;
          bestFeature = fIdx;
          bestThreshold = threshold;
          bestLeftIdx = leftIdx;
          bestRightIdx = rightIdx;
        }
      }
    }

    if (bestFeature === -1) {
      return { value: mean(y) };
    }

    const leftX = bestLeftIdx.map(i => X[i]);
    const leftY = bestLeftIdx.map(i => y[i]);
    const rightX = bestRightIdx.map(i => X[i]);
    const rightY = bestRightIdx.map(i => y[i]);

    return {
      feature: bestFeature,
      threshold: bestThreshold,
      left: this._buildTree(leftX, leftY, depth + 1),
      right: this._buildTree(rightX, rightY, depth + 1),
    };
  }

  _selectFeatures(nFeatures) {
    const maxF = this.maxFeatures || Math.max(1, Math.floor(Math.sqrt(nFeatures)));
    const indices = Array.from({ length: nFeatures }, (_, i) => i);
    // Shuffle and take first maxF
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices.slice(0, maxF);
  }

  _mse(arr) {
    const m = mean(arr);
    return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
  }

  predict(X) {
    return X.map(row => this._predictOne(row, this.tree));
  }

  _predictOne(row, node) {
    if (node.value !== undefined) return node.value;
    if (row[node.feature] <= node.threshold) return this._predictOne(row, node.left);
    return this._predictOne(row, node.right);
  }
}

class RandomForestRegressor {
  constructor(nEstimators = 20, maxDepth = 12, minSamplesSplit = 5) {
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.trees = [];
    this.featureImportances = [];
  }

  fit(X, y) {
    this.trees = [];
    const n = X.length;
    const nFeatures = X[0].length;

    for (let t = 0; t < this.nEstimators; t++) {
      // Bootstrap sample
      const indices = sampleWithReplacement(Array.from({ length: n }, (_, i) => i), n);
      const sampleX = indices.map(i => X[i]);
      const sampleY = indices.map(i => y[i]);

      const tree = new DecisionTreeRegressor(this.maxDepth, this.minSamplesSplit);
      tree.fit(sampleX, sampleY);
      this.trees.push(tree);
    }

    // Compute feature importances via permutation
    this._computeFeatureImportances(X, y, nFeatures);
  }

  _computeFeatureImportances(X, y, nFeatures) {
    const basePreds = this.predict(X);
    const baseMSE = this._mse(y, basePreds);
    this.featureImportances = new Array(nFeatures).fill(0);

    for (let f = 0; f < nFeatures; f++) {
      // Permute feature f
      const permX = X.map(row => [...row]);
      const permValues = permX.map(row => row[f]);
      // Shuffle
      for (let i = permValues.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [permValues[i], permValues[j]] = [permValues[j], permValues[i]];
      }
      for (let i = 0; i < permX.length; i++) permX[i][f] = permValues[i];

      const permPreds = this.predict(permX);
      const permMSE = this._mse(y, permPreds);
      this.featureImportances[f] = Math.max(0, permMSE - baseMSE);
    }

    // Normalize
    const total = this.featureImportances.reduce((a, b) => a + b, 0);
    if (total > 0) {
      this.featureImportances = this.featureImportances.map(v => v / total);
    }
  }

  _mse(actual, predicted) {
    let sum = 0;
    for (let i = 0; i < actual.length; i++) sum += (actual[i] - predicted[i]) ** 2;
    return sum / actual.length;
  }

  predict(X) {
    const allPreds = this.trees.map(tree => tree.predict(X));
    return X.map((_, i) => mean(allPreds.map(preds => preds[i])));
  }
}

// ========== ANOMALY DETECTION (Isolation Forest style) ==========

function detectAnomalies(data, contamination = 0.03) {
  // Use z-score based isolation: points that are far from the mean in multiple dimensions
  const nCols = data[0].length;
  const standardized = [];

  for (let c = 0; c < nCols; c++) {
    const col = data.map(row => row[c]);
    standardized.push(standardize(col));
  }

  // Compute anomaly score for each row: sum of absolute z-scores
  const scores = data.map((_, i) => {
    let totalScore = 0;
    for (let c = 0; c < nCols; c++) {
      totalScore += Math.abs(standardized[c][i]);
    }
    return totalScore / nCols;
  });

  // Sort scores and find threshold at contamination percentile
  const sortedScores = [...scores].sort((a, b) => b - a);
  const thresholdIdx = Math.floor(contamination * sortedScores.length);
  const threshold = sortedScores[thresholdIdx] || sortedScores[0];

  return scores.map(s => s >= threshold ? -1 : 1); // -1 = anomaly, 1 = normal
}

// ========== DATA PROCESSING ==========

function parseCSVData(csvText) {
  const result = Papa.parse(csvText, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });

  // Filter to only numeric columns
  const numericCols = [];
  if (result.data.length > 0) {
    const firstRow = result.data[0];
    for (const col of Object.keys(firstRow)) {
      if (typeof firstRow[col] === "number" && !isNaN(firstRow[col])) {
        numericCols.push(col);
      }
    }
  }

  // Drop rows with NaN in numeric columns
  const cleanData = result.data.filter(row =>
    numericCols.every(col => typeof row[col] === "number" && !isNaN(row[col]))
  );

  return { data: cleanData, numericCols };
}

function autoDetectColumns(numericCols) {
  // Detect voltage, current, and output columns for power computation
  const lower = numericCols.map(c => c.toLowerCase());

  // Look for voltage columns (u_dc, v_dc, voltage, vdc, etc.)
  const voltageCols = numericCols.filter((c, i) =>
    /u_dc|v_dc|voltage|vdc|v_in|vin|u_k|vbus/i.test(c)
  );

  // Look for current columns (i_a, i_b, i_c, current, etc.)
  const currentCols = numericCols.filter((c, i) =>
    /^i_[a-c]|current|i_[a-c]_k$|^i[abc]/i.test(c)
  );

  // Look for output voltage columns (u_a, u_b, u_c, v_out, etc.)
  const outputVoltageCols = numericCols.filter((c, i) =>
    /u_[a-c]|v_out|vout|u_[a-c]_k/i.test(c)
  );

  // Look for power column
  const powerCol = numericCols.find((c, i) =>
    /power|watt|p_out|pout/i.test(c)
  );

  // Look for duty cycle columns
  const dutyCols = numericCols.filter((c, i) =>
    /^d_[a-c]|duty/i.test(c)
  );

  return { voltageCols, currentCols, outputVoltageCols, powerCol, dutyCols };
}

function computePower(data, numericCols) {
  const detected = autoDetectColumns(numericCols);

  // If power column already exists, use it
  if (detected.powerCol) {
    return data.map(row => row[detected.powerCol]);
  }

  // Try to compute power from voltage * current pairs
  if (detected.outputVoltageCols.length > 0 && detected.currentCols.length > 0) {
    return data.map(row => {
      let power = 0;
      const pairs = Math.min(detected.outputVoltageCols.length, detected.currentCols.length);
      for (let p = 0; p < pairs; p++) {
        power += (row[detected.outputVoltageCols[p]] || 0) * (row[detected.currentCols[p]] || 0);
      }
      return power;
    });
  }

  // If we have voltage and current columns, use V * I
  if (detected.voltageCols.length > 0 && detected.currentCols.length > 0) {
    return data.map(row => {
      let power = 0;
      for (const col of detected.currentCols) {
        power += (row[detected.voltageCols[0]] || 0) * (row[col] || 0);
      }
      return power;
    });
  }

  // Fallback: use the product of first two numeric columns as proxy
  if (numericCols.length >= 2) {
    return data.map(row => row[numericCols[0]] * row[numericCols[1]]);
  }

  // Last resort: use first numeric column
  return data.map(row => row[numericCols[0]]);
}

function selectFeatures(data, numericCols, powerValues) {
  // Select feature columns: all numeric columns that aren't the computed power
  const detected = autoDetectColumns(numericCols);

  // Prefer voltage + current columns as features
  let featureCols = [];

  if (detected.voltageCols.length > 0) featureCols.push(...detected.voltageCols);
  if (detected.currentCols.length > 0) featureCols.push(...detected.currentCols);
  if (detected.dutyCols.length > 0) featureCols.push(...detected.dutyCols);

  // If we don't have enough feature columns, use all numeric columns
  if (featureCols.length < 2) {
    featureCols = numericCols.filter(c => c !== detected.powerCol);
  }

  // Remove duplicates
  featureCols = [...new Set(featureCols)];

  // Limit to at most 8 features for performance
  if (featureCols.length > 8) {
    featureCols = featureCols.slice(0, 8);
  }

  return featureCols;
}

// ========== CHART MANAGEMENT ==========

const charts = {};

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

// ========== TAB MANAGEMENT ==========

document.addEventListener("DOMContentLoaded", function () {

    // Create Fault Detection tab
    createFaultDetectionTab();


    // =========================
    // TAB SWITCHING
    // =========================

    const tabBtns =
        document.querySelectorAll(".tab-btn");

    const tabContents =
        document.querySelectorAll(".tab-content");


    tabBtns.forEach(function(btn) {

        btn.addEventListener("click", function() {

            // Hide ALL tabs
            tabContents.forEach(function(content) {
                content.classList.remove("active");
            });


            // Remove active from ALL buttons
            tabBtns.forEach(function(button) {
                button.classList.remove("active");
            });


            // Activate clicked tab
            btn.classList.add("active");

            const target =
                document.getElementById(
                    btn.dataset.tab
                );

            if (target) {
                target.classList.add("active");
            }

        });

    });


    // =========================
    // UPLOAD HANDLERS
    // =========================

    setupUpload(
        "file-prediction",
        "upload-area-prediction",
        handlePrediction,
        "prediction-status"
    );


    setupUpload(
        "file-anomaly",
        "upload-area-anomaly",
        handleAnomaly,
        "anomaly-status"
    );


    setupUpload(
        "file-optimization",
        "upload-area-optimization",
        handleOptimization,
        "optimization-status"
    );

});

function setupUpload(inputId, areaId, handler, statusId) {
  const input = document.getElementById(inputId);
  const area = document.getElementById(areaId);

  if (!input || !area) return;

  input.addEventListener("change", (e) => {
    if (e.target.files.length > 0) processFile(e.target.files[0], handler, statusId);
  });

  area.addEventListener("dragover", (e) => {
    e.preventDefault();
    area.classList.add("drag-over");
  });

  area.addEventListener("dragleave", () => {
    area.classList.remove("drag-over");
  });

  area.addEventListener("drop", (e) => {
    e.preventDefault();
    area.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) processFile(e.dataTransfer.files[0], handler, statusId);
  });
}

function processFile(file, handler, statusId) {
  const statusEl = document.getElementById(statusId);
  if (!file.name.endsWith(".csv")) {
    statusEl.textContent = "Please upload a CSV file.";
    statusEl.className = "status-msg error";
    return;
  }

  statusEl.textContent = "Processing file: " + file.name + "...";
  statusEl.className = "status-msg loading";

  const reader = new FileReader();
  reader.onload = function (e) {
    try {
      handler(e.target.result, statusEl);
    } catch (err) {
      statusEl.textContent = "Error processing file: " + err.message;
      statusEl.className = "status-msg error";
    }
  };
  reader.readAsText(file);
}

// ========== POWER PREDICTION HANDLER ==========

function handlePrediction(csvText, statusEl) {
  const { data, numericCols } = parseCSVData(csvText);

  if (data.length < 10) {
    statusEl.textContent = "Not enough data rows. Need at least 10 rows.";
    statusEl.className = "status-msg error";
    return;
  }

  if (numericCols.length < 2) {
    statusEl.textContent = "Not enough numeric columns. Need at least 2.";
    statusEl.className = "status-msg error";
    return;
  }

  statusEl.textContent = "Computing power values and training Random Forest model...";

  // Use setTimeout to allow UI to update
  setTimeout(() => {
    try {
      const powerValues = computePower(data, numericCols);
      const featureCols = selectFeatures(data, numericCols, powerValues);

      // Build X matrix
      const X = data.map(row => featureCols.map(col => row[col]));
      const y = powerValues;

      // Subsample if dataset is large
      let trainX, trainY, testX, testY;
      const maxSamples = 3000;
      let sampleData = X;
      let sampleY = y;

      if (X.length > maxSamples) {
        // Sample uniformly
        const step = Math.floor(X.length / maxSamples);
        sampleData = [];
        sampleY = [];
        for (let i = 0; i < X.length; i += step) {
          sampleData.push(X[i]);
          sampleY.push(y[i]);
        }
      }

      // Train/test split (80/20, no shuffle to preserve time series)
      const splitIdx = Math.floor(sampleData.length * 0.8);
      trainX = sampleData.slice(0, splitIdx);
      trainY = sampleY.slice(0, splitIdx);
      testX = sampleData.slice(splitIdx);
      testY = sampleY.slice(splitIdx);

      // Train Random Forest
      const rf = new RandomForestRegressor(15, 10, 5);
      rf.fit(trainX, trainY);

      // Predict
      const predictions = rf.predict(testX);

      // Metrics
      const mseVal = mean(testY.map((v, i) => (v - predictions[i]) ** 2));
      const yMean = mean(testY);
      const ssTot = testY.reduce((s, v) => s + (v - yMean) ** 2, 0);
      const ssRes = testY.reduce((s, v, i) => s + (v - predictions[i]) ** 2, 0);
      const r2Val = ssTot > 0 ? 1 - ssRes / ssTot : 0;

      // Update UI
      document.getElementById("r2-score").textContent = r2Val.toFixed(4);
      document.getElementById("mse-value").textContent = mseVal.toFixed(2);
      document.getElementById("sample-count").textContent = sampleData.length;
      document.getElementById("prediction-results").style.display = "block";

      // Prediction chart - show subset for readability
      const maxDisplay = 200;
      const displayStep = Math.max(1, Math.floor(testY.length / maxDisplay));
      const displayActual = [];
      const displayPred = [];
      const displayLabels = [];
      for (let i = 0; i < testY.length; i += displayStep) {
        displayActual.push(testY[i]);
        displayPred.push(predictions[i]);
        displayLabels.push(i);
      }

      destroyChart("predictionChart");
      charts["predictionChart"] = new Chart(document.getElementById("predictionChart"), {
        type: "line",
        data: {
          labels: displayLabels,
          datasets: [
            {
              label: "Actual Power",
              data: displayActual,
              borderColor: "#3b82f6",
              backgroundColor: "rgba(59,130,246,0.1)",
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
            },
            {
              label: "Predicted Power (Random Forest)",
              data: displayPred,
              borderColor: "#f97316",
              borderWidth: 2,
              pointRadius: 0,
              borderDash: [5, 5],
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Random Forest: Actual vs Predicted Power" },
          },
          scales: {
            x: { title: { display: true, text: "Samples" } },
            y: { title: { display: true, text: "Power (W)" } },
          },
        },
      });

      // Feature importance chart
      destroyChart("featureImportanceChart");
      charts["featureImportanceChart"] = new Chart(document.getElementById("featureImportanceChart"), {
        type: "bar",
        data: {
          labels: featureCols,
          datasets: [
            {
              label: "Feature Importance",
              data: rf.featureImportances,
              backgroundColor: featureCols.map((_, i) =>
                `hsl(${(i * 360) / featureCols.length}, 70%, 55%)`
              ),
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Feature Importance in Power Prediction" },
          },
          scales: {
            x: { title: { display: true, text: "Features" } },
            y: { title: { display: true, text: "Importance" } },
          },
        },
      });

      statusEl.textContent = "Analysis complete! Used " + featureCols.length + " features from " + data.length + " data rows.";
      statusEl.className = "status-msg success";
    } catch (err) {
      statusEl.textContent = "Error during analysis: " + err.message;
      statusEl.className = "status-msg error";
    }
  }, 50);
}

// ========== ANOMALY DETECTION HANDLER ==========

function handleAnomaly(csvText, statusEl) {
  const { data, numericCols } = parseCSVData(csvText);

  if (data.length < 10) {
    statusEl.textContent = "Not enough data rows. Need at least 10 rows.";
    statusEl.className = "status-msg error";
    return;
  }

  statusEl.textContent = "Detecting anomalies...";

  setTimeout(() => {
    try {
      const powerValues = computePower(data, numericCols);

      // Select key columns for anomaly detection
      const featureCols = selectFeatures(data, numericCols, powerValues);
      const matrix = data.map(row => featureCols.map(col => row[col]));

      // Subsample if needed
      let sampleMatrix = matrix;
      let samplePower = powerValues;
      const maxSamples = 5000;
      if (matrix.length > maxSamples) {
        const step = Math.floor(matrix.length / maxSamples);
        sampleMatrix = [];
        samplePower = [];
        for (let i = 0; i < matrix.length; i += step) {
          sampleMatrix.push(matrix[i]);
          samplePower.push(powerValues[i]);
        }
      }

      // Run anomaly detection
      const anomalyLabels = detectAnomalies(sampleMatrix, 0.03);

      const totalSamples = sampleMatrix.length;
      const anomalyCount = anomalyLabels.filter(v => v === -1).length;
      const anomalyRate = ((anomalyCount / totalSamples) * 100).toFixed(2);

      // Update metrics
      document.getElementById("anomaly-total").textContent = totalSamples;
      document.getElementById("anomaly-count").textContent = anomalyCount;
      document.getElementById("anomaly-rate").textContent = anomalyRate + "%";
      document.getElementById("anomaly-results").style.display = "block";

      // Chart
      const maxDisplay = 500;
      const displayStep = Math.max(1, Math.floor(samplePower.length / maxDisplay));
      const displayPower = [];
      const displayAnomalyX = [];
      const displayAnomalyY = [];
      const displayLabels = [];

      for (let i = 0; i < samplePower.length; i += displayStep) {
        displayPower.push(samplePower[i]);
        displayLabels.push(i);
        if (anomalyLabels[i] === -1) {
          displayAnomalyX.push(displayLabels.length - 1);
          displayAnomalyY.push(samplePower[i]);
        }
      }

      // Create scatter data for anomalies
      const anomalyData = displayAnomalyX.map((x, idx) => ({
        x: displayLabels[x],
        y: displayAnomalyY[idx],
      }));

      destroyChart("anomalyChart");
      charts["anomalyChart"] = new Chart(document.getElementById("anomalyChart"), {
        type: "line",
        data: {
          labels: displayLabels,
          datasets: [
            {
              label: "Power",
              data: displayPower,
              borderColor: "#3b82f6",
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false,
            },
            {
              label: "Anomaly",
              data: anomalyData,
              type: "scatter",
              borderColor: "#ef4444",
              backgroundColor: "#ef4444",
              pointRadius: 4,
              showLine: false,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Abnormal Energy Consumption Detection" },
          },
          scales: {
            x: { title: { display: true, text: "Samples" } },
            y: { title: { display: true, text: "Power" } },
          },
        },
      });

      // Suggestions
      const avgPower = mean(samplePower);
      const suggestionsEl = document.getElementById("anomaly-suggestions");
      suggestionsEl.innerHTML = `
        <h4>Suggestions</h4>
        <div class="suggestion-list">
          <div class="suggestion-item warning">
            <strong>${anomalyCount} anomalies detected</strong> (${anomalyRate}% of data points)
            <br>These points show abnormal energy consumption patterns.
          </div>
          <div class="suggestion-item info">
            <strong>Average Power:</strong> ${avgPower.toFixed(2)} W
          </div>
          <div class="suggestion-item">
            <strong>Recommendation:</strong> Check appliances during anomaly periods.
            High deviations from normal patterns may indicate equipment malfunction,
            overloading, or inefficient operation.
          </div>
        </div>
      `;

      statusEl.textContent = "Anomaly detection complete! Found " + anomalyCount + " anomalies.";
      statusEl.className = "status-msg success";
    } catch (err) {
      statusEl.textContent = "Error during analysis: " + err.message;
      statusEl.className = "status-msg error";
    }
  }, 50);
}

// ========== ENERGY OPTIMIZATION HANDLER ==========

function handleOptimization(csvText, statusEl) {
  const { data, numericCols } = parseCSVData(csvText);

  if (data.length < 10) {
    statusEl.textContent = "Not enough data rows. Need at least 10 rows.";
    statusEl.className = "status-msg error";
    return;
  }

  statusEl.textContent = "Running energy optimization analysis...";

  setTimeout(() => {
    try {
      const powerValues = computePower(data, numericCols);
      const featureCols = selectFeatures(data, numericCols, powerValues);
      const matrix = data.map(row => featureCols.map(col => row[col]));

      // Subsample
      let sampleMatrix = matrix;
      let samplePower = [...powerValues];
      const maxSamples = 5000;
      if (matrix.length > maxSamples) {
        const step = Math.floor(matrix.length / maxSamples);
        sampleMatrix = [];
        samplePower = [];
        for (let i = 0; i < matrix.length; i += step) {
          sampleMatrix.push(matrix[i]);
          samplePower.push(powerValues[i]);
        }
      }

      // Detect anomalies
      const anomalyLabels = detectAnomalies(sampleMatrix, 0.03);
      const avgPower = mean(samplePower);

      // Optimize
      const optimizedPower = samplePower.map((p, i) => {
        if (anomalyLabels[i] === -1) return p * 0.75; // Aggressive reduction for anomalies
        if (Math.abs(p) > Math.abs(avgPower)) return p * 0.9; // 10% reduction for above-average
        return p;
      });

      const originalTotal = samplePower.reduce((a, b) => a + Math.abs(b), 0);
      const optimizedTotal = optimizedPower.reduce((a, b) => a + Math.abs(b), 0);
      const savingPercent = originalTotal > 0 ? ((originalTotal - optimizedTotal) / originalTotal * 100) : 0;

      // Update metrics
      document.getElementById("original-power").textContent = originalTotal.toFixed(2) + " W";
      document.getElementById("optimized-power").textContent = optimizedTotal.toFixed(2) + " W";
      document.getElementById("saving-percent").textContent = savingPercent.toFixed(2) + "%";
      document.getElementById("optimization-results").style.display = "block";

      // Chart
      const maxDisplay = 300;
      const displayStep = Math.max(1, Math.floor(samplePower.length / maxDisplay));
      const displayOriginal = [];
      const displayOptimized = [];
      const displayLabels = [];

      for (let i = 0; i < samplePower.length; i += displayStep) {
        displayOriginal.push(samplePower[i]);
        displayOptimized.push(optimizedPower[i]);
        displayLabels.push(i);
      }

      destroyChart("optimizationChart");
      charts["optimizationChart"] = new Chart(document.getElementById("optimizationChart"), {
        type: "line",
        data: {
          labels: displayLabels,
          datasets: [
            {
              label: "Original Power",
              data: displayOriginal,
              borderColor: "#3b82f6",
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
            },
            {
              label: "Optimized Power",
              data: displayOptimized,
              borderColor: "#22c55e",
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Smart Energy Optimization" },
          },
          scales: {
            x: { title: { display: true, text: "Samples" } },
            y: { title: { display: true, text: "Power (W)" } },
          },
        },
      });

      // Suggestions
      const anomalyCount = anomalyLabels.filter(v => v === -1).length;
      const aboveAvgCount = samplePower.filter(p => Math.abs(p) > Math.abs(avgPower)).length;

      const suggestionsEl = document.getElementById("optimization-suggestions");
      suggestionsEl.innerHTML = `
        <h4>Optimization Summary</h4>
        <div class="suggestion-list">
          <div class="suggestion-item success">
            <strong>Energy Saving: ${savingPercent.toFixed(2)}%</strong>
            <br>Total savings of ${(originalTotal - optimizedTotal).toFixed(2)} W across ${samplePower.length} samples.
          </div>
          <div class="suggestion-item warning">
            <strong>${anomalyCount} anomalous points</strong> received 25% power reduction (aggressive).
          </div>
          <div class="suggestion-item info">
            <strong>${aboveAvgCount} above-average points</strong> received 10% power reduction.
          </div>
          <div class="suggestion-item">
            <strong>Recommendations:</strong>
            <ul>
              <li>Check appliances during abnormal usage periods</li>
              <li>Reduce load during peak hours for above-average consumption</li>
              <li>Normal usage periods are already efficient</li>
            </ul>
          </div>
        </div>
      `;

      statusEl.textContent = "Optimization complete! Potential saving: " + savingPercent.toFixed(2) + "%";
      statusEl.className = "status-msg success";
    } catch (err) {
      statusEl.textContent = "Error during analysis: " + err.message;
      statusEl.className = "status-msg error";
    }
  }, 50);
}
// ==================== FAULT DETECTION ====================

function createFaultDetectionTab() {

    const tabs = document.querySelector(".tabs");

    if (!tabs) return;

    // Remove ALL existing Fault Detection buttons
    tabs.querySelectorAll(".tab-btn").forEach(function(btn) {
        if (btn.textContent.trim() === "Fault Detection") {
            btn.remove();
        }
    });

    // Remove old dynamically-created Fault Detection panel
    const oldPanel =
        document.getElementById("fault-detection");

    if (oldPanel) {
        oldPanel.remove();
    }

    // Continue with the rest of createFaultDetectionTab()
    // Add tab
    const faultTab = document.createElement("button");
    faultTab.className = "tab";
    faultTab.dataset.tab = "fault";
    faultTab.textContent = "Fault Detection";

    // Insert after Energy Optimization tab
    tabs.appendChild(faultTab);

    // Add content
    const faultContent = document.createElement("div");
    faultContent.className = "tab-pane";
  faultContent.style.display = "none";
    faultContent.id = "fault";

    faultContent.innerHTML = `
        <div class="card">
            <h2>Recorded Fault Detection</h2>
            <p>
                Upload hardware CSV containing
                <b>Fault_Status</b> and <b>DateTime</b>.
            </p>

            <div class="upload-area" id="upload-area-fault">
                <input type="file" id="file-fault" accept=".csv">
                <label for="file-fault">
                    Upload Fault Detection CSV
                </label>
            </div>

            <div id="fault-status"></div>

            <div class="metrics-grid">

                <div class="metric-card">
                    <h3>Fault Events</h3>
                    <div id="fault-event-count">0</div>
                </div>

                <div class="metric-card">
                    <h3>Fault Samples</h3>
                    <div id="fault-sample-count">0</div>
                </div>

                <div class="metric-card">
                    <h3>Total Fault Duration</h3>
                    <div id="fault-total-duration">0 sec</div>
                </div>

                <div class="metric-card">
                    <h3>Without Timestamp</h3>
                    <div id="fault-no-time">0</div>
                </div>

            </div>

            <div class="chart-container">
                <canvas id="faultSummaryChart"></canvas>
            </div>

            <h3>Fault Type Summary</h3>
            <div id="fault-summary-table"></div>

            <h3>Fault Event Timeline</h3>
            <div id="fault-event-table"></div>
        </div>
    `;

    tabContent.appendChild(faultContent);

    // Move Fault Detection after Energy Optimization
    const energyTab = tabs.querySelector('[data-tab="optimization"]');

    if (energyTab) {
        energyTab.after(faultTab);
    }

    const energyContent = document.getElementById("optimization");

    if (energyContent) {
        energyContent.after(faultContent);
    }
}


// Convert status into categories
function getFaultCategories(status) {

    const s = String(status)
        .toUpperCase()
        .trim();

    if (!s || s === "NORMAL") {
        return [];
    }

    const categories = [];

    if (s.includes("CURRENT")) {
        categories.push("CURRENT");
    }

    if (s.includes("VOLTAGE")) {
        categories.push("VOLTAGE");
    }

    if (s.includes("TEMP")) {
        categories.push("TEMPERATURE");
    }

    if (s.includes("MOSFET")) {
        categories.push("MOSFET");
    }

    return categories;
}


// Parse DateTime
function parseHardwareTimestamp(value) {

    if (!value || String(value).trim() === "") {
        return null;
    }

    const date = new Date(value);

    if (isNaN(date.getTime())) {
        return null;
    }

    return date;
}


// Format duration
function formatDuration(seconds) {

    if (seconds === null || seconds === undefined) {
        return "Unknown";
    }

    seconds = Math.round(seconds);

    const hours = Math.floor(seconds / 3600);
    seconds %= 3600;

    const minutes = Math.floor(seconds / 60);
    seconds %= 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m ${seconds}s`;
    }

    if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
    }

    return `${seconds}s`;
}


// Calculate sampling interval
function calculateSamplingInterval(rows) {

    const times = rows
        .map(row => parseHardwareTimestamp(row.DateTime))
        .filter(t => t !== null)
        .sort((a, b) => a - b);

    if (times.length < 2) {
        return 5000;
    }

    const differences = [];

    for (let i = 1; i < times.length; i++) {

        const diff = times[i] - times[i - 1];

        if (diff > 0) {
            differences.push(diff);
        }
    }

    if (!differences.length) {
        return 5000;
    }

    differences.sort((a, b) => a - b);

    return differences[Math.floor(differences.length / 2)];
}


// Main fault analysis
function analyseRecordedFaults(data) {

    const samplingInterval =
        calculateSamplingInterval(data);

    const events = [];

    let currentEvent = null;

    for (let i = 0; i < data.length; i++) {

        const status = String(data[i].Fault_Status || "")
            .trim();

        const categories = getFaultCategories(status);

        // NORMAL row
        if (categories.length === 0) {

            if (currentEvent) {
                events.push(currentEvent);
                currentEvent = null;
            }

            continue;
        }

        const timestamp =
            parseHardwareTimestamp(data[i].DateTime);

        // Start new event
        if (!currentEvent) {

            currentEvent = {
                start: timestamp,
                end: timestamp,
                samples: 0,
                statuses: [],
                categories: new Set(),
                hasTimestamp: false
            };
        }

        currentEvent.samples++;

        currentEvent.statuses.push(status);

        categories.forEach(c =>
            currentEvent.categories.add(c)
        );

        if (timestamp) {

            currentEvent.hasTimestamp = true;

            if (
                !currentEvent.start ||
                timestamp < currentEvent.start
            ) {
                currentEvent.start = timestamp;
            }

            if (
                !currentEvent.end ||
                timestamp > currentEvent.end
            ) {
                currentEvent.end = timestamp;
            }
        }
    }

    // Last event
    if (currentEvent) {
        events.push(currentEvent);
    }


    // Calculate duration
    events.forEach(event => {

        if (
            event.start &&
            event.end
        ) {

            event.duration =
                (event.end - event.start)
                + samplingInterval;

        } else {

            event.duration = null;
        }
    });


    return {
        events,
        samplingInterval
    };
}


// Render summary
function renderFaultSummaryTable(events) {

    const summary = {};

    ["CURRENT", "VOLTAGE", "TEMPERATURE", "MOSFET"]
        .forEach(type => {

            summary[type] = {
                events: 0,
                samples: 0,
                duration: 0
            };

        });


    events.forEach(event => {

        event.categories.forEach(type => {

            summary[type].events++;

            summary[type].samples +=
                event.samples;

            if (event.duration !== null) {

                summary[type].duration +=
                    event.duration / 1000;
            }

        });

    });


    let html = `
        <table class="data-table">

            <thead>
                <tr>
                    <th>Fault Type</th>
                    <th>Events</th>
                    <th>Samples</th>
                    <th>Total Duration</th>
                </tr>
            </thead>

            <tbody>
    `;


    Object.keys(summary).forEach(type => {

        html += `
            <tr>
                <td>${type}</td>
                <td>${summary[type].events}</td>
                <td>${summary[type].samples}</td>
                <td>
                    ${formatDuration(
                        summary[type].duration
                    )}
                </td>
            </tr>
        `;

    });


    html += `
            </tbody>
        </table>
    `;


    document.getElementById(
        "fault-summary-table"
    ).innerHTML = html;

    return summary;
}


// Render event timeline
function renderFaultEventTable(events) {

    let html = `
        <table class="data-table">

            <thead>
                <tr>
                    <th>Event</th>
                    <th>Fault Type</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Samples</th>
                    <th>Duration</th>
                </tr>
            </thead>

            <tbody>
    `;


    events.forEach((event, index) => {

        const categories =
            [...event.categories].join(" + ");


        html += `
            <tr>

                <td>Fault ${index + 1}</td>

                <td>
                    ${categories || "UNKNOWN"}
                </td>

                <td>
                    ${
                        event.start
                        ? event.start.toLocaleString()
                        : "No timestamp"
                    }
                </td>

                <td>
                    ${
                        event.end
                        ? event.end.toLocaleString()
                        : "No timestamp"
                    }
                </td>

                <td>
                    ${event.samples}
                </td>

                <td>
                    ${formatDuration(
                        event.duration === null
                        ? null
                        : event.duration / 1000
                    )}
                </td>

            </tr>
        `;

    });


    html += `
            </tbody>
        </table>
    `;


    document.getElementById(
        "fault-event-table"
    ).innerHTML = html;
}


// Chart
function renderFaultChart(summary) {

    const canvas =
        document.getElementById(
            "faultSummaryChart"
        );

    if (!canvas) return;

    if (window.faultChart) {
        window.faultChart.destroy();
    }

    window.faultChart = new Chart(
        canvas,
        {
            type: "bar",

            data: {

                labels: Object.keys(summary),

                datasets: [
                    {
                        label: "Fault Events",

                        data: Object.values(summary)
                            .map(x => x.events)
                    }
                ]

            },

            options: {
                responsive: true,

                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        }
    );
}


// File handler
function handleFaultDetection(csvText, statusEl) {

    const parsed = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false,

        // Remove BOM and spaces from CSV headers
        transformHeader: function(header) {
            return String(header)
                .replace(/^\uFEFF/, "")
                .trim();
        }
    });

    let data = parsed.data || [];

    if (data.length === 0) {
        statusEl.textContent = "CSV contains no data.";
        statusEl.className = "status-msg error";
        return;
    }

    // Get actual CSV headers
    const columns = Object.keys(data[0]);

    console.log("CSV columns:", columns);

    // Find Fault_Status irrespective of spaces/case
    function findColumn(names) {

        return columns.find(function(col) {

            const cleanCol = String(col)
                .toLowerCase()
                .replace(/[\s_]+/g, "");

            return names.some(function(name) {

                const cleanName = name
                    .toLowerCase()
                    .replace(/[\s_]+/g, "");

                return cleanCol === cleanName;
            });
        });
    }

    const faultColumn = findColumn([
        "Fault_Status",
        "Fault Status",
        "FaultStatus",
        "Fault"
    ]);

    const dateColumn = findColumn([
        "DateTime",
        "Date Time",
        "Timestamp",
        "Time"
    ]);

    // DEBUG MESSAGE
    console.log("Fault column:", faultColumn);
    console.log("Date column:", dateColumn);

    if (!faultColumn) {

        statusEl.textContent =
            "Fault_Status not found. Columns detected: " +
            columns.join(", ");

        statusEl.className = "status-msg error";

        return;
    }

    if (!dateColumn) {

        statusEl.textContent =
            "DateTime not found. Columns detected: " +
            columns.join(", ");

        statusEl.className = "status-msg error";

        return;
    }

    // Convert actual CSV columns to standard names
    data = data.map(function(row) {

        return {
            ...row,

            Fault_Status: row[faultColumn],
            DateTime: row[dateColumn]
        };

    });

    statusEl.textContent =
        "Analysing faults...";

    statusEl.className =
        "status-msg loading";


    setTimeout(function() {

        try {

            const result =
                analyseRecordedFaults(data);


            // =========================
            // UPDATE METRICS
            // =========================

            document.getElementById(
                "fault-events"
            ).textContent =
                result.events.length;


            document.getElementById(
                "fault-samples"
            ).textContent =
                result.totalFaultSamples;


            const totalDuration =
                result.events.reduce(
                    function(sum, event) {

                        return sum +
                            (
                                event.duration !== null
                                    ? event.duration
                                    : 0
                            );

                    },
                    0
                );


            document.getElementById(
                "fault-duration"
            ).textContent =
                formatDuration(totalDuration);


            document.getElementById(
                "fault-untimed"
            ).textContent =
                result.untimedFaultSamples;


            // =========================
            // FAULT TABLE
            // =========================

            renderFaultSummaryTable(
                result.summary
            );


            renderFaultEventTable(
                result.events
            );


            // =========================
            // FAULT CHART
            // =========================

            renderFaultChart(
                result.summary
            );


            // Show results
            document.getElementById(
                "fault-results"
            ).style.display = "block";


            statusEl.textContent =
                "Fault analysis complete! " +
                result.events.length +
                " fault events found.";

            statusEl.className =
                "status-msg success";


        } catch (error) {

            console.error(
                "Fault analysis error:",
                error
            );

            statusEl.textContent =
                "Error: " +
                error.message;

            statusEl.className =
                "status-msg error";
        }

    }, 50);
}
