const API = new URLSearchParams(location.search).get('api') || 'http://127.0.0.1:8000';
const $ = (id) => document.getElementById(id);
const fileInput = $('dataset-file');
let selectedFile = null;
let health = null;

function toast(message, error = false) {
  const el = $('toast'); el.textContent = message; el.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { el.className = 'toast'; }, 5200);
}
function formatBytes(bytes) { return bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`; }
function setStatus(text, state = '') { $('api-status').className = `api-status ${state}`; $('api-status').lastElementChild.textContent = text; }

async function checkHealth() {
  try {
    const response = await fetch(`${API}/api/health`);
    if (!response.ok) throw new Error(`Model API responded ${response.status}`);
    health = await response.json();
    if (health.base_model) $('base-model-name').textContent = health.base_model;
    if (health.fine_tuned_model) $('tuned-model-name').textContent = health.fine_tuned_model;
    if (health.adapter_available) {
      setStatus(`Connected to ${health.provider}; both model IDs are configured.`, 'ready');
      $('adapter-badge').textContent = 'Model ready';
    } else {
      setStatus(health.configuration_message || 'Configure the fine-tuned model endpoint before running analysis.', 'error');
      $('adapter-badge').textContent = 'Not configured';
    }
  } catch (error) {
    health = null;
    setStatus('Model API offline. Start the local API and configure the trained adapter.', 'error');
    $('adapter-badge').textContent = 'Not connected';
  }
  updateRunButton();
}

function updateRunButton() { $('run-analysis').disabled = !selectedFile || !health?.adapter_available; }
function selectFile(file) {
  if (!file) return;
  const ext = file.name.toLowerCase().split('.').pop();
  if (!['csv', 'xlsx', 'xls'].includes(ext)) { toast('Choose a CSV or Excel file.', true); return; }
  selectedFile = file;
  $('file-name').textContent = file.name; $('file-size').textContent = formatBytes(file.size);
  $('file-row').hidden = false; $('dropzone').classList.add('has-file');
  updateRunButton(); inspectFile(file);
}

async function inspectFile(file) {
  try {
    const data = new FormData(); data.append('file', file);
    // Preview is served by the same parser contract, so Excel and CSV are inspected consistently.
    const response = await fetch(`${API}/api/inspect-file`, { method: 'POST', body: data });
    if (!response.ok) return;
    const result = await response.json();
    $('record-count').textContent = result.records.toLocaleString();
    $('column-count').textContent = result.columns.length;
    $('class-count').textContent = result.classes ?? '—';
    $('dataset-type').textContent = result.domain || 'Digital Items';
    $('overview-foot').textContent = result.has_labels ? 'Ground-truth labels found; evaluation metrics will be computed.' : 'No actual_action labels; recommendations will be shown without accuracy metrics.';
  } catch (_) { /* Keep the no-analysis overview until the API can inspect the file. */ }
}

fileInput.addEventListener('change', (event) => selectFile(event.target.files?.[0]));
$('remove-file').addEventListener('click', () => {
  selectedFile = null; fileInput.value = ''; $('file-row').hidden = true; $('record-count').textContent = '—'; $('column-count').textContent = '—'; $('class-count').textContent = '—'; $('dataset-type').textContent = 'Awaiting upload'; $('overview-foot').textContent = 'Upload a dataset to see its structure.'; updateRunButton();
});
const drop = $('dropzone');
for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove('dragover'); });
drop.addEventListener('drop', (event) => selectFile(event.dataTransfer.files?.[0]));

$('run-analysis').addEventListener('click', async () => {
  if (!selectedFile || !health?.adapter_available) return;
  const button = $('run-analysis'); button.disabled = true; button.innerHTML = '<span>◌</span> Running analysis…';
  setStatus('Running base and fine-tuned inference on the uploaded records…');
  const form = new FormData(); form.append('file', selectedFile);
  try {
    const response = await fetch(`${API}/api/analyze-file`, { method: 'POST', body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `Analysis failed (${response.status}).`);
    renderResult(result);
    setStatus(result.metrics ? 'Evaluation complete. Metrics are calculated from this upload.' : 'Predictions complete. Ground-truth labels were not provided, so metrics are unavailable.', 'ready');
    toast(result.metrics ? `Evaluated ${result.records} records using both models.` : `Generated recommendations for ${result.records} records.`);
  } catch (error) {
    setStatus('Analysis did not complete.', 'error'); toast(error.message, true);
  } finally {
    button.innerHTML = '<span>▶</span> Run Analysis'; updateRunButton();
  }
});

function pct(value) { return `${(value * 100).toFixed(1)}%`; }
function gain(base, tuned) { const pp = (tuned - base) * 100; return `${pp >= 0 ? '+' : ''}${pp.toFixed(1)} pp`; }
function fillMetrics(prefix, metrics) {
  $( `${prefix}-accuracy`).textContent = pct(metrics.accuracy);
  $(`${prefix}-precision`).textContent = pct(metrics.precision);
  $(`${prefix}-recall`).textContent = pct(metrics.recall);
  $(`${prefix}-f1`).textContent = pct(metrics.f1);
}
function renderChart(base, tuned) {
  const names = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
  const b = [base.accuracy, base.precision, base.recall, base.f1], t = [tuned.accuracy, tuned.precision, tuned.recall, tuned.f1];
  const w = 560, h = 168, left = 36, top = 14, bottom = 29, chartH = h - top - bottom, groupW = 122;
  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Measured model metric comparison">`;
  for (let tick = 0; tick <= 100; tick += 20) { const y = top + chartH * (1 - tick / 100); svg += `<line x1="${left}" x2="${w - 6}" y1="${y}" y2="${y}" stroke="#dce5f1"/><text x="2" y="${y + 4}" fill="#526888" font-size="10">${tick}%</text>`; }
  names.forEach((name, i) => {
    const x = left + 15 + i * groupW, bh = Math.max(0, b[i] * chartH), th = Math.max(0, t[i] * chartH);
    svg += `<rect x="${x}" y="${top + chartH - bh}" width="34" height="${bh}" rx="4" fill="#348df4"/><rect x="${x + 39}" y="${top + chartH - th}" width="34" height="${th}" rx="4" fill="#18b579"/>`;
    if (b[i] > 0) svg += `<text x="${x + 17}" y="${top + chartH - bh - 4}" text-anchor="middle" font-size="10" fill="#0d1745">${Math.round(b[i] * 100)}%</text>`;
    if (t[i] > 0) svg += `<text x="${x + 56}" y="${top + chartH - th - 4}" text-anchor="middle" font-size="10" fill="#0d1745">${Math.round(t[i] * 100)}%</text>`;
    svg += `<text x="${x + 37}" y="${h - 5}" text-anchor="middle" font-size="10" fill="#162858">${name}</text>`;
  });
  svg += '</svg>'; $('performance-chart').innerHTML = svg;
}
function renderMatrix(id, matrix, labels, green = false) {
  const max = Math.max(1, ...matrix.flat());
  let html = '<div class="matrix-grid"><div></div>' + labels.map(l => `<span>${escapeHtml(l)}</span>`).join('');
  matrix.forEach((row, ri) => { html += `<span class="row-label">${escapeHtml(labels[ri])}</span>`; row.forEach(value => { const alpha = .08 + .86 * value / max; const bg = green ? `rgba(19,173,114,${alpha})` : `rgba(30,103,225,${alpha})`; html += `<b style="background:${bg};color:${value/max>.62?'white':'#17264a'}">${value}</b>`; }); });
  html += '</div><div class="matrix-axes"><span>Actual ↓</span><span>Predicted →</span></div>'; $(id).className = `matrix-render ${green ? 'matrix-green' : ''}`; $(id).innerHTML = html;
}
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function renderPredictions(rows) {
  const body = $('prediction-rows');
  body.innerHTML = rows.map((r, i) => {
    const actual = r.actual_action || '—', ba = r.base.action, fa = r.fine_tuned.action;
    let result = 'Prediction';
    if (actual !== '—') result = fa === actual ? '<span class="correct">✓ Correct</span>' : '<span class="incorrect">✕ Incorrect</span>';
    const detail = r.item_text || '';
    return `<tr title="${escapeHtml(r.fine_tuned.reason)}"><td>${i + 1}</td><td class="item-detail">${escapeHtml(detail)}</td><td>${escapeHtml(actual)}</td><td class="prediction-action action-${escapeHtml(ba)}">${escapeHtml(ba)}</td><td class="prediction-action action-${escapeHtml(fa)}">${escapeHtml(fa)}</td><td>${result}</td></tr>`;
  }).join('');
  $('prediction-count').textContent = `${rows.length} rows`;
  $('table-foot').textContent = `Showing all ${rows.length} predictions. Scroll the table to view every row.`;
}
function renderResult(result) {
  $('record-count').textContent = result.records.toLocaleString();
  $('overview-foot').textContent = result.metrics ? (result.dataset_provenance === 'synthetic' ? 'Scores are real model outputs on a synthetic held-out set; they may not reflect real-world performance.' : 'Metrics calculated on this labeled dataset.') : 'Unlabeled records: metrics unavailable until actual_action labels are supplied.';
  renderPredictions(result.predictions || []);
  if (!result.metrics) {
    for (const id of ['base-accuracy','base-precision','base-recall','base-f1','tuned-accuracy','tuned-precision','tuned-recall','tuned-f1','accuracy-gain','precision-gain','recall-gain','f1-gain']) $(id).textContent = '—';
    $('performance-chart').innerHTML = '<div class="chart-empty">Predictions are ready. Add actual_action labels to calculate comparable model metrics.</div>';
    $('base-matrix').className = 'matrix-empty'; $('base-matrix').textContent = 'Ground-truth labels not provided.';
    $('tuned-matrix').className = 'matrix-empty green-empty'; $('tuned-matrix').textContent = 'Ground-truth labels not provided.';
    $('insight').innerHTML = '<span>♜</span><div><strong>Recommendations ready for review</strong><p>Metrics require actual_action labels. Recommendations do not execute cleanup actions.</p></div>';
    return;
  }
  const b = result.metrics.base, t = result.metrics.fine_tuned;
  fillMetrics('base', b); fillMetrics('tuned', t); renderChart(b, t);
  $('accuracy-gain').textContent = gain(b.accuracy, t.accuracy); $('precision-gain').textContent = gain(b.precision, t.precision); $('recall-gain').textContent = gain(b.recall, t.recall); $('f1-gain').textContent = gain(b.f1, t.f1);
  renderMatrix('base-matrix', b.confusion_matrix, b.labels); renderMatrix('tuned-matrix', t.confusion_matrix, t.labels, true);
  const better = t.f1 >= b.f1;
  $('insight').innerHTML = `<span>♜</span><div><strong>${better ? 'Fine-tuned model performed better on this upload' : 'Base model performed better on this upload'}</strong><p>Macro F1 changed by ${gain(b.f1, t.f1)} on ${result.records} labeled records. Review per-class scores and predictions before relying on recommendations.</p></div>`;
}

checkHealth();
