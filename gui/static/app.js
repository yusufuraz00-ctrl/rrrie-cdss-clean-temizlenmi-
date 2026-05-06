/* Merged from app.final-result.js. */
(function attachDecisionBoardView() {
  function esc(value) {
    const raw = String(value ?? "");
    if (typeof window.escapeHTML === "function") return window.escapeHTML(raw);
    return raw.replace(/[&<>'"`]/g, (token) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
      "`": "&#96;",
    }[token] || token));
  }

  function clamp01(value) {
    const score = Number(value || 0);
    if (!Number.isFinite(score)) return 0;
    return Math.max(0, Math.min(1, score));
  }

  function pct(value) {
    return `${Math.round(clamp01(value) * 100)}%`;
  }

  function label(value, fallback = "--") {
    const text = String(value || "").trim().replace(/_/g, " ");
    return text || fallback;
  }

  function tone(value) {
    const score = clamp01(value);
    if (score >= 0.75) return "high";
    if (score >= 0.5) return "med";
    return "low";
  }

  function metric(value, digits = 1) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? number.toFixed(digits) : "0.0";
  }

  function values(items) {
    return Array.isArray(items) ? items.filter(Boolean) : [];
  }

  function icon(name) {
    return `<span class="material-symbols-outlined">${esc(name)}</span>`;
  }

  function list(items, emptyText, className = "decision-list") {
    const safe = values(items);
    if (!safe.length) return `<p class="decision-muted">${esc(emptyText)}</p>`;
    return `<ul class="${esc(className)}">${safe.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>`;
  }

  function scoreChip(value, suffix = "") {
    return `<span class="decision-score ${tone(value)}">${pct(value)}${suffix ? ` ${esc(suffix)}` : ""}</span>`;
  }

  function section(title, subtitle, body, className = "") {
    const classes = ["decision-panel", className].filter(Boolean).join(" ");
    return `
      <section class="${esc(classes)}">
        <div class="decision-panel-head">
          <h3>${esc(title)}</h3>
          ${subtitle ? `<span>${esc(subtitle)}</span>` : ""}
        </div>
        ${body}
      </section>
    `;
  }

  function collapsibleSection(title, subtitle, body, open = false, className = "") {
    const classes = ["decision-panel", "decision-collapsible", className].filter(Boolean).join(" ");
    return `
      <details class="${esc(classes)}"${open ? " open" : ""}>
        <summary>
          <span>${esc(title)}</span>
          ${subtitle ? `<small>${esc(subtitle)}</small>` : ""}
        </summary>
        <div class="decision-collapsible-body">
          ${body}
        </div>
      </details>
    `;
  }

  function evidenceCoverageChip(label, coverage, starved) {
    const value = Number.isFinite(coverage) ? coverage : null;
    if (starved) {
      return `<span class="decision-evidence-coverage starved" title="No supporting evidence retrieved for this candidate">no evidence</span>`;
    }
    if (value === null) return "";
    const pctVal = Math.round(Math.max(0, Math.min(1, value)) * 100);
    const toneCls = pctVal >= 60 ? "high" : (pctVal >= 30 ? "med" : "low");
    return `<span class="decision-evidence-coverage ${toneCls}" title="Fraction of retrieved evidence that mentions this candidate">${pctVal}% evidence</span>`;
  }

  function groundingChip(verdict, score) {
    if (!verdict || verdict === "unchecked") return "";
    const pctVal = Math.round(Math.max(0, Math.min(1, Number(score || 0))) * 100);
    const cls = verdict === "pass" ? "high" : (verdict === "demote" ? "med" : "low");
    const text = verdict === "pass" ? `grounded ${pctVal}%`
              : verdict === "demote" ? `partial grounding ${pctVal}%`
              : `dropped ${pctVal}%`;
    return `<span class="decision-grounding-chip ${cls}" title="Inline grounding verdict for rationale">${text}</span>`;
  }

  function candidatesHtml(candidates, view) {
    const safe = values(candidates).filter((item) => item && (item.label || item.dx));
    if (!safe.length) {
      return `<p class="decision-muted">No structured ranked differential was attached.</p>`;
    }
    const coverageMap = (view && view.coverage_per_candidate) || {};
    const starved = new Set(values(view && view.starved_candidates));
    return `
      <div class="decision-ranked-list">
        ${safe.slice(0, 8).map((candidate, index) => {
          const name = candidate.label || candidate.dx || "candidate";
          const rationale = values(candidate.rationale);
          const missing = values(candidate.evidence_needed || candidate.missing_confirmers);
          const unsafe = values(candidate.unsafe_interventions);
          const coverage = coverageMap[name];
          const isStarved = starved.has(name);
          const coverageHtml = evidenceCoverageChip(name, coverage, isStarved);
          const grounding = groundingChip(candidate.grounding_verdict, candidate.grounding_score);
          const rationaleText = rationale.length ? rationale.slice(0, 2).join(" ") : "No rationale snippet attached.";
          return `
            <article class="decision-dx-row">
              <div class="decision-rank">${index + 1}</div>
              <div class="decision-dx-main">
                <div class="decision-dx-title">
                  <strong>${esc(name)}</strong>
                  ${candidate.must_not_miss ? `<span class="decision-badge critical">must not miss</span>` : ""}
                  ${candidate.family_label ? `<span class="decision-badge">${esc(candidate.family_label)}</span>` : ""}
                  ${coverageHtml}
                  ${grounding}
                </div>
                <p class="decision-rationale${rationale.length ? "" : " decision-muted"}">${esc(rationaleText)}</p>
                ${missing.length ? `<small>Needed: ${missing.slice(0, 4).map(esc).join(", ")}</small>` : ""}
                ${unsafe.length ? `<small class="critical">Unsafe assumptions: ${unsafe.slice(0, 3).map(esc).join(", ")}</small>` : ""}
              </div>
              ${scoreChip(candidate.score || candidate.conf)}
            </article>
          `;
        }).join("")}
      </div>
    `;
  }

  function evidenceHtml(evidence) {
    const items = values(evidence?.items);
    if (!items.length) return `<p class="decision-muted">No linked source item was attached to this packet.</p>`;
    // Stash items globally so the drawer can read them without re-rendering.
    if (!window.__rrrieEvidence) window.__rrrieEvidence = {};
    return `
      <div class="decision-evidence-list">
        ${items.slice(0, 12).map((item, idx) => {
          const title = item.title || item.citation || item.source || "Evidence item";
          const relation = item.relation_type ? label(item.relation_type) : "support";
          const status = item.verification_status ? label(item.verification_status) : "unverified";
          const trust = Number.isFinite(Number(item.trust_score)) ? Math.round(Number(item.trust_score) * 100) : null;
          const eid = `ev_${Date.now().toString(36)}_${idx}`;
          window.__rrrieEvidence[eid] = item;
          return `
            <article class="decision-evidence-item" tabindex="0" role="button" data-evidence-id="${esc(eid)}" aria-label="Open evidence detail">
              <strong>${esc(title)}</strong>
              <span>${esc(relation)} / ${esc(status)}${trust !== null ? ` · trust ${trust}%` : ""}</span>
            </article>
          `;
        }).join("")}
      </div>
    `;
  }

  function abstentionBannerHtml(view) {
    if (!view || !view.abstention_recommended) return "";
    const reason = String(view.abstention_reason || "low_margin_and_high_grounding_risk").replace(/_/g, " ");
    const margin = Number(view.abstention_margin || 0);
    const risk = Number(view.abstention_grounding_risk || 0);
    return `
      <aside class="decision-abstention-banner" role="status">
        <div class="decision-abstention-icon">${icon("warning")}</div>
        <div class="decision-abstention-body">
          <strong>System recommends abstention pending objective data.</strong>
          <p>The fused-score margin (${margin.toFixed(2)}) is below the safety threshold and grounding risk is ${(risk * 100).toFixed(0)}%. Treat the differential below as candidate hypotheses, not as a stable diagnosis. Reason: <em>${esc(reason)}</em>.</p>
        </div>
      </aside>
    `;
  }

  function evidenceQualityFooter(view) {
    const starved = values(view && view.starved_candidates);
    if (!starved.length) return "";
    return `<p class="decision-evidence-warning">${icon("error")}<span>No external evidence was retrieved for: <strong>${starved.map(esc).join(", ")}</strong>. Reasoning for these candidates is narrative-only.</span></p>`;
  }

  function exportButtonHtml(view) {
    const caseId = String(view?.case_id || "case");
    return `
      <button type="button" class="decision-export-btn" data-action="rrrie-export" data-case="${esc(caseId)}" title="Download decision packet (JSON) and open print view">
        ${icon("download")} Export packet
      </button>
    `;
  }

  function safetyHtml(safety) {
    const issues = values(safety?.issues);
    const actions = values(safety?.mandatory_actions);
    const blocked = values(safety?.blocked_orders);
    const issueRows = issues.map((issue) => `
      <li>
        <strong>${esc(label(issue.severity || issue.issue_type || "flag"))}</strong>
        <span>${esc(issue.detail || issue.description || issue.issue_type || "Safety flag")}</span>
      </li>
    `).join("");
    const actionRows = actions.map((item) => `<li>${esc(item.action || item)}</li>`).join("");
    const blockedRows = blocked.map((item) => `<li>${esc(item.order_name || item)}${item.fatal_risk ? `: ${esc(item.fatal_risk)}` : ""}</li>`).join("");
    return `
      <p class="decision-copy">${esc(safety?.summary || "No safety summary was attached.")}</p>
      ${issues.length ? `<h4>Verification issues</h4><ul class="decision-safety-list">${issueRows}</ul>` : ""}
      ${actions.length ? `<h4>Mandatory actions</h4><ul class="decision-list">${actionRows}</ul>` : ""}
      ${blocked.length ? `<h4>Blocked orders</h4><ul class="decision-list critical">${blockedRows}</ul>` : ""}
    `;
  }

  function timingHtml(timing) {
    const timingSource = timing?.stages || timing || {};
    const entries = Object.entries(timingSource).filter(([, value]) => value && (value.time || value.model_wait_s || value.llm_calls));
    if (!entries.length) return `<p class="decision-muted">No stage profiler data was attached.</p>`;
    return entries.map(([name, value]) => `
      <div class="decision-trace-row">
        <strong>${esc(label(name))}</strong>
        <span>${metric(value.time)}s wall</span>
        <span>${metric(value.model_wait_s)}s model</span>
        <span>${Number(value.llm_calls || 0)} calls</span>
      </div>
    `).join("");
  }

  function runtimeHtml(runtime, timing, trace) {
    const chips = [
      ["Operation", runtime?.operation_mode],
      ["Profile", runtime?.runtime_profile],
      ["Speed", runtime?.runtime_speed_profile],
      ["Engine", runtime?.engine_mode],
      ["Model", runtime?.engine_model],
    ].filter(([, value]) => String(value || "").trim());
    const traceItems = values(trace);
    return `
      <div class="decision-runtime-chips">
        ${chips.length ? chips.map(([name, value]) => `<span><strong>${esc(name)}</strong>${esc(label(value))}</span>`).join("") : `<span><strong>Runtime</strong>No runtime metadata</span>`}
      </div>
      <div class="decision-trace-stack">${timingHtml(timing)}</div>
      ${traceItems.length ? `<h4>Reasoning trace</h4>${list(traceItems.slice(0, 10), "No reasoning trace was attached.")}` : ""}
    `;
  }

  function feedbackHtml(view, top) {
    const key = `fb_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    window[key] = {
      caseId: String(view?.case_id || ""),
      topLabel: String(top?.label || top?.dx || ""),
      findingsSummary: String(view?.summary || ""),
    };
    return `
      <div class="decision-feedback-row" data-fbkey="${key}">
        <button type="button" data-outcome="correct" data-fbkey="${key}">
          ${icon("check_circle")} Correct
        </button>
        <button type="button" data-outcome="incorrect" data-fbkey="${key}">
          ${icon("cancel")} Incorrect
        </button>
        <input data-fbkey="${key}" class="feedback-gt-input" type="text" placeholder="Actual diagnosis (optional)" />
        <span class="feedback-status-span" data-fbkey="${key}"></span>
      </div>
    `;
  }

  function buildHtml(view) {
    const top = view?.top_differential || {};
    const safety = view?.safety_state || {};
    const confidence = view?.confidence || {};
    const evidence = view?.evidence_trace || {};
    const timing = view?.timing || {};
    const runtime = view?.runtime || {};
    const family = view?.family || {};
    const mechanism = view?.mechanism || {};
    const candidates = values(view?.disease_candidates);
    const requiredData = values(view?.required_data);
    const reasoningTrace = values(view?.reasoning_trace);
    const topScore = top.score || confidence.diagnosis;
    const evidenceScore = confidence.evidence || evidence.coverage;
    const safetyMode = label(safety.mode || view?.decision || "unresolved");

    const boardMetrics = `
      <div class="decision-metrics">
        <span><strong>${pct(topScore)}</strong>top diagnosis</span>
        <span><strong>${pct(evidenceScore)}</strong>evidence</span>
        <span><strong>${pct(confidence.reliability)}</strong>reliability</span>
        <span><strong>${pct(confidence.closure)}</strong>closure</span>
      </div>
    `;

    // Stash the latest view so the export button can serialize the live data.
    window.__rrrieLastView = view;

    return `
      <article class="decision-board">
        ${abstentionBannerHtml(view)}
        <header class="decision-hero">
          <div class="decision-hero-copy">
            <span class="decision-kicker">Decision Board</span>
            <h2>${esc(top.label || top.dx || "No stable disease anchor")}</h2>
            <p>${esc(view?.summary || "No summary was attached to this diagnostic packet.")}</p>
            ${boardMetrics}
            <div class="decision-anchor compact">
              <strong>${esc(top.label || top.dx || "Unresolved")}</strong>
              <span>${esc(family.label || top.family_label || "family unresolved")}</span>
              <p>${esc(mechanism.primary_mechanism || mechanism.active_state || "Mechanism not attached.")}</p>
            </div>
          </div>
          <div class="decision-hero-side">
            <span class="decision-badge ${tone(topScore)}">${esc(safetyMode)}</span>
            <span class="decision-route">${esc(label(view?.decision || "clinical decision"))}</span>
            ${exportButtonHtml(view)}
          </div>
        </header>

        <div class="decision-grid">
          ${section("Ranked differential", `${candidates.length || 0} candidates`, candidatesHtml(candidates, view), "decision-wide")}

          ${collapsibleSection("Safety posture", safetyMode, safetyHtml(safety), false)}

          ${collapsibleSection("Missing data", `${requiredData.length || 0} requests`, list(requiredData, "No structured follow-up request was generated."), false)}

          ${collapsibleSection("Evidence quality", `${pct(evidenceScore)} coverage`, `
            <div class="decision-quality-strip">
              <span><strong>${pct(evidence.coverage)}</strong>coverage</span>
              <span><strong>${pct(evidence.contradiction_mass)}</strong>contradiction</span>
              <span><strong>${esc(label(evidence.query_hygiene || "not reported"))}</strong>query hygiene</span>
            </div>
            ${evidenceQualityFooter(view)}
            ${evidenceHtml(evidence)}
          `, false, "decision-wide")}

          ${collapsibleSection("Runtime trace", "audit trail", runtimeHtml(runtime, timing, reasoningTrace), false, "decision-wide")}

          ${collapsibleSection("Feedback actions", "learning path", feedbackHtml(view, top), false, "decision-wide decision-feedback-panel")}
        </div>
      </article>
    `;
  }

  window.RRRIEFinalResultView = { buildHtml };
})();

/* Main application script. */
/*
   RRRIE-CDSS Frontend - app.js v4.0
   Full atomic bug-fix pass:
   - Fixed mode button active states
   - Fixed switchTab to update sidenav too
   - Fixed setConnectionState to use new DOM structure
   - Fixed loadCaseToReasoning to actually switch tab and focus
   - Fixed sendMessage to clear placeholder and enable scroll
   - Fixed handleFinalResult to use correct data structure
   - Fixed handleHealth to map VRAM to all status points
   - Fixed footer stat IDs
   - Fixed overview stats (case count sync)

   Backend event types:
     ack, stage_start, stage_complete, stage_result,
     token, thinking_start, thinking_token, thinking_end,
     info, red_flags, zebra_flags,
     api_call, api_result, api_error,
     final_result, error, health, pong
*/

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// State
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
let ws;
let isConnected = false;
let currentMessageDiv = null;
let currentThinkBlock = null;
let currentStage = null;
let mode = "thinking";
let localMode = false;
let isCloudGemini = true;
let pipelineStartTime = 0;
let timerInterval = null;
let currentGroundTruth = null;
let currentTestCaseId = null;
let bootstrapSnapshot = null;
let runtimeEffectiveState = {};
let analysisInFlight = false;
let healthProbeStartedAt = 0;
let fetchedCases = {};
let progressUiState = {
  state: "idle",
  stage: null,
  pct: null,
  evidenceCount: 0,
  activity: [],
};
let completedSwarmStages = new Set();
let smartDrawerAutoOpened = false;

const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const WS_URL = `${WS_PROTOCOL}://${window.location.host}/ws/chat`;
const TYPED_DELTA_MAGIC = "RRD1";
const HTTP_TYPED_MEDIA_TYPE = "application/vnd.rrrie.typed-delta";
const typedTextEncoder = new TextEncoder();
const typedTextDecoder = new TextDecoder();

function concatTypedChunks(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new Uint8Array(total);
  let offset = 0;
  chunks.forEach((chunk) => {
    buffer.set(chunk, offset);
    offset += chunk.length;
  });
  return buffer;
}

function encodeTypedNumber(value) {
  const buffer = new Uint8Array(9);
  const view = new DataView(buffer.buffer);
  buffer[0] = "D".charCodeAt(0);
  view.setFloat64(1, Number(value), false);
  return buffer;
}

function encodeTypedString(value) {
  const raw = typedTextEncoder.encode(String(value));
  const header = new Uint8Array(5);
  const view = new DataView(header.buffer);
  header[0] = "S".charCodeAt(0);
  view.setUint32(1, raw.length, false);
  return concatTypedChunks([header, raw]);
}

function encodeTypedValue(value) {
  if (value === null || value === undefined) return Uint8Array.of("N".charCodeAt(0));
  if (typeof value === "boolean") return Uint8Array.of(value ? "T".charCodeAt(0) : "F".charCodeAt(0));
  if (typeof value === "number") return encodeTypedNumber(value);
  if (typeof value === "string") return encodeTypedString(value);
  if (Array.isArray(value)) {
    const header = new Uint8Array(5);
    const view = new DataView(header.buffer);
    header[0] = "A".charCodeAt(0);
    view.setUint32(1, value.length, false);
    return concatTypedChunks([header, ...value.map((item) => encodeTypedValue(item))]);
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    const header = new Uint8Array(5);
    const view = new DataView(header.buffer);
    header[0] = "O".charCodeAt(0);
    view.setUint32(1, entries.length, false);
    const chunks = [header];
    entries.forEach(([key, item]) => {
      chunks.push(encodeTypedString(key));
      chunks.push(encodeTypedValue(item));
    });
    return concatTypedChunks(chunks);
  }
  return encodeTypedString(String(value));
}

function packTypedDelta(payload) {
  const magic = typedTextEncoder.encode(TYPED_DELTA_MAGIC);
  return concatTypedChunks([magic, encodeTypedValue(payload || {})]).buffer;
}

function typedReadExact(buffer, cursor, size) {
  const end = cursor + size;
  if (end > buffer.length) throw new Error("truncated typed delta payload");
  return [buffer.slice(cursor, end), end];
}

function decodeTypedValue(buffer, cursor = 0) {
  const [tagRaw, afterTag] = typedReadExact(buffer, cursor, 1);
  const tag = String.fromCharCode(tagRaw[0]);
  if (tag === "N") return [null, afterTag];
  if (tag === "T") return [true, afterTag];
  if (tag === "F") return [false, afterTag];
  if (tag === "D") {
    const [raw, next] = typedReadExact(buffer, afterTag, 8);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const value = view.getFloat64(0, false);
    return [Number.isInteger(value) ? Math.trunc(value) : value, next];
  }
  if (tag === "S") {
    const [lenRaw, next] = typedReadExact(buffer, afterTag, 4);
    const lenView = new DataView(lenRaw.buffer, lenRaw.byteOffset, lenRaw.byteLength);
    const size = lenView.getUint32(0, false);
    const [raw, end] = typedReadExact(buffer, next, size);
    return [typedTextDecoder.decode(raw), end];
  }
  if (tag === "A") {
    const [lenRaw, next] = typedReadExact(buffer, afterTag, 4);
    const lenView = new DataView(lenRaw.buffer, lenRaw.byteOffset, lenRaw.byteLength);
    const count = lenView.getUint32(0, false);
    let index = next;
    const values = [];
    for (let i = 0; i < count; i += 1) {
      const [item, itemCursor] = decodeTypedValue(buffer, index);
      values.push(item);
      index = itemCursor;
    }
    return [values, index];
  }
  if (tag === "O") {
    const [lenRaw, next] = typedReadExact(buffer, afterTag, 4);
    const lenView = new DataView(lenRaw.buffer, lenRaw.byteOffset, lenRaw.byteLength);
    const count = lenView.getUint32(0, false);
    let index = next;
    const values = {};
    for (let i = 0; i < count; i += 1) {
      const [key, keyCursor] = decodeTypedValue(buffer, index);
      const [item, itemCursor] = decodeTypedValue(buffer, keyCursor);
      values[String(key)] = item;
      index = itemCursor;
    }
    return [values, index];
  }
  throw new Error(`unknown typed delta tag: ${tag}`);
}

function unpackTypedDelta(payload) {
  const buffer = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const magic = typedTextDecoder.decode(buffer.slice(0, 4));
  if (magic !== TYPED_DELTA_MAGIC) throw new Error("invalid typed delta header");
  const [value, cursor] = decodeTypedValue(buffer, 4);
  if (cursor !== buffer.length) throw new Error("unexpected trailing bytes in typed delta payload");
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("typed delta root payload must be an object");
  return value;
}

function sendTypedMessage(payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(packTypedDelta(payload));
  return true;
}

async function readTypedHttpResponse(response) {
  const contentType = String(response.headers.get("content-type") || "").toLowerCase();
  if (contentType.includes(HTTP_TYPED_MEDIA_TYPE)) {
    return unpackTypedDelta(await response.arrayBuffer());
  }
  return response.json();
}

async function fetchTypedPayload(url, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", HTTP_TYPED_MEDIA_TYPE);
  let body = options.body;
  if (body && typeof body === "object" && !(body instanceof ArrayBuffer) && !(body instanceof Blob) && !(body instanceof Uint8Array) && !(body instanceof FormData)) {
    headers.set("Content-Type", HTTP_TYPED_MEDIA_TYPE);
    body = packTypedDelta(body);
  }
  const response = await fetch(url, { ...options, headers, body });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return readTypedHttpResponse(response);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Tested Cases (localStorage)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function getTestedCases() {
  try { return JSON.parse(localStorage.getItem("rrrie_tested_cases") || "[]"); }
  catch { return []; }
}
function markCaseAsTested(id) {
  const t = getTestedCases();
  if (!t.includes(id)) { t.push(id); localStorage.setItem("rrrie_tested_cases", JSON.stringify(t)); }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// DOM Cache
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
let dom = {};

function cacheDom() {
  dom = {
    // Status
    statusDotInner: document.getElementById("statusDotInner"),
    statusText: document.getElementById("statusText"),
    // Reasoning
    messages: document.getElementById("messages"),
    messagesPlaceholder: document.getElementById("messages-placeholder"),
    chatScrollArea: document.getElementById("chat-scroll-area"),
    input: document.getElementById("userInput"),
    sendBtn: document.getElementById("sendButton"),
    stopBtn: document.getElementById("stopButton"),
    mobileStopBtn: document.getElementById("mobileStopButton"),
    trackerStats: document.getElementById("trackerStats"),
    analysisStateBadge: document.getElementById("analysisStateBadge"),
    analysisWorkspace: document.getElementById("analysisWorkspace"),
    analysisStageLabel: document.getElementById("analysisStageLabel"),
    analysisStageCaption: document.getElementById("analysisStageCaption"),
    analysisElapsed: document.getElementById("analysisElapsed"),
    analysisProgressValue: document.getElementById("analysisProgressValue"),
    analysisProgressFill: document.getElementById("analysisProgressFill"),
    analysisProgressRing: document.getElementById("analysisProgressRing"),
    analysisActivityStream: document.getElementById("analysisActivityStream"),
    resultPreviewSlot: document.getElementById("resultPreviewSlot"),
    swarmMap: document.getElementById("agentSwarmMap"),
    swarmNodes: document.querySelectorAll(".swarm-node"),
    swarmEdges: document.querySelectorAll(".swarm-edge"),
    swarmActiveLabel: document.getElementById("swarmActiveLabel"),
    swarmFocusLabel: document.getElementById("swarmFocusLabel"),
    swarmFocusNote: document.getElementById("swarmFocusNote"),
    swarmPhaseMetric: document.getElementById("swarmPhaseMetric"),
    swarmEvidenceMetric: document.getElementById("swarmEvidenceMetric"),
    swarmProgressMetric: document.getElementById("swarmProgressMetric"),
    swarmProgressValue: document.getElementById("swarmProgressValue"),
    evCoverageCount: document.getElementById("evCoverageCount"),
    evContradiction: document.getElementById("evContradiction"),
    evQueryHygiene: document.getElementById("evQueryHygiene"),
    memT1: document.getElementById("memT1"),
    operationModeSelect: document.getElementById("operationModeSelect"),
    runtimeProfileSelect: document.getElementById("runtimeProfileSelect"),
    detailsDrawer: document.getElementById("evidenceStatePanel"),
    detailsDrawerBackdrop: document.getElementById("detailsDrawerBackdrop"),
    detailsDrawerToggle: document.getElementById("detailsDrawerToggle"),
    detailsDrawerClose: document.getElementById("detailsDrawerClose"),
    // Registry
    registryList: document.getElementById("registry-list"),
    regCaseCount: document.getElementById("reg-case-count"),
    caseSourceSummary: document.getElementById("case-source-summary"),
    casePolicyLabel: document.getElementById("case-policy-label"),
    regDetailEmpty: document.getElementById("reg-detail-empty"),
    regDetailContent: document.getElementById("reg-detail-content"),
    detName: document.getElementById("det-name"),
    detId: document.getElementById("det-id"),
    detSource: document.getElementById("det-source"),
    detPolicy: document.getElementById("det-policy"),
    detHypothesis: document.getElementById("det-hypothesis"),
    detNarrative: document.getElementById("det-narrative"),
    detConfScore: document.getElementById("det-conf-score"),
    caseLabDrawer: document.getElementById("caseLabDrawer"),
    caseLabBackdrop: document.getElementById("caseLabBackdrop"),
    caseLabContent: document.getElementById("caseLabContent"),
    // Overview
    overviewOperationMode: document.getElementById("overview-operation-mode"),
    overviewRuntimeProfile: document.getElementById("overview-runtime-profile"),
    overviewEngineMode: document.getElementById("overview-engine-mode"),
    overviewLlmStatus: document.getElementById("overview-llm-status"),
    overviewLlmCaption: document.getElementById("overview-llm-caption"),
    overviewCaseCount: document.getElementById("overview-case-count"),
    overviewLatency: document.getElementById("overview-latency"),
    // Footer
    footerLlmStatus: document.getElementById("footer-llm-status"),
    footerLatency: document.getElementById("footer-latency"),
    footerVram: document.getElementById("footer-vram"),
    // Navigation
    viewSections: document.querySelectorAll(".view-section"),
    navLinks: document.querySelectorAll(".nav-link"),
    sidenavItems: document.querySelectorAll(".sidenav-item"),
    mobileNavItems: document.querySelectorAll(".mobile-nav-item"),
  };
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Init
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function setComposerBusy(isBusy) {
  const busy = Boolean(isBusy);
  const shell = document.getElementById("composerShell");
  shell?.classList.toggle("is-running", busy);
  if (dom.sendBtn) dom.sendBtn.disabled = busy;
  [dom.stopBtn, dom.mobileStopBtn].forEach((button) => {
    if (!button) return;
    button.classList.toggle("hidden", !busy);
    button.disabled = !busy;
  });
}

function restoreComposerDock() {
  const shell = document.getElementById("composerShell");
  if (!shell) return;
  shell.classList.remove("is-inline", "is-result-inline");
  document.body.classList.remove("result-mode");
  dom.messages?.classList.remove("has-final-result");
  dom.chatScrollArea?.classList.remove("has-final-result");
  if (shell.parentElement !== document.body) {
    const footer = document.querySelector(".footer");
    document.body.insertBefore(shell, footer || document.getElementById("detailsDrawerBackdrop") || null);
  }
}

function dockComposerAfterResult(resultNode) {
  const shell = document.getElementById("composerShell");
  if (!shell || !resultNode) return;
  shell.classList.remove("is-running");
  shell.classList.add("is-inline", "is-result-inline");
  document.body.classList.add("result-mode");
  dom.messages?.classList.add("has-final-result");
  dom.chatScrollArea?.classList.add("has-final-result");
  resultNode.insertAdjacentElement("afterend", shell);
}

function setPanelInert(panel, inert) {
  if (!panel) return;
  if (inert) panel.setAttribute("inert", "");
  else panel.removeAttribute("inert");
}

function panelFocusable(panel) {
  if (!panel) return [];
  return Array.from(panel.querySelectorAll([
    "a[href]",
    "button:not([disabled])",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    "[tabindex]:not([tabindex='-1'])",
  ].join(","))).filter((element) => element.offsetParent !== null || element === document.activeElement);
}

function activeDrawerPanel() {
  if (document.body.classList.contains("details-drawer-docked")) return null;
  if (document.body.classList.contains("details-drawer-open")) return dom.detailsDrawer;
  if (document.body.classList.contains("case-lab-open")) return dom.caseLabDrawer;
  return null;
}

function trapDrawerFocus(event) {
  const panel = activeDrawerPanel();
  if (!panel) return;
  const focusable = panelFocusable(panel);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function handleDelegatedAction(event) {
  const actionTarget = event.target.closest("[data-action]");
  if (actionTarget) {
    const action = actionTarget.dataset.action;
    event.preventDefault();
    switch (action) {
      case "switch-tab":
        switchTab(actionTarget.dataset.tabTarget || "reasoning");
        if (actionTarget.dataset.refreshLearning === "true") loadLearningStats();
        return;
      case "set-mode":
        setMode(actionTarget.dataset.mode || "thinking");
        return;
      case "toggle-details":
        toggleDetailsDrawer();
        return;
      case "close-details":
        closeDetailsDrawer();
        return;
      case "open-case-lab":
        openCaseLab();
        return;
      case "close-case-lab":
        closeCaseLab();
        return;
      case "close-reg-detail":
        closeRegDetail();
        return;
      case "load-case-to-reasoning":
        loadCaseToReasoning();
        return;
      case "set-example":
        setExample(actionTarget.dataset.example || "");
        return;
      case "switch-panel-tab":
        rrSwitchPanelTab(actionTarget.dataset.panelTab || actionTarget.dataset.tab || "evidence");
        return;
      case "load-learning-stats":
        loadLearningStats();
        return;
      case "stop-analysis":
        stopAnalysis();
        return;
      case "send-message":
        sendMessage();
        return;
      case "rrrie-export":
        exportDecisionPacket();
        return;
      case "rrrie-evidence-close":
        closeEvidenceDrawer();
        return;
      default:
        break;
    }
  }

  const evidenceTarget = event.target.closest("[data-evidence-id]");
  if (evidenceTarget) {
    event.preventDefault();
    openEvidenceDrawer(evidenceTarget.dataset.evidenceId);
    return;
  }

  const caseAction = event.target.closest("[data-case-action]");
  if (caseAction) {
    const caseId = caseAction.dataset.caseId || "";
    const action = caseAction.dataset.caseAction;
    if (action === "promote" || action === "reject") {
      event.preventDefault();
      updateGeneratedCaseStatus(caseId, action);
    }
    return;
  }

  const feedbackButton = event.target.closest("[data-outcome][data-fbkey]");
  if (feedbackButton && feedbackButton.closest(".decision-feedback-row")) {
    event.preventDefault();
    window._submitFeedbackFromBtn?.(feedbackButton);
  }
}

// --- Evidence drill-down drawer ----------------------------------------
function openEvidenceDrawer(evidenceId) {
  const item = (window.__rrrieEvidence || {})[evidenceId];
  if (!item) return;
  let drawer = document.getElementById("rrrie-evidence-drawer");
  if (!drawer) {
    drawer = document.createElement("aside");
    drawer.id = "rrrie-evidence-drawer";
    drawer.className = "rrrie-evidence-drawer";
    drawer.setAttribute("role", "dialog");
    drawer.setAttribute("aria-modal", "true");
    drawer.setAttribute("aria-label", "Evidence detail");
    document.body.appendChild(drawer);
    drawer.addEventListener("click", (event) => {
      if (event.target === drawer) closeEvidenceDrawer();
    });
  }
  const url = item.citation || item.source || "";
  const isUrl = /^https?:\/\//i.test(String(url));
  const trust = Number.isFinite(Number(item.trust_score)) ? `${Math.round(Number(item.trust_score) * 100)}%` : "n/a";
  const linked = Array.isArray(item.linked_hypotheses) && item.linked_hypotheses.length
    ? item.linked_hypotheses.join(", ")
    : "(no linked hypotheses)";
  const escapeHtml = window.escapeHTML || ((s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])));
  drawer.innerHTML = `
    <div class="rrrie-evidence-drawer-card" role="document">
      <div class="rrrie-evidence-drawer-head">
        <h3>${escapeHtml(item.title || item.citation || "Evidence item")}</h3>
        <button type="button" class="rrrie-evidence-close" data-action="rrrie-evidence-close" aria-label="Close evidence detail">&times;</button>
      </div>
      <dl class="rrrie-evidence-drawer-meta">
        <div><dt>Source</dt><dd>${escapeHtml(item.source || "(unknown)")}</dd></div>
        <div><dt>Trust score</dt><dd>${escapeHtml(trust)}</dd></div>
        <div><dt>Relation</dt><dd>${escapeHtml(item.relation_type || "support")}</dd></div>
        <div><dt>Verification</dt><dd>${escapeHtml(item.verification_status || "unverified")}</dd></div>
        <div><dt>Linked candidates</dt><dd>${escapeHtml(linked)}</dd></div>
        ${isUrl ? `<div><dt>Citation</dt><dd><a href="${escapeHtml(url)}" target="_blank" rel="noreferrer noopener">${escapeHtml(url)}</a></dd></div>` : ""}
      </dl>
      <h4>Excerpt</h4>
      <p class="rrrie-evidence-drawer-excerpt">${escapeHtml(item.excerpt || "(no excerpt attached)")}</p>
    </div>
  `;
  drawer.classList.add("open");
  document.documentElement.classList.add("rrrie-evidence-drawer-open");
}

function closeEvidenceDrawer() {
  const drawer = document.getElementById("rrrie-evidence-drawer");
  if (drawer) drawer.classList.remove("open");
  document.documentElement.classList.remove("rrrie-evidence-drawer-open");
}

// --- Export packet ----------------------------------------------------
function exportDecisionPacket() {
  const view = window.__rrrieLastView;
  if (!view) {
    console.warn("No decision packet to export yet");
    return;
  }
  const blob = new Blob([JSON.stringify(view, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const caseId = String(view.case_id || "case").replace(/[^a-z0-9_-]+/gi, "_");
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = url;
  link.download = `decision_packet_${caseId}_${ts}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  // Open browser print view as a printable artifact (uses print stylesheet).
  setTimeout(() => window.print(), 80);
}

async function init() {
  cacheDom();
  setPanelInert(dom.detailsDrawer, true);
  setPanelInert(dom.caseLabDrawer, true);
  setComposerBusy(false);
  updateAnalysisState("idle");
  await loadBootstrapState();
  syncOperationModeUi();
  connectWebSocket();

  if (dom.input) {
    dom.input.addEventListener("keydown", handleKeydown);
    dom.input.addEventListener("input", autoResize);
  }
  if (dom.operationModeSelect) dom.operationModeSelect.addEventListener("change", handleOperationModeChange);
  if (dom.runtimeProfileSelect) dom.runtimeProfileSelect.addEventListener("change", handleRuntimeProfileChange);
  document.getElementById("llmModeToggle")?.addEventListener("change", toggleLlmMode);
  if (dom.detailsDrawerBackdrop) dom.detailsDrawerBackdrop.addEventListener("click", closeDetailsDrawer);
  if (dom.caseLabBackdrop) dom.caseLabBackdrop.addEventListener("click", closeCaseLab);
  document.addEventListener("click", handleDelegatedAction);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Tab") trapDrawerFocus(event);
    if (event.key === "Escape") {
      closeDetailsDrawer();
      closeCaseLab();
      closeEvidenceDrawer();
    }
  });

  // Default to the clinical reasoning workspace.
  switchTab('reasoning');
  if (window.matchMedia?.("(min-width: 1560px)").matches) {
    document.body.classList.add("details-drawer-open", "details-drawer-docked");
    setPanelInert(dom.detailsDrawer, false);
    dom.detailsDrawerToggle?.setAttribute("aria-expanded", "true");
  }
  await fetchTestCases();
  requestAnimationFrame(() => {
    if (dom.chatScrollArea && dom.messagesPlaceholder) dom.chatScrollArea.scrollTop = 0;
  });
}

function setDetailsDrawer(open, options = {}) {
  const next = Boolean(open);
  const focus = options.focus !== false;
  document.body.classList.toggle("details-drawer-open", next);
  document.body.classList.remove("details-drawer-docked");
  setPanelInert(dom.detailsDrawer, !next);
  if (dom.detailsDrawerToggle) dom.detailsDrawerToggle.setAttribute("aria-expanded", next ? "true" : "false");
  if (next && focus) requestAnimationFrame(() => dom.detailsDrawerClose?.focus());
}

function openDetailsDrawer() {
  setDetailsDrawer(true);
}

function closeDetailsDrawer() {
  setDetailsDrawer(false);
}

function toggleDetailsDrawer() {
  setDetailsDrawer(!document.body.classList.contains("details-drawer-open"));
}

window.openDetailsDrawer = openDetailsDrawer;
window.closeDetailsDrawer = closeDetailsDrawer;
window.toggleDetailsDrawer = toggleDetailsDrawer;

function setCaseLab(open) {
  const next = Boolean(open);
  document.body.classList.toggle("case-lab-open", next);
  setPanelInert(dom.caseLabDrawer, !next);
  if (next) loadGeneratedCaseLab();
}

function openCaseLab() { setCaseLab(true); }
function closeCaseLab() { setCaseLab(false); }

window.openCaseLab = openCaseLab;
window.closeCaseLab = closeCaseLab;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Bootstrap
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
async function loadBootstrapState() {
  try {
    bootstrapSnapshot = await fetchTypedPayload(`/bootstrap?ts=${Date.now()}`, { cache: "no-store" });
    applyBootstrapSnapshot(bootstrapSnapshot);
  } catch (e) {
    console.warn("[BOOTSTRAP] fallback:", e);
    applyBootstrapSnapshot({
      requested: { operation_mode: "local_core_with_online_r2", runtime_profile: "auto", local_mode: true },
      effective: { operation_mode: "local_core_with_online_r2", runtime_profile: "auto", local_mode: true },
    });
  }
}

function applyBootstrapSnapshot(snapshot) {
  const req = snapshot?.requested || {};
  const eff = snapshot?.effective || {};
  runtimeEffectiveState = { ...eff };

  const opMode = req.operation_mode || eff.operation_mode || "local_core_with_online_r2";
  const rtProfile = req.runtime_profile || eff.runtime_profile || "auto";

  localMode = opMode !== "cloud_reference";
  isCloudGemini = !localMode;
  setLocalToggleVisualState();

  if (dom.operationModeSelect) dom.operationModeSelect.value = opMode;
  if (dom.runtimeProfileSelect) dom.runtimeProfileSelect.value = rtProfile;

  // Restore persisted reasoning_mode (fast / thinking / deep) from preferences.
  const savedMode = snapshot?.preferences?.reasoning_mode;
  if (savedMode && ["fast", "thinking", "deep"].includes(savedMode)) {
    setMode(savedMode);
  }

  updateRuntimeUi(runtimeEffectiveState, null);
}

function currentRequestedOperationMode() {
  return dom.operationModeSelect?.value || (localMode ? "local_core_with_online_r2" : "cloud_reference");
}
function currentRequestedRuntimeProfile() {
  return dom.runtimeProfileSelect?.value || "auto";
}

function updateAnalysisState(state) {
  const normalized = String(state || "idle").toLowerCase();
  const states = {
    idle: { label: "Idle" },
    running: { label: "Running" },
    cancelling: { label: "Cancelling" },
    completed: { label: "Completed" },
    cancelled: { label: "Cancelled" },
    error: { label: "Interrupted" },
    offline: { label: "Offline" },
  };

  const next = states[normalized] || states.idle;
  if (dom.analysisStateBadge) {
    dom.analysisStateBadge.textContent = next.label;
    dom.analysisStateBadge.className = `state-pill analysis-state-pill is-${normalized}`;
  }
  renderProgressState({ state: normalized });
}

function stageLabelFor(stage) {
  const normalized = normalizeStageName(stage);
  if (!normalized || normalized === "idle") return "Ready";
  return STAGE_TITLES?.[normalized]?.label || String(normalized).replace(/_/g, " ");
}

function renderProgressState(patch = {}) {
  progressUiState = { ...progressUiState, ...patch };
  const state = progressUiState.state || "idle";
  const stage = progressUiState.stage || "idle";
  const pct = Number(progressUiState.pct);
  const hasPct = progressUiState.pct !== null && progressUiState.pct !== undefined && Number.isFinite(pct);
  const clampedPct = hasPct ? Math.max(0, Math.min(100, pct)) : null;

  if (dom.analysisWorkspace) {
    dom.analysisWorkspace.dataset.state = state;
    dom.analysisWorkspace.dataset.stage = stage;
    dom.analysisWorkspace.dataset.progress = hasPct ? "determinate" : (state === "running" ? "indeterminate" : "idle");
    dom.analysisWorkspace.style.setProperty("--progress-angle", `${hasPct ? clampedPct * 3.6 : 0}deg`);
  }
  setSwarmRunState(state);
  if (dom.analysisStageLabel) dom.analysisStageLabel.textContent = stageLabelFor(stage);
  if (dom.analysisStageCaption && patch.caption) dom.analysisStageCaption.textContent = String(patch.caption);
  if (dom.analysisProgressValue) dom.analysisProgressValue.textContent = hasPct ? `${Math.round(clampedPct)}%` : "--";
  if (dom.analysisProgressRing) {
    if (hasPct) {
      dom.analysisProgressRing.setAttribute("aria-valuenow", String(Math.round(clampedPct)));
      dom.analysisProgressRing.setAttribute("aria-valuetext", `${Math.round(clampedPct)} percent complete`);
    } else {
      dom.analysisProgressRing.removeAttribute("aria-valuenow");
      dom.analysisProgressRing.setAttribute(
        "aria-valuetext",
        state === "running" ? "Analysis running with measured stage progress pending" : "No analysis running"
      );
    }
  }
  if (dom.swarmMap && hasPct) dom.swarmMap.style.setProperty("--swarm-progress", `${clampedPct}%`);
  if (dom.swarmProgressMetric) dom.swarmProgressMetric.textContent = hasPct ? `${Math.round(clampedPct)}%` : state;
  if (dom.swarmEvidenceMetric) dom.swarmEvidenceMetric.textContent = String(progressUiState.evidenceCount || 0);
  if (dom.analysisProgressFill) dom.analysisProgressFill.style.width = hasPct ? `${clampedPct}%` : "";
  if (dom.messages) dom.messages.setAttribute("aria-busy", state === "running" ? "true" : "false");
}

function setActivityStream(items) {
  if (!dom.analysisActivityStream) return;
  const safeItems = items.slice(-2);
  dom.analysisActivityStream.innerHTML = safeItems.map((item, index) => {
    const icon = item.tone === "success" ? "check_circle" : item.tone === "error" ? "error" : item.tone === "warn" ? "warning" : "radio_button_checked";
    const role = index === safeItems.length - 1 ? "is-latest" : "is-secondary";
    return `
      <div class="act analysis-activity-item ${escapeHTML(item.tone || "muted")} ${role}">
        <span class="material-symbols-outlined">${icon}</span>
        <span>${escapeHTML(item.text)}</span>
        <time>${role === "is-latest" ? "latest" : "prior"}</time>
      </div>
    `;
  }).join("");
}

function pushActivity(text, tone = "info") {
  const item = { text: String(text || "").trim(), tone };
  if (!item.text) return;
  progressUiState.activity = [...(progressUiState.activity || []), item].slice(-8);
  setActivityStream(progressUiState.activity);
}

function resetActivityStream() {
  progressUiState.activity = [];
  setActivityStream([{ text: "Waiting for a patient narrative.", tone: "muted" }]);
}

function formatModeLabel(value) {
  const normalized = String(value || "").trim();
  if (!normalized) return "--";
  return normalized.replace(/_/g, " ");
}

function updateRuntimeUi(runtime, llmServer) {
  const state = runtime || {};
  const operationMode = state.operation_mode || currentRequestedOperationMode();
  const runtimeProfile = state.runtime_profile || currentRequestedRuntimeProfile();
  const engineMode = state.engine_mode || (isCloudGemini ? "cloud_gemini" : "local_qwen");
  const llmStatus = llmServer?.status || "";
  const llmReady = llmStatus === "online" || llmStatus === "ok";
  const rttMs = Number(state.websocket_rtt_ms || 0);

  if (dom.overviewOperationMode) dom.overviewOperationMode.textContent = formatModeLabel(operationMode);
  if (dom.overviewRuntimeProfile) dom.overviewRuntimeProfile.textContent = formatModeLabel(runtimeProfile);
  if (dom.overviewEngineMode) dom.overviewEngineMode.textContent = formatModeLabel(engineMode);

  if (dom.overviewLlmStatus) {
    dom.overviewLlmStatus.textContent = llmReady ? "Online" : "Limited";
    dom.overviewLlmStatus.className = llmReady ? "delta" : "delta warn";
  }
  if (dom.overviewLlmCaption) {
    dom.overviewLlmCaption.textContent = llmReady
      ? "Backend and inference transport responding normally"
      : "Transport is reachable but inference runtime is degraded or unavailable";
  }

  if (dom.footerLlmStatus) {
    dom.footerLlmStatus.textContent = llmReady ? "Inference: Ready" : "Inference: Degraded";
    dom.footerLlmStatus.className = llmReady ? "ok" : "warn";
  }

  if (rttMs > 0) {
    const rttLabel = `Transport RTT: ${Math.round(rttMs)}ms`;
    if (dom.footerLatency) dom.footerLatency.textContent = rttLabel;
    if (dom.overviewLatency) dom.overviewLatency.textContent = rttLabel;
  } else {
    if (dom.footerLatency) dom.footerLatency.textContent = "Transport RTT: --";
    if (dom.overviewLatency) dom.overviewLatency.textContent = "Transport RTT: --";
  }
}

function requestHealthSnapshot() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  healthProbeStartedAt = performance.now();
  sendTypedMessage({ type: "health" });
}

function persistRuntimePreferences() {
  return fetchTypedPayload("/api/preferences", {
    method: "POST",
    body: {
      operation_mode_default: currentRequestedOperationMode(),
      runtime_profile_default: currentRequestedRuntimeProfile(),
      reasoning_mode: mode,
    },
  }).catch(e => console.error("Pref save:", e));
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// WebSocket
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function connectWebSocket() {
  try { ws = new WebSocket(WS_URL); } catch { setConnectionState("error", "WS Error"); return; }
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    isConnected = true;
    setConnectionState("ready", "Connected");
    requestHealthSnapshot();
  };
  ws.onclose = () => {
    isConnected = false;
    analysisInFlight = false;
    setConnectionState("error", "Disconnected");
    setComposerBusy(false);
    setTimeout(connectWebSocket, 3000);
  };
  ws.onerror = () => {
    setConnectionState("error", "WS Error");
  };
  ws.onmessage = (ev) => {
    try {
      if (ev.data instanceof ArrayBuffer) {
        handleServerMessage(unpackTypedDelta(ev.data));
        return;
      }
      if (ev.data instanceof Blob) {
        ev.data.arrayBuffer()
          .then((buffer) => handleServerMessage(unpackTypedDelta(buffer)))
          .catch((error) => console.error("[WS parse]", error));
        return;
      }
      throw new Error("unexpected non-binary websocket frame");
    } catch (e) { console.error("[WS parse]", e); }
  };
}

function setConnectionState(cls, text) {
  if (dom.statusDotInner) {
    dom.statusDotInner.className = cls === "ready"
      ? "w-2 h-2 rounded-full bg-vital animate-pulse"
      : "w-2 h-2 rounded-full bg-critical animate-pulse";
  }
  if (dom.statusText) dom.statusText.textContent = text;
  if (dom.sendBtn) dom.sendBtn.disabled = !isConnected || analysisInFlight;
  if (cls !== "ready" && dom.messages) dom.messages.setAttribute("aria-busy", "false");
  updateAnalysisState(cls === "ready" ? (analysisInFlight ? "running" : "idle") : "offline");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Mode & Toggle
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function toggleLlmMode() {
  const toggle = document.getElementById("llmModeToggle");
  isCloudGemini = toggle.checked;
  localMode = !isCloudGemini;
  const label = document.getElementById("llmModeLabel");
  if (label) label.textContent = isCloudGemini ? "Cloud (Gemini)" : "Local";
  setLocalToggleVisualState();
  syncOperationModeUi();
  persistRuntimePreferences();
}

function setLocalToggleVisualState() {
  const toggle = document.getElementById("llmModeToggle");
  if (toggle) toggle.checked = isCloudGemini;
  const label = document.getElementById("llmModeLabel");
  if (label) label.textContent = isCloudGemini ? "Cloud (Gemini)" : "Local";
}

function setMode(newMode) {
  mode = newMode;
  // Update all mode buttons
  ["fast", "thinking", "deep"].forEach(m => {
    const btn = document.getElementById(`btn${m.charAt(0).toUpperCase() + m.slice(1)}`);
    if (!btn) return;
    btn.classList.toggle("active", m === newMode);
  });

  // Do not force cloud just because ATOM (deep) mode is clicked.
  // The local/cloud toggle is separate.
  // We only adjust runtime profile automatically if needed.
  if (dom.operationModeSelect && dom.runtimeProfileSelect) {
    if (newMode === "fast") {
      dom.runtimeProfileSelect.value = "compact_4gb";
    } else if (newMode === "deep") {
      dom.runtimeProfileSelect.value = "auto";
    } else {
      dom.runtimeProfileSelect.value = "auto";
    }
  }
  
  setLocalToggleVisualState();
  syncOperationModeUi();
  persistRuntimePreferences();

  // Update composer mode label
  const modeLabels = { fast: "Fast", thinking: "Thinking", deep: "Deep" };
  const composerLabel = document.getElementById("composerModeLabel");
  if (composerLabel) composerLabel.textContent = modeLabels[newMode] || "";
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Tab Navigation
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function switchTab(tabId) {
  document.body.dataset.view = tabId;
  if (tabId !== "reasoning") closeDetailsDrawer();
  const subtitles = {
    reasoning: "Clinical agent workspace",
    registry: "Case memory & evidence",
    schedule: "Continual learning loop",
    overview: "Runtime, gates & telemetry",
  };
  const brandSub = document.getElementById("brandSubTitle");
  if (brandSub && subtitles[tabId]) brandSub.textContent = subtitles[tabId];
  const composerShell = document.getElementById("composerShell");
  if (composerShell) composerShell.style.display = tabId === "reasoning" ? "" : "none";

  // Show/hide sections
  if (dom.viewSections) {
    dom.viewSections.forEach(s => {
      s.classList.toggle("active", s.id === `view-${tabId}`);
    });
  }
  // Update top nav
  if (dom.navLinks) {
    dom.navLinks.forEach(l => {
      l.classList.toggle("active", l.id === `nav-${tabId}`);
    });
  }
  // Update side nav
  if (dom.sidenavItems) {
    dom.sidenavItems.forEach(item => {
      const isActive = item.id === `sidenav-${tabId}`;
      item.classList.toggle("active", isActive);
      item.setAttribute("aria-selected", isActive ? "true" : "false");
    });
  }
  if (dom.mobileNavItems) {
    dom.mobileNavItems.forEach(item => {
      item.classList.toggle("is-active", item.id === `mobile-nav-${tabId}`);
    });
  }
  requestAnimationFrame(() => dom.chatScrollArea?.scrollTo?.({ top: 0 }));
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Operation Mode
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function handleOperationModeChange() {
  if (!dom.operationModeSelect) return;
  localMode = dom.operationModeSelect.value !== "cloud_reference";
  isCloudGemini = !localMode;
  setLocalToggleVisualState();
  syncOperationModeUi();
  persistRuntimePreferences();
}
function handleRuntimeProfileChange() { persistRuntimePreferences(); }

function syncOperationModeUi() {
  if (!dom.operationModeSelect) return;
  const disabled = mode === "deep";
  dom.operationModeSelect.disabled = disabled;
  if (disabled) {
    dom.operationModeSelect.value = "cloud_reference";
  } else if (!localMode) {
    dom.operationModeSelect.value = "cloud_reference";
  } else if (dom.operationModeSelect.value === "cloud_reference") {
    dom.operationModeSelect.value = "local_core_with_online_r2";
  }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Clinical Registry
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function sourceLabel(sourceType, sourceLabelRaw) {
  const raw = String(sourceLabelRaw || sourceType || "clinical").trim();
  if (!raw) return "CLINICAL";
  return raw.replace(/_/g, " ").toUpperCase();
}

function sourceTone(sourceType) {
  const type = String(sourceType || "").toLowerCase();
  if (type.includes("who")) return "who";
  if (type.includes("pubmed")) return "pubmed";
  if (type.includes("generated")) return "generated";
  return "clinical";
}

function renderSourceSummary(meta = {}) {
  if (!dom.caseSourceSummary) return;
  const bySource = meta.by_source || {};
  const entries = [
    ["clinical", "Clinical", bySource.clinical || 0],
    ["who", "WHO", bySource.who || 0],
    ["pubmed", "PubMed", bySource.pubmed || 0],
  ].filter(([, , count]) => Number(count) > 0);
  const policy = String(meta.policy || "curated_real_cases_only");
  const chips = entries.length
    ? entries.map(([type, labelText, count]) => `<span class="source-chip is-${type}">${labelText}: ${count}</span>`).join("")
    : `<span class="source-chip">Curated sources: 0</span>`;
  dom.caseSourceSummary.innerHTML = `
    <span class="source-chip is-policy">${policy.replace(/_/g, " ")}</span>
    ${chips}
  `;
  if (dom.casePolicyLabel) {
    dom.casePolicyLabel.textContent = policy === "curated_real_cases_only"
      ? "Only curated real cases are shown here."
      : "Generated drafts are excluded from this library.";
  }
}

async function fetchTestCases() {
  if (!dom.registryList) return;
  fetchedCases = {};

  try {
    const data = await fetchTypedPayload(`/api/test-cases?ts=${Date.now()}`, { cache: "no-store" });
    const cases = Array.isArray(data.cases) ? data.cases : [];
    const meta = data.meta || {};

    // Remove skeleton
    const skel = document.getElementById("registry-skeleton");
    if (skel) skel.remove();
    renderSourceSummary(meta);

    // Update count everywhere
    if (dom.regCaseCount) dom.regCaseCount.textContent = cases.length;
    if (dom.overviewCaseCount) dom.overviewCaseCount.textContent = cases.length;

    if (cases.length === 0) {
      dom.registryList.innerHTML = `
        <div class="case-empty-state">
          <span class="material-symbols-outlined">folder_managed</span>
          <p class="case-empty-title">No curated real cases found</p>
          <p class="case-empty-copy">The default library only shows promoted, source-grounded case packets. Generated drafts stay in Case Lab until review.</p>
        </div>`;
      return;
    }

    dom.registryList.innerHTML = "";
    cases.forEach(c => renderCaseRow(c));
  } catch (e) {
    console.error("[Registry]", e);
    const skel = document.getElementById("registry-skeleton");
    if (skel) skel.remove();
    if (dom.registryList) {
      dom.registryList.innerHTML = `
        <div class="case-empty-state">
          <span class="material-symbols-outlined">error_outline</span>
          <p class="case-empty-title">Registry load failed</p>
          <p class="case-empty-copy">${escapeHTML(String(e))}</p>
        </div>`;
    }
  }
}

function renderCaseRow(caseData) {
  fetchedCases[caseData.id] = caseData;
  const confidence = Math.round((caseData.expected_output?.confidence || 0.95) * 100);
  const row = document.createElement("button");
  row.type = "button";
  row.className = "case-row clinical-case-row";
  row.setAttribute("data-case-id", caseData.id);
  row.setAttribute("data-source", sourceTone(caseData.source_type));
  row.setAttribute("aria-pressed", "false");
  row.setAttribute("aria-label", `Open curated clinical case ${caseData.title}`);
  row.onclick = () => showPatientDetail(caseData.id);
  row.innerHTML = `
    <span class="case-id">${escapeHTML(caseData.id)}</span>
    <div class="case-main">
      <strong>${escapeHTML(caseData.title)}</strong>
      <p>${escapeHTML(caseData.filename || caseData.id)}</p>
    </div>
    <span class="case-fam">${escapeHTML(sourceLabel(caseData.source_type, caseData.source_label))}</span>
    <span class="case-cell">${escapeHTML(caseData.diagnosis || "Awaiting analysis")}</span>
    <span class="case-cell">${caseData.real_case ? "curated" : "review"}</span>
    <div>
      <span class="case-conf">${confidence}%</span>
      <div class="conf-bar"><i style="width:${confidence}%"></i></div>
    </div>
  `;
  dom.registryList.appendChild(row);
}

function showPatientDetail(id) {
  const c = fetchedCases[id];
  if (!c) return;
  currentTestCaseId = id;

  if (dom.regDetailEmpty) dom.regDetailEmpty.classList.add("hidden");
  if (dom.regDetailContent) {
    dom.regDetailContent.classList.remove("hidden");
    dom.regDetailContent.classList.add("flex");
  }

  if (dom.detName) dom.detName.textContent = c.title;
  if (dom.detId) dom.detId.textContent = `UID: ${c.id}`;
  if (dom.detSource) dom.detSource.textContent = `${sourceLabel(c.source_type, c.source_label)} | ${c.filename || "curated packet"}`;
  if (dom.detPolicy) {
    dom.detPolicy.textContent = c.real_case
      ? "Promoted real-case packet. Draft/generated cases are not mixed into this library."
      : "Review-only packet. Confirm provenance before clinical use.";
  }
  if (dom.detNarrative) dom.detNarrative.textContent = c.patient_text || "(No narrative available)";
  if (dom.detHypothesis) dom.detHypothesis.textContent = c.expected_output?.reasoning || "Primary clinical suspicion requires swarm validation.";
  if (dom.detConfScore) dom.detConfScore.textContent = `Confidence: ${Math.round((c.expected_output?.confidence || 0.95) * 100)}%`;

  // Highlight selected row
  document.querySelectorAll("[data-case-id]").forEach(el => {
    el.classList.toggle("selected", el.getAttribute("data-case-id") === id);
    el.setAttribute("aria-pressed", el.getAttribute("data-case-id") === id ? "true" : "false");
  });
}

function closeRegDetail() {
  if (dom.regDetailEmpty) dom.regDetailEmpty.classList.remove("hidden");
  if (dom.regDetailContent) {
    dom.regDetailContent.classList.add("hidden");
    dom.regDetailContent.classList.remove("flex");
  }
  currentTestCaseId = null;
  document.querySelectorAll("[data-case-id]").forEach(el => el.classList.remove("selected"));
}

function loadCaseToReasoning() {
  if (!currentTestCaseId) return;
  const c = fetchedCases[currentTestCaseId];
  if (c && dom.input) {
    dom.input.value = c.patient_text || "";
    currentGroundTruth = c.expected_output_raw || c.expected_output || null;
    autoResize();
  }
  switchTab("reasoning");
  // Focus after transition
  setTimeout(() => { if (dom.input) dom.input.focus(); }, 100);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// UI helpers
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function renderCaseLabBucket(title, cases, toneName) {
  const values = Array.isArray(cases) ? cases : [];
  const items = values.length ? values.slice(0, 12).map((item) => `
    <article class="case-lab-item">
      <div>
        <div class="case-lab-item-title">${escapeHTML(item.title || "Untitled draft")}</div>
        <div class="case-lab-item-meta">
          <span>${escapeHTML(item.id || "draft")}</span>
          <span>${escapeHTML(item.source_label || item.source_origin || "generated")}</span>
          <span>${escapeHTML(item.decision_label || item.status || "review")}</span>
        </div>
      </div>
      <p>${escapeHTML(item.diagnosis || "Unknown expected anchor")}</p>
      ${toneName === "review" ? `
        <div class="case-lab-actions">
          <button type="button" data-case-action="promote" data-case-id="${escapeHTML(item.id || "")}">Promote</button>
          <button type="button" data-case-action="reject" data-case-id="${escapeHTML(item.id || "")}">Reject</button>
        </div>` : ""}
    </article>
  `).join("") : `
    <div class="case-lab-empty small">
      <span class="material-symbols-outlined">inventory_2</span>
      <p>No ${escapeHTML(title.toLowerCase())} cases.</p>
    </div>`;
  return `
    <section class="case-lab-bucket is-${toneName}">
      <header>
        <span>${escapeHTML(title)}</span>
        <strong>${values.length}</strong>
      </header>
      ${items}
    </section>
  `;
}

async function loadGeneratedCaseLab() {
  if (!dom.caseLabContent) return;
  dom.caseLabContent.innerHTML = `
    <div class="case-lab-empty">
      <span class="material-symbols-outlined">progress_activity</span>
      <p>Loading generated case buckets.</p>
    </div>`;
  try {
    const data = await fetchTypedPayload(`/api/generated-cases?ts=${Date.now()}`, { cache: "no-store" });
    const meta = data.meta || {};
    dom.caseLabContent.innerHTML = `
      <div class="case-lab-policy">
        <span class="source-chip is-policy">${escapeHTML(String(meta.policy || "conservative_auto_review").replace(/_/g, " "))}</span>
        <p>Drafts are review-only. Promoted cases can enter the curated library after provenance checks.</p>
      </div>
      ${renderCaseLabBucket("Needs review", data.review, "review")}
      ${renderCaseLabBucket("Promoted generated cases", data.accepted, "accepted")}
    `;
  } catch (e) {
    console.error("[CaseLab]", e);
    dom.caseLabContent.innerHTML = `
      <div class="case-lab-empty">
        <span class="material-symbols-outlined">error</span>
        <p>Case Lab could not load.</p>
        <small>${escapeHTML(String(e))}</small>
      </div>`;
  }
}

async function updateGeneratedCaseStatus(caseId, action) {
  if (!caseId) return;
  try {
    const resp = await fetch(`/api/generated-cases/${encodeURIComponent(caseId)}/${action}`, { method: "POST" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    await loadGeneratedCaseLab();
    await fetchTestCases();
    showToast(action === "promote" ? "Draft promoted to curated library" : "Draft rejected", "success", 2000);
  } catch (e) {
    console.error("[CaseLab action]", e);
    showToast(`Case Lab action failed: ${String(e)}`, "error", 3500);
  }
}

window.promoteGeneratedCase = (caseId) => updateGeneratedCaseStatus(caseId, "promote");
window.rejectGeneratedCase = (caseId) => updateGeneratedCaseStatus(caseId, "reject");

function formatTime(date) {
  return (date || new Date()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function autoResize() {
  if (!dom.input) return;
  dom.input.style.height = "auto";
  dom.input.style.height = Math.min(dom.input.scrollHeight, 160) + "px";

  // Update char counter
  const counter = document.getElementById("charCounter");
  if (counter) {
    const len = dom.input.value.length;
    counter.textContent = len > 0 ? String(len) : "";
    counter.className = "char-counter" + (len > 2000 ? " over" : len > 1500 ? " warn" : "");
  }
}

function setExample(text) {
  if (!dom.input) return;
  dom.input.value = text;
  autoResize();
  dom.input.focus();
}

function handleKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function scrollToBottom() {
  // Scroll the chat scroll area
  const area = dom.chatScrollArea;
  if (!area) return;
  const workspaceState = dom.analysisWorkspace?.dataset?.state;
  if (workspaceState === "running" && dom.analysisWorkspace) {
    area.scrollTop = Math.max(0, dom.analysisWorkspace.offsetTop - 8);
    return;
  }
  area.scrollTop = area.scrollHeight;
}

function escapeHTML(str) {
  if (!str) return "";
  return String(str).replace(/[&<>'"`]/g, t => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;", "`": "&#96;"
  }[t] || t));
}
window.escapeHTML = escapeHTML;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Messaging
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function sendMessage() {
  if (!isConnected) {
    showToast("Not connected to server. Retrying.", "error");
    return;
  }
  if (analysisInFlight) {
    showToast("An analysis is already running.", "error");
    return;
  }
  if (!dom.input) return;
  const text = dom.input.value.trim();
  if (!text) return;

  // Switch to reasoning tab automatically
  restoreComposerDock();
  switchTab("reasoning");
  closeDetailsDrawer();
  smartDrawerAutoOpened = false;

  // Remove empty state placeholder
  if (dom.messagesPlaceholder && dom.messagesPlaceholder.parentNode) {
    dom.messagesPlaceholder.remove();
    dom.messagesPlaceholder = null;
  }

  resetTracker();
  resetProgressUi();
  activateGlobalProgressBar();
  resetActivityStream();
  pushActivity("Case received. Preparing staged clinical analysis.", "info");
  appendUserMessage(text);
  if (dom.messages) dom.messages.setAttribute("aria-busy", "true");

  const payload = {
    type: "chat",
    patient_text: text,
    source: "interactive",
    execution_mode: isCloudGemini ? "cloud_gemini" : "local_qwen",
    metadata: {
      requested_operation_mode: currentRequestedOperationMode(),
      runtime_profile: currentRequestedRuntimeProfile(),
      ui_mode: mode,
      local_only: localMode,
      expected_output: currentGroundTruth || null,
    },
  };
  payload.context = payload.metadata;

  dom.input.value = "";
  currentGroundTruth = null;
  autoResize();
  analysisInFlight = true;
  setComposerBusy(true);
  updateAnalysisState("running");

  showTypingIndicator();
  currentMessageDiv = null; // will be created on first stage_start or token
  currentStage = null;
  currentThinkBlock = null;

  pipelineStartTime = Date.now();
  startLiveTimer();

  try {
    sendTypedMessage(payload);
  } catch (e) {
    handleError({ message: "Network connection lost. Please refresh." });
  }
}

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user-message user-msg";
  div.innerHTML = `
    <div>${escapeHTML(text)}</div>
    <div class="msg-time">${formatTime()}</div>
  `;
  dom.messages.appendChild(div);
  scrollToBottom();
}

let _typingIndicatorEl = null;

function showTypingIndicator() {
  if (_typingIndicatorEl) return;
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.id = "typing-indicator";
  el.innerHTML = `
    <div class="agent-avatar">
      <span class="material-symbols-outlined" aria-hidden="true">psychology</span>
    </div>
    <div class="typing-dots" title="AI is reasoning...">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  dom.messages.appendChild(el);
  _typingIndicatorEl = el;
  scrollToBottom();
}

function removeTypingIndicator() {
  if (_typingIndicatorEl) {
    _typingIndicatorEl.remove();
    _typingIndicatorEl = null;
  }
}

function createAIMessageContainer() {
  removeTypingIndicator();
  const wrap = document.createElement("div");
  wrap.className = "message agent-message agent-msg";
  wrap.innerHTML = `
    <div class="agent-avatar">
        <span class="material-symbols-outlined" aria-hidden="true">psychology</span>
    </div>
    <div class="agent-message-body">
      <div class="ai-bubble agent-bubble"></div>
      <div class="msg-time">${formatTime()}</div>
    </div>
  `;
  dom.messages.appendChild(wrap);
  scrollToBottom();
  return wrap.querySelector(".ai-bubble");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Pipeline Tracker
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function resetTracker() {
  const tl = document.getElementById("pipelineTimeline");
  if (tl) {
    tl.innerHTML = '<div class="timeline-kicker">Clinical stage tracker</div>' + STAGE_ORDER.map((stage) => {
      const info = STAGE_TITLES[stage] || { label: stage, icon: "radio_button_checked" };
      return `
        <div id="step-${stage}" class="timeline-node-v2 pending" data-stage="${stage}">
          <span class="material-symbols-outlined timeline-node-icon" aria-hidden="true">${info.icon}</span>
          <div class="timeline-node-copy">
            <p class="timeline-node-title">${escapeHTML(info.label)}</p>
            <p class="timeline-node-desc" id="step-${stage}-desc">Pending</p>
          </div>
        </div>
      `;
    }).join("");
  }
  resetSwarmMap();
}

const STAGE_TITLES = {
  INTAKE:       { label: "Intake & Risk",      icon: "person_search" },
  R2:           { label: "Evidence Retrieval", icon: "cloud_download" },
  DIFFERENTIAL: { label: "Differential",       icon: "psychology_alt" },
  OUTCOME:      { label: "Outcome Simulation", icon: "science" },
  VERIFY:       { label: "Chief Consilium",    icon: "fact_check" },
  ACTION:       { label: "Action Plan",        icon: "assignment" },
  MEMORY:       { label: "Memory Commit",      icon: "save" },
};
const STAGE_ACTIVITY_COPY = {
  INTAKE: "Parsing narrative, timing, risk, and symptom signals.",
  R2: "Retrieving evidence intent and source candidates.",
  DIFFERENTIAL: "Ranking diagnostic candidates and contradictions.",
  OUTCOME: "Stress-testing the leading differential against outcome risk.",
  VERIFY: "Checking safety posture and must-not-miss conflicts.",
  ACTION: "Drafting next steps and missing-data requests.",
  MEMORY: "Preparing the final packet for local learning.",
};
const STAGE_ORDER = ["INTAKE", "R2", "DIFFERENTIAL", "OUTCOME", "VERIFY", "ACTION", "MEMORY"];

function stopLiveTimer() { clearInterval(timerInterval); timerInterval = null; }

function normalizeStageName(stage) {
  const raw = String(stage || "").trim().toUpperCase().replace(/[-\s]+/g, "_");
  if (!raw || raw === "IDLE" || raw === "DONE") return raw.toLowerCase();
  const base = raw.replace(/_LOOP$/, "").replace(/^STAGE_/, "");
  const aliases = {
    RETRIEVE: "R2",
    RETRIEVAL: "R2",
    EVIDENCE: "R2",
    R2_RETRIEVAL: "R2",
    DIAGNOSIS: "DIFFERENTIAL",
    DIFFERENTIALS: "DIFFERENTIAL",
    CONSILIUM: "VERIFY",
    VERIFICATION: "VERIFY",
    OUTCOMES: "OUTCOME",
    PLAN: "ACTION",
    ACTIONS: "ACTION",
    LEARNING: "MEMORY",
  };
  return aliases[base] || base;
}

function setSwarmRunState(state) {
  const normalized = String(state || "idle").toLowerCase();
  if (dom.swarmMap) dom.swarmMap.dataset.runState = normalized;
  const pct = progressUiState.pct;
  if (dom.swarmProgressMetric && (pct === null || pct === undefined || !Number.isFinite(Number(pct)))) {
    dom.swarmProgressMetric.textContent = normalized;
  }
}

function resetSwarmMap() {
  completedSwarmStages = new Set();
  setSwarmRunState("idle");
  if (dom.swarmMap) dom.swarmMap.dataset.activeStage = "idle";
  if (dom.swarmMap) dom.swarmMap.style.setProperty("--swarm-progress", "0%");
  if (dom.swarmActiveLabel) dom.swarmActiveLabel.textContent = "Ready";
  if (dom.swarmFocusLabel) dom.swarmFocusLabel.textContent = "Idle";
  if (dom.swarmFocusNote) dom.swarmFocusNote.textContent = "No active diagnostic task.";
  if (dom.swarmProgressValue) dom.swarmProgressValue.textContent = "Awaiting a patient narrative.";
  if (dom.swarmPhaseMetric) dom.swarmPhaseMetric.textContent = "--";
  if (dom.swarmEvidenceMetric) dom.swarmEvidenceMetric.textContent = "0";
  if (dom.swarmProgressMetric) dom.swarmProgressMetric.textContent = "idle";
  dom.swarmNodes?.forEach((node) => {
    node.classList.remove("is-active", "is-complete", "is-error", "is-muted", "is-handoff");
    node.dataset.state = "pending";
  });
  dom.swarmEdges?.forEach((edge) => {
    edge.classList.remove("is-active", "is-complete", "is-error");
  });
}

function paintSwarm(activeStage, tone = "active") {
  const normalized = normalizeStageName(activeStage);
  const activeIndex = STAGE_ORDER.indexOf(normalized);
  if (dom.swarmMap) dom.swarmMap.dataset.activeStage = normalized || "idle";

  dom.swarmNodes?.forEach((node) => {
    const nodeStage = normalizeStageName(node.dataset.stage);
    const nodeIndex = STAGE_ORDER.indexOf(nodeStage);
    const isComplete = completedSwarmStages.has(nodeStage);
    const isActive = nodeStage === normalized && tone !== "complete";
    node.classList.toggle("is-complete", isComplete);
    node.classList.toggle("is-active", isActive);
    node.classList.toggle("is-error", tone === "error" && isActive);
    node.classList.toggle("is-muted", activeIndex >= 0 && nodeIndex > activeIndex + 1);
    node.dataset.state = isComplete ? "complete" : isActive ? tone : "pending";
    if (isActive && node.dataset.motionKey !== `${normalized}:${tone}`) {
      node.dataset.motionKey = `${normalized}:${tone}`;
      node.classList.remove("is-handoff");
      void node.offsetWidth;
      node.classList.add("is-handoff");
      window.setTimeout(() => node.classList.remove("is-handoff"), 420);
    }
    if (!isActive) delete node.dataset.motionKey;
  });

  dom.swarmEdges?.forEach((edge) => {
    const from = normalizeStageName(edge.dataset.from);
    const to = normalizeStageName(edge.dataset.to);
    const fromComplete = completedSwarmStages.has(from);
    const toComplete = completedSwarmStages.has(to);
    const active = from === normalized || to === normalized;
    edge.classList.toggle("is-complete", fromComplete && (toComplete || to === normalized));
    edge.classList.toggle("is-active", active && tone !== "complete");
    edge.classList.toggle("is-error", active && tone === "error");
  });
}

function setSwarmStage(stage, tone = "active") {
  const normalized = normalizeStageName(stage);
  if (!normalized || normalized === "idle" || normalized === "done") return;
  if (tone === "complete") completedSwarmStages.add(normalized);
  setSwarmRunState(tone === "error" ? "error" : "running");
  paintSwarm(normalized, tone);
  if (dom.swarmProgressValue) {
    const index = STAGE_ORDER.indexOf(normalized);
    const label = STAGE_TITLES[normalized]?.label || normalized;
    const copy = STAGE_ACTIVITY_COPY[normalized] || (index >= 0 ? `${index + 1}/7 ${label}` : label);
    dom.swarmProgressValue.textContent = copy;
    if (dom.swarmActiveLabel) dom.swarmActiveLabel.textContent = label;
    if (dom.swarmFocusLabel) dom.swarmFocusLabel.textContent = label;
    if (dom.swarmFocusNote) dom.swarmFocusNote.textContent = copy;
    if (dom.swarmPhaseMetric) dom.swarmPhaseMetric.textContent = index >= 0 ? `${index + 1}/7` : "--";
  }
  if (dom.swarmMap) {
    const index = STAGE_ORDER.indexOf(normalized);
    const percent = index >= 0 ? (index / Math.max(1, STAGE_ORDER.length - 1)) * 100 : 0;
    dom.swarmMap.style.setProperty("--swarm-progress", `${Math.max(0, Math.min(100, percent))}%`);
  }
}

function setStageActive(stage, description = "") {
  const normalizedStage = normalizeStageName(stage);
  const tl = document.getElementById("pipelineTimeline");
  if (!tl) return;
  const info = STAGE_TITLES[normalizedStage] || { label: normalizedStage || stage, icon: "radio_button_checked" };
  tl.querySelectorAll(".timeline-node-v2.active").forEach((item) => item.classList.remove("active"));

  let node = document.getElementById(`step-${normalizedStage}`);
  if (!node) {
    node = document.createElement("div");
    node.id = `step-${normalizedStage}`;
    node.className = "timeline-node-v2 pending";
    node.innerHTML = `
      <span class="material-symbols-outlined timeline-node-icon" aria-hidden="true">${info.icon}</span>
      <div class="timeline-node-copy">
          <p class="timeline-node-title">${escapeHTML(info.label)}</p>
          <p class="timeline-node-desc" id="step-${normalizedStage}-desc">Pending</p>
      </div>
    `;
    tl.appendChild(node);
  }

  node.classList.remove("pending", "complete", "error");
  node.classList.add("active");
  const icon = node.querySelector(".material-symbols-outlined");
  if (icon) {
    icon.className = "material-symbols-outlined timeline-node-icon";
    icon.textContent = info.icon;
  }
  const desc = document.getElementById(`step-${normalizedStage}-desc`);
  const descText = description || "Processing";
  if (desc) desc.textContent = descText;
  currentStage = normalizedStage;
  setSwarmStage(normalizedStage, "active");
  renderProgressState({ state: "running", stage: normalizedStage, caption: descText });
  pushActivity(`${info.label}: ${descText}`, "info");
}

function completeStage(stage) {
  const normalizedStage = normalizeStageName(stage);
  const el = document.getElementById(`step-${normalizedStage}`);
  if (!el) return;
  el.classList.remove("active", "pending", "error");
  el.classList.add("complete");
  const icon = el.querySelector(".material-symbols-outlined");
  if (icon) {
    icon.className = "material-symbols-outlined timeline-node-icon";
    icon.textContent = "check_circle";
  }
  const desc = document.getElementById(`step-${normalizedStage}-desc`);
  if (desc) desc.textContent = "Done";
  setSwarmStage(normalizedStage, "complete");
  pushActivity(`${STAGE_TITLES[normalizedStage]?.label || normalizedStage} complete`, "success");
}

function startLiveTimer() {
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - pipelineStartTime) / 1000).toFixed(1);
    if (dom.trackerStats) dom.trackerStats.textContent = `${currentStage || "Swarm"} | ${elapsed}s`;
    if (dom.analysisElapsed) dom.analysisElapsed.textContent = `${elapsed}s`;
  }, 200);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Toast
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
const TOAST_ICONS = {
  info:    "info",
  success: "check",
  error:   "error",
  warning: "warning",
};

function showToast(msg, type = "info", duration = 3500) {
  const container = document.getElementById("toast-container") || document.body;
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.setAttribute("role", type === "error" ? "alert" : "status");
  t.setAttribute("aria-live", type === "error" ? "assertive" : "polite");
  const dismissId = `toast-dismiss-${Date.now()}`;
  t.innerHTML = `
    <div class="toast-glyph"><span class="material-symbols-outlined">${TOAST_ICONS[type] || "info"}</span></div>
    <div class="toast-body"><span class="toast-title">${escapeHTML(msg)}</span></div>
    <button class="toast-dismiss" id="${dismissId}" aria-label="Dismiss notification"><span class="material-symbols-outlined">close</span></button>
  `;
  container.appendChild(t);

  t.querySelector(`#${dismissId}`).onclick = (e) => { e.stopPropagation(); dismissToast(t); };
  t.onclick = () => dismissToast(t);

  setTimeout(() => dismissToast(t), duration);
}

function dismissToast(el) {
  if (!el || !el.parentNode) return;
  el.classList.add("toast-out");
  el.addEventListener("animationend", () => el.remove(), { once: true });
  // Fallback in case animationend doesn't fire
  setTimeout(() => { if (el.parentNode) el.remove(); }, 400);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// WebSocket Message Router
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function handleServerMessage(data) {
  switch (data.type) {
    case "ack":             handleAck(data); break;
    case "cancelled":       handleCancelled(data); break;
    case "stage_start":     handleStageStart(data); break;
    case "stage_complete":  handleStageComplete(data); break;
    case "stage_progress":  handleStageProgress(data); break;
    case "eta_update":      handleEtaUpdate(data); break;
    case "evidence_tick":   handleEvidenceTick(data); break;
    case "candidate_preview": handleCandidatePreview(data); break;
    case "token":           handleToken(data); break;
    case "thinking_start":  handleThinkingStart(data); break;
    case "thinking_token":  handleThinkingToken(data); break;
    case "thinking_end":    /* handled by stage_complete */ break;
    case "info":            handleInfo(data); break;
    case "final_result":    handleFinalResult(data); break;
    case "error":           handleError(data); break;
    case "health":          handleHealth(data); break;
    default:                console.log("[WS unhandled]", data.type);
  }
  scrollToBottom();
}

function handleHealth(data) {
  const runtime = data?.data?.runtime || {};
  const llmServer = data?.data?.llm_server || {};
  const rttMs = healthProbeStartedAt > 0 ? Math.max(0, performance.now() - healthProbeStartedAt) : 0;
  healthProbeStartedAt = 0;
  runtimeEffectiveState = {
    ...runtimeEffectiveState,
    ...runtime,
    websocket_rtt_ms: rttMs > 0 ? Math.round(rttMs) : runtimeEffectiveState.websocket_rtt_ms,
  };

  const vramRaw = runtime.free_vram_gb ?? runtime.free_vram;
  const vramStr = (vramRaw != null) ? `${Number(vramRaw).toFixed(2)} GB` : "--";

  if (dom.memT1) dom.memT1.textContent = vramStr;
  if (dom.footerVram) dom.footerVram.textContent = `VRAM: ${vramStr}`;
  updateRuntimeUi(runtimeEffectiveState, llmServer);
}

function handleStageStart(data) {
  const stage = data.stage;
  setStageActive(stage, data.description || data.title || "Processing");
  currentThinkBlock = null;
}

function handleStageComplete(data) {
  const stage = data.stage;
  completeStage(stage);
  if (currentThinkBlock && currentThinkBlock.getAttribute("data-stage") === stage) currentThinkBlock = null;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Phase 1b - progress / ETA / evidence / candidate preview
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
const PHASE_INSIGHTS = {
  INTAKE: [
    "Parsing vital signs and chief complaint...",
    "Extracting structured findings from narrative...",
    "Scoring clinical red flags...",
  ],
  R2: [
    "Planning evidence retrieval intents...",
    "Querying clinical literature and guidelines...",
    "Cross-checking against case memory...",
    "Scoring retrieved evidence for relevance...",
  ],
  DIFFERENTIAL: [
    "Launching adaptive swarm panel...",
    "General physician and specialists weighing in...",
    "Running red-team challenger on top candidates...",
    "Calibrating epidemiological priors...",
  ],
  OUTCOME: [
    "Simulating test-time mortality outcomes...",
    "Stress-testing top hypotheses for missed diagnoses...",
  ],
  VERIFY: [
    "Chief consilium synthesizing final consensus...",
    "Safety gate: checking must-not-miss coverage...",
  ],
  ACTION: [
    "Generating clinical directives and next steps...",
  ],
  MEMORY: [
    "Committing reasoning trace to cognitive store...",
  ],
};

let _insightInterval = null;
let _insightIndex = 0;

function startPhaseInsightRotator(phase) {
  const strip = document.getElementById("phaseInsightStrip");
  if (!strip) return;
  const messages = PHASE_INSIGHTS[phase];
  if (!messages || !messages.length) {
    strip.classList.add("hidden");
    return;
  }
  strip.classList.remove("hidden");
  _insightIndex = 0;
  const render = () => {
    strip.classList.remove("fading");
    // Force reflow so the animation re-runs.
    void strip.offsetWidth;
    strip.textContent = messages[_insightIndex % messages.length];
    strip.classList.add("fading");
    _insightIndex += 1;
  };
  render();
  if (_insightInterval) clearInterval(_insightInterval);
  _insightInterval = setInterval(render, 3800);
}

function stopPhaseInsightRotator() {
  if (_insightInterval) clearInterval(_insightInterval);
  _insightInterval = null;
  const strip = document.getElementById("phaseInsightStrip");
  if (strip) {
    strip.textContent = "";
    strip.classList.add("hidden");
  }
}

function activateGlobalProgressBar() {
  const bar = document.getElementById("globalProgressBar");
  if (!bar) return;
  bar.classList.remove("done");
  bar.classList.add("active");
  setProgressIndeterminate();
}

function finishGlobalProgressBar() {
  const bar = document.getElementById("globalProgressBar");
  if (!bar) return;
  bar.classList.add("done");
  setProgressPct(100);
  renderProgressState({ state: "completed", pct: 100, caption: "Analysis completed. Final clinical packet is ready." });
  setTimeout(() => {
    bar.classList.remove("active");
  }, 1200);
}

function setProgressPct(pct) {
  const bar = document.getElementById("globalProgressBar");
  const fill = document.getElementById("globalProgressFill");
  const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
  if (fill) fill.style.width = `${safePct}%`;
  if (bar) {
    bar.setAttribute("aria-valuenow", String(Math.round(safePct)));
    bar.removeAttribute("aria-busy");
  }
  renderProgressState({ pct: safePct });
}

function setProgressIndeterminate() {
  const bar = document.getElementById("globalProgressBar");
  const fill = document.getElementById("globalProgressFill");
  if (fill) fill.style.width = "0%";
  if (bar) {
    bar.removeAttribute("aria-valuenow");
    bar.setAttribute("aria-busy", "true");
  }
  renderProgressState({ state: "running", pct: null, caption: "Analysis started. Waiting for measured stage progress." });
}

function handleAck(_data) {
  activateGlobalProgressBar();
  pushActivity("Backend acknowledged the case and opened the pipeline.", "success");
  showResultSkeleton();
}

function handleStageProgress(data) {
  const pct = Number(data?.progressPct ?? data?.pct);
  if (Number.isFinite(pct)) setProgressPct(pct);
  if (data?.stage && data.stage !== "DONE") {
    setSwarmStage(data.stage, "active");
    renderProgressState({
      state: "running",
      stage: normalizeStageName(data.stage),
      caption: data.message || data.description || `${stageLabelFor(data.stage)} in progress`,
    });
  }
  if (data?.stage && data.stage !== "DONE" && data.stage !== currentStage) {
    startPhaseInsightRotator(data.stage);
  }
  if (data?.stage === "DONE") {
    finishGlobalProgressBar();
    stopPhaseInsightRotator();
  }
}

function handleEtaUpdate(data) {
  const badge = document.getElementById("etaBadge");
  const text = document.getElementById("etaBadgeText");
  if (!badge || !text) return;
  const remaining = Math.max(0, Number(data?.remaining_s) || 0);
  if (remaining <= 0) {
    badge.classList.add("hidden");
    return;
  }
  badge.classList.remove("hidden");
  const mins = Math.floor(remaining / 60);
  const secs = Math.round(remaining % 60);
  text.textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  const conf = Number(data?.confidence) || 0.4;
  badge.style.opacity = String(0.5 + Math.min(0.5, conf * 0.8));
}

function handleEvidenceTick(data) {
  const ticker = document.getElementById("evidenceTicker");
  const digit = document.getElementById("evidenceTickerCount");
  const count = Math.max(0, Number(data?.count) || 0);
  setSwarmStage("R2", "active");
  if (dom.evCoverageCount) dom.evCoverageCount.textContent = String(count);
  if (data?.contradiction_mass != null && dom.evContradiction) {
    dom.evContradiction.textContent = `${Math.round(Number(data.contradiction_mass) * 100)}%`;
  }
  if (data?.query_hygiene != null && dom.evQueryHygiene) {
    dom.evQueryHygiene.textContent = String(data.query_hygiene).replace(/_/g, " ");
  }
  if (count > 0 && !smartDrawerAutoOpened && document.body.dataset.view === "reasoning") {
    smartDrawerAutoOpened = true;
    rrSwitchPanelTab("evidence");
    pushActivity("Evidence packet is ready in Details.", "info");
  }
  if (!ticker || !digit) return;
  ticker.classList.remove("hidden");
  if (String(count) !== digit.textContent) {
    digit.textContent = String(count);
    digit.classList.remove("flash");
    void digit.offsetWidth;
    digit.classList.add("flash");
    if (count > progressUiState.evidenceCount) {
      pushActivity(`${count} evidence item${count === 1 ? "" : "s"} attached to the packet.`, "info");
    }
    progressUiState.evidenceCount = count;
    if (dom.swarmEvidenceMetric) dom.swarmEvidenceMetric.textContent = String(count);
  }
}

function handleCandidatePreview(data) {
  const strip = document.getElementById("candidatePreviewStrip");
  if (!strip) return;
  const candidates = Array.isArray(data?.candidates) ? data.candidates : [];
  if (!candidates.length) return;
  setSwarmStage("DIFFERENTIAL", "active");
  strip.classList.remove("hidden");
  pushActivity(`${candidates.length} differential candidate${candidates.length === 1 ? "" : "s"} surfaced.`, "info");
  strip.innerHTML = `
    <div class="candidate-preview-strip-title">
      <span class="material-symbols-outlined !text-[12px]">psychology_alt</span>
      Candidates
      <small>${candidates.length} active</small>
    </div>
  ` + candidates.slice(0, 2).map((c) => {
    const rank = Number(c?.rank) || 0;
    const dx = String(c?.dx || "").replace(/_/g, " ");
    const conf = Number(c?.conf) || 0;
    const confText = conf > 0 ? `${Math.round(conf * 100)}%` : "-";
    return `
      <div class="candidate-preview-card">
        <span class="candidate-preview-rank">${rank}</span>
        <span class="candidate-preview-dx">${escapeHTML(dx)}</span>
        <span class="candidate-preview-conf">${confText}</span>
      </div>
    `;
  }).join("");
}

function showResultSkeleton() {
  // Avoid stacking multiple skeletons across reruns.
  const existing = document.getElementById("resultSkeleton");
  if (existing) existing.remove();
  const host = dom.resultPreviewSlot || dom.messages;
  if (!host) return;
  const wrap = document.createElement("div");
  wrap.id = "resultSkeleton";
  wrap.className = "result-skeleton-stack";
  wrap.innerHTML = `
    <div class="result-skeleton">
      <div class="skeleton-line short"></div>
      <div class="skeleton-block"></div>
      <div class="skeleton-line long"></div>
      <div class="skeleton-line med"></div>
    </div>
    <div class="result-skeleton">
      <div class="skeleton-line short"></div>
      <div class="skeleton-line long"></div>
      <div class="skeleton-line med"></div>
    </div>
  `;
  host.appendChild(wrap);
}

function clearResultSkeleton() {
  const existing = document.getElementById("resultSkeleton");
  if (existing) existing.remove();
}

function resetProgressUi() {
  setProgressPct(0);
  const bar = document.getElementById("globalProgressBar");
  if (bar) { bar.classList.remove("active", "done"); }
  resetSwarmMap();
  renderProgressState({ state: "idle", stage: null, pct: null, caption: "Enter a patient narrative to start a focused diagnostic pass." });
  if (dom.analysisElapsed) dom.analysisElapsed.textContent = "0.0s";
  progressUiState.evidenceCount = 0;
  smartDrawerAutoOpened = false;
  if (dom.evCoverageCount) dom.evCoverageCount.textContent = "0";
  if (dom.evContradiction) dom.evContradiction.textContent = "-";
  if (dom.evQueryHygiene) dom.evQueryHygiene.textContent = "-";
  resetActivityStream();
  const eta = document.getElementById("etaBadge");
  if (eta) eta.classList.add("hidden");
  const ev = document.getElementById("evidenceTicker");
  if (ev) ev.classList.add("hidden");
  const strip = document.getElementById("candidatePreviewStrip");
  if (strip) { strip.classList.add("hidden"); strip.innerHTML = ""; }
  stopPhaseInsightRotator();
  clearResultSkeleton();
}

function handleToken(data) {
  if (!currentThinkBlock) return;
  const ct = currentThinkBlock.querySelector(".think-content");
  if (ct) {
    ct.textContent += (data.content || "");
    scrollToBottom();
  }
}

function handleThinkingStart(data) {
  if (!currentThinkBlock) return;
  const sub = document.createElement("div");
  sub.className = "think-sub";
  sub.innerHTML = `
    <div class="think-sub-head">
        <span class="material-symbols-outlined">psychology</span>
        <span>Internal Reasoning</span>
    </div>
    <div class="think-sub-content"></div>
  `;
  const tc = currentThinkBlock.querySelector(".think-content");
  if (tc) tc.appendChild(sub);
}

function handleThinkingToken(data) {
  if (!currentThinkBlock) return;
  const subs = currentThinkBlock.querySelectorAll(".think-sub-content");
  const last = subs[subs.length - 1];
  if (last) {
    last.textContent += (data.content || "");
    scrollToBottom();
  }
}

function handleInfo(data) {
  // Update stage description text if we know which stage
  if (currentStage) {
    const desc = document.getElementById(`step-${currentStage}-desc`);
    if (desc && data.content) {
      desc.textContent = String(data.content).substring(0, 60);
      renderProgressState({ state: "running", stage: currentStage, caption: String(data.content).substring(0, 140) });
    }
  }
}

function handleFinalResult(data) {
  removeTypingIndicator();
  stopLiveTimer();
  analysisInFlight = false;
  if (dom.messages) dom.messages.setAttribute("aria-busy", "false");
  updateAnalysisState("completed");
  setComposerBusy(false);

  // Mark last stage complete
  if (currentStage) completeStage(currentStage);
  setStageActive("MEMORY");
  renderProgressState({
    state: "completed",
    stage: "MEMORY",
    pct: 100,
    caption: "Analysis completed. Final clinical packet is ready.",
  });
  setTimeout(() => {
    completeStage("MEMORY");
    setSwarmRunState("completed");
    if (dom.swarmProgressValue) dom.swarmProgressValue.textContent = "decision ready";
    renderProgressState({
      state: "completed",
      stage: "MEMORY",
      pct: 100,
      caption: "Analysis completed. Final clinical packet is ready.",
    });
  }, 600);

  // Phase 1b: finish progress UI
  finishGlobalProgressBar();
  stopPhaseInsightRotator();
  clearResultSkeleton();
  const eta = document.getElementById("etaBadge"); if (eta) eta.classList.add("hidden");
  pushActivity("Final diagnostic packet rendered.", "success");

  // Update tracker stats
  const elapsed = pipelineStartTime ? ((Date.now() - pipelineStartTime) / 1000).toFixed(1) : "?";
  if (dom.trackerStats) dom.trackerStats.textContent = `Complete | ${elapsed}s`;

  // Build result HTML
  const result = data.view || data || {};
  let html = `<div class="analysis-complete-note">
    <div><span class="material-symbols-outlined" aria-hidden="true">check_circle</span>Analysis Complete</div>
    <p>Diagnostic reasoning session completed.</p>
  </div>`;

  if (window.RRRIEFinalResultView && typeof window.RRRIEFinalResultView.buildHtml === "function") {
    try {
      const built = window.RRRIEFinalResultView.buildHtml(result);
      if (built) html = built;
    } catch (e) {
      console.error("[FinalResult render]", e);
    }
  }

  if (dom.messagesPlaceholder && dom.messagesPlaceholder.parentNode) {
    dom.messagesPlaceholder.remove();
    dom.messagesPlaceholder = null;
  }
  const host = dom.resultPreviewSlot || dom.messages;
  if (!host) return;
  host.querySelectorAll(".final-result-view").forEach((node) => node.remove());
  const finalDiv = document.createElement("div");
  finalDiv.className = "final-result-view";
  finalDiv.innerHTML = html;
  host.appendChild(finalDiv);
  currentMessageDiv = null;
  dockComposerAfterResult(finalDiv);
  requestAnimationFrame(() => {
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    finalDiv.scrollIntoView({ block: "start", behavior: reducedMotion ? "auto" : "smooth" });
  });

  showToast("Analysis complete", "success", 2500);
  requestHealthSnapshot();
  currentStage = null;
}

function handleError(data) {
  removeTypingIndicator();
  stopLiveTimer();
  analysisInFlight = false;
  if (dom.messages) dom.messages.setAttribute("aria-busy", "false");
  resetProgressUi();
  restoreComposerDock();
  updateAnalysisState("error");
  setComposerBusy(false);
  if (currentStage) {
    setSwarmStage(currentStage, "error");
    const el = document.getElementById(`step-${normalizeStageName(currentStage)}`);
    if (el) {
      el.classList.remove("active", "pending", "complete");
      el.classList.add("error");
    }
  }

  const msg = data.message || data.content || "An unexpected error occurred.";
  renderProgressState({ state: "error", pct: null, caption: msg });
  pushActivity(msg, "error");
  if (currentMessageDiv) {
    const err = document.createElement("div");
    err.className = "error-card";
    err.innerHTML = `<span class="material-symbols-outlined">error</span><span>${escapeHTML(msg)}</span>`;
    currentMessageDiv.appendChild(err);
    scrollToBottom();
  } else {
    showToast(msg, "error");
  }
  requestHealthSnapshot();
  currentStage = null;
}

function handleCancelled(data) {
  removeTypingIndicator();
  stopLiveTimer();
  analysisInFlight = false;
  if (dom.messages) dom.messages.setAttribute("aria-busy", "false");
  resetProgressUi();
  restoreComposerDock();
  updateAnalysisState("cancelled");
  setComposerBusy(false);
  if (currentStage) {
    setSwarmStage(currentStage, "active");
    setSwarmRunState("cancelled");
  }

  const msg = data?.content || "Active analysis cancelled.";
  renderProgressState({ state: "cancelled", pct: null, caption: msg });
  pushActivity(msg, "warn");
  if (currentMessageDiv) {
    const note = document.createElement("div");
    note.className = "analysis-cancel-note";
    note.innerHTML = `<span class="material-symbols-outlined" aria-hidden="true">cancel</span><span>${escapeHTML(msg)}</span>`;
    currentMessageDiv.appendChild(note);
    scrollToBottom();
  } else {
    showToast(msg, "info");
  }
  requestHealthSnapshot();
  currentStage = null;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Entry Point
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
function stopAnalysis() {
  if (!analysisInFlight) return;
  if (dom.stopBtn) dom.stopBtn.disabled = true;
  if (dom.mobileStopBtn) dom.mobileStopBtn.disabled = true;
  if (dom.trackerStats) dom.trackerStats.textContent = "Swarm | Cancelling";
  updateAnalysisState("cancelling");
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      sendTypedMessage({ type: "cancel" });
      return;
    } catch (e) {
      console.error("[WS cancel]", e);
    }
  }

  stopLiveTimer();
  analysisInFlight = false;
  setComposerBusy(false);
  if (dom.messages) dom.messages.setAttribute("aria-busy", "false");
  updateAnalysisState(isConnected ? "idle" : "offline");
  if (dom.trackerStats) dom.trackerStats.textContent = "Swarm | Cancel unavailable";
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Cognitive Learning - Feedback & Dashboard
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

window._submitFeedbackFromBtn = function(btn) {
  const key = btn.dataset.fbkey;
  const outcome = btn.dataset.outcome;
  const fbData = window[key] || {};
  const section = btn.closest("[data-fbkey]") || btn.parentElement?.parentElement;
  const gtInput = section?.querySelector(".feedback-gt-input");
  const statusEl = section?.querySelector(".feedback-status-span");
  const groundTruth = gtInput ? gtInput.value.trim() : "";
  submitDxFeedback(outcome, fbData.caseId, fbData.topLabel, fbData.findingsSummary, statusEl, groundTruth);
};

async function submitDxFeedback(outcome, caseId, topLabel, findingsSummary, statusEl, groundTruth) {
  if (statusEl) statusEl.textContent = "Sending...";
  try {
    const payload = {
      case_id: caseId || "unknown",
      outcome,
      eligible_for_learning: true,
      feedback: {
        top_label: topLabel,
        findings_summary: findingsSummary,
        ...(groundTruth ? { diagnosis: groundTruth } : {}),
      },
    };
    const resp = await fetch("/api/vnext/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (statusEl) {
      statusEl.textContent = resp.ok
        ? (outcome === "correct" ? "Recorded - model updated" : "Recorded - model penalized")
        : "Error sending feedback";
      statusEl.style.color = resp.ok ? (outcome === "correct" ? "#255f9f" : "#991b1b") : "#dc2626";
    }
  } catch (e) {
    if (statusEl) { statusEl.textContent = "Network error"; statusEl.style.color = "#dc2626"; }
  }
}

async function loadLearningStats() {
  try {
    const resp = await fetch("/api/learning/stats");
    if (!resp.ok) return;
    const stats = await resp.json();
    const el = (id) => document.getElementById(id);
    if (el("learning-cases-count")) el("learning-cases-count").textContent = stats.cases_in_memory ?? "-";
    if (el("learning-syndromes-count")) el("learning-syndromes-count").textContent = stats.known_syndromes ?? "-";
    if (el("learning-mlp-updates")) el("learning-mlp-updates").textContent = stats.mlp_updates ?? "-";
  } catch (_) {}
}

window.submitDxFeedback = submitDxFeedback;
window.loadLearningStats = loadLearningStats;

// â”€â”€ v7.2 right panel tab switcher â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function rrSwitchPanelTab(name) {
  const tabs = document.querySelectorAll(".rr-panel-tab");
  const panels = document.querySelectorAll(".rr-panel-tabpanel");
  tabs.forEach((t) => {
    const active = t.dataset.tab === name;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
  });
  panels.forEach((p) => p.classList.toggle("active", p.dataset.tab === name));
}
window.rrSwitchPanelTab = rrSwitchPanelTab;

// Mirror health snapshot into runtime tab metrics
function rrUpdateRuntimeTab(snapshot) {
  if (!snapshot) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el && val != null) el.textContent = val; };
  set("runtimeVram", snapshot.vram || "-");
  set("runtimeRtt", snapshot.rtt || "-");
  set("runtimeStatus", snapshot.status || "Connected");
}
window.rrUpdateRuntimeTab = rrUpdateRuntimeTab;

window.addEventListener("load", init);
