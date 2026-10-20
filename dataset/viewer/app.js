const SEVERITY_COLORS = { CRITICAL: "#7b241c", HIGH: "#e74c3c", MEDIUM: "#e67e22", LOW: "#27ae60", UNKNOWN: "#6b7a8d" };
const PLATFORM_LABELS = { x: "X", reddit: "Reddit", telegram: "Telegram", discord: "Discord" };
const state = { allReports: [], reports: [], selectedIndex: 0 };
const els = {
  metricReports: document.getElementById("metricReports"),
  metricCreated: document.getElementById("metricCreated"),
  pageSummary: document.getElementById("pageSummary"),
  detailPanel: document.getElementById("detailPanel"),
  prevPageButton: document.getElementById("prevPageButton"),
  nextPageButton: document.getElementById("nextPageButton"),
  threatCategoryFilter: document.getElementById("threatCategoryFilter"),
  threatSeverityFilter: document.getElementById("threatSeverityFilter"),
  platformFilter: document.getElementById("platformFilter"),
  reportIdInput: document.getElementById("reportIdInput"),
};

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
function formatNumber(value) { return Number(value || 0).toLocaleString("en-US"); }
function formatDate(value) {
  if (!value) { return "No date"; }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) { return value; }
  return parsed.toLocaleString("en-US", { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}
function platformLabels(values) { return (values || []).map((p) => PLATFORM_LABELS[p] || p); }
function listText(values, fallback = "Unspecified") {
  const items = Array.isArray(values) ? values.filter(Boolean) : [];
  return items.length ? items.join(", ") : fallback;
}
function applyPayload(payload) {
  if (!payload || !Array.isArray(payload.reports)) {
    renderEmptyDetail("Dataset unavailable", "reports_viewer_data.js did not expose a reports array.");
    return;
  }
  const severityRank = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  const rank = (r) => (r.severity in severityRank ? severityRank[r.severity] : 4);
  state.allReports = [...payload.reports].sort((a, b) => rank(a) - rank(b) || Number(a.report_index || 0) - Number(b.report_index || 0));
  populateFilters();
  applyFilters();
}
function uniqueValues(values) { return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b))); }
function setSelectOptions(select, values, allLabel) {
  const current = select.value;
  select.innerHTML = [`<option value="">${escapeHtml(allLabel)}</option>`].concat(values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`)).join("");
  select.value = values.includes(current) ? current : "";
}
function populateFilters() {
  setSelectOptions(els.threatCategoryFilter, uniqueValues(state.allReports.map((r) => r.paper_category)), "All categories");
  setSelectOptions(els.threatSeverityFilter, ["CRITICAL", "HIGH", "MEDIUM", "LOW"].filter((s) => state.allReports.some((r) => r.severity === s)), "All severities");
  setSelectOptions(els.platformFilter, ["x", "reddit", "telegram", "discord"].filter((p) => state.allReports.some((r) => (r.platforms || []).includes(p))).map((p) => PLATFORM_LABELS[p]), "All platforms");
}
function applyFilters() {
  const category = els.threatCategoryFilter.value;
  const severity = els.threatSeverityFilter.value;
  const platform = Object.keys(PLATFORM_LABELS).find((k) => PLATFORM_LABELS[k] === els.platformFilter.value) || "";
  const idQuery = els.reportIdInput.value.trim().toLowerCase();
  state.reports = state.allReports.filter((r) =>
    (!category || r.paper_category === category) &&
    (!severity || r.severity === severity) &&
    (!platform || (r.platforms || []).includes(platform)) &&
    (!idQuery || String(r.report_id).toLowerCase().includes(idQuery)));
  state.selectedIndex = 0;
  renderAll();
}
function selectedReport() { return state.reports[state.selectedIndex] || null; }
function renderAll() {
  const report = selectedReport();
  const total = state.reports.length;
  els.metricReports.textContent = `${formatNumber(total)} / ${formatNumber(state.allReports.length)}`;
  els.metricCreated.textContent = formatDate(report?.created_at);
  els.pageSummary.textContent = total ? `${formatNumber(state.selectedIndex + 1)} of ${formatNumber(total)}` : "0 reports";
  els.prevPageButton.disabled = state.selectedIndex <= 0;
  els.nextPageButton.disabled = state.selectedIndex >= total - 1 || total === 0;
  renderDetail();
}
function selectNextReport(offset) {
  const total = state.reports.length;
  if (!total) { return; }
  state.selectedIndex = Math.min(Math.max(0, state.selectedIndex + offset), total - 1);
  renderAll();
}
function metaChip(label, value, modifier) {
  return `<span class="meta-chip metric-meta-chip metric-meta-chip--${escapeHtml(modifier)}"><span class="meta-chip-label">${escapeHtml(label)}</span><span class="meta-chip-value">${escapeHtml(value)}</span></span>`;
}
function sectionHtml(title, body) {
  return `<div class="section-card"><div class="section-title">${escapeHtml(title)}</div>${body || '<div class="empty-text">None reported.</div>'}</div>`;
}
function snapshotSectionHtml(title, body) {
  return `<div class="snapshot-evidence-block"><div class="snapshot-subtitle">${escapeHtml(title)}</div>${body || '<div class="empty-text">None reported.</div>'}</div>`;
}
function findingsHtml(groups) {
  if (!groups.length) { return ""; }
  let n = 0;
  return groups.map((g) => `<div class="finding-group-title">${escapeHtml(g.heading)}</div>` + (g.items || []).map((item) => {
    n += 1;
    return `<div class="finding-item"><span class="finding-num">${n}</span><p class="finding-text">${escapeHtml(item)}</p></div>`;
  }).join("")).join("");
}
function iocsHtml(items) {
  if (!items.length) { return ""; }
  const rows = items.slice(0, 80).map((i) => `<tr><td class="ioc-type">${escapeHtml(i.type)}</td><td class="ioc-val">${escapeHtml(i.value)}</td><td>${escapeHtml(i.extraction_type || "")}</td><td>${escapeHtml((i.platforms || []).join(", "))}</td></tr>`).join("");
  return `<div class="ioc-table-wrap"><table class="ioc-table kv-table"><thead><tr><th>Type</th><th>Value</th><th>Extraction</th><th>Source platform</th></tr></thead><tbody>${rows}</tbody></table></div>${items.length > 80 ? '<div class="section-note block-note">Showing first 80 IoCs.</div>' : ""}`;
}
function chipsHtml(items) {
  return items.length ? `<div class="ttp-list">${items.map((t) => `<span class="ttp-chip">${escapeHtml(t)}</span>`).join("")}</div>` : "";
}
function recommendationsHtml(items) {
  if (!items.length) { return ""; }
  const groups = [["immediate", "Immediate Actions"], ["short_term", "Short-Term Actions"], ["long_term", "Long-Term Actions"]];
  const render = (list) => `<ul>${list.map((r) => `<li><strong>${escapeHtml(r.action)}</strong><br/><span class="section-note">${escapeHtml(r.rationale)}</span></li>`).join("")}</ul>`;
  if (!items.some((r) => r.horizon)) { return `<div class="recommendation-group">${render(items)}</div>`; }
  return groups.map(([key, label]) => {
    const list = items.filter((r) => (r.horizon || "immediate") === key);
    return list.length ? `<div class="recommendation-group"><div class="finding-group-title">${label}</div>${render(list)}</div>` : "";
  }).join("");
}
function timelineHtml(items) {
  if (!items.length) { return ""; }
  return `<ul class="timeline-list">${items.map((t) => `<li><span class="timeline-stamp">${escapeHtml(t.timestamp || "undated")}</span>${escapeHtml(t.event)}<div class="section-note">${escapeHtml(t.significance)}</div></li>`).join("")}</ul>`;
}
function sourcesHtml(items) {
  if (!items.length) { return ""; }
  return `<table class="kv-table"><tbody>${items.map((s) => `<tr><th>source ${escapeHtml(s.source_number)}</th><td>${escapeHtml(s.summary)}<div class="source-ref">${escapeHtml(s.source_ref)}</div></td></tr>`).join("")}</tbody></table>`;
}
function sourcePostsHtml(items) {
  if (!items.length) { return ""; }
  return `<div class="child-source-post-list">${items.map((p, index) => `<div class="child-source-post-row"><div class="child-source-post-url-cell"><div class="child-source-post-meta"><span class="child-source-post-index">[${index + 1}]</span><span class="child-source-post-url-text">${escapeHtml(PLATFORM_LABELS[p.platform] || p.platform)} · ${escapeHtml(formatDate(p.timestamp))}</span></div><div class="source-ref">${escapeHtml(p.source_ref)}</div></div><div class="child-source-post-body-cell">${escapeHtml(p.text || "")}</div></div>`).join("")}</div>`;
}
function renderDetail() {
  const report = selectedReport();
  if (!report) { renderEmptyDetail("No report", "No report matches the current filters."); return; }
  const rt = report.report_text || {};
  const severity = report.severity || "UNKNOWN";
  const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.UNKNOWN;
  els.detailPanel.innerHTML = `
    <div class="exec-card">
      <div class="exec-header" style="border-bottom-color:${escapeHtml(color)}">
        <div class="exec-header-main">
          <div class="exec-badges"><span class="sev-badge" style="background:${escapeHtml(color)}">${escapeHtml(severity)}</span><span class="type-badge">${escapeHtml(report.paper_category || "Uncategorized")}</span></div>
          <div class="exec-title">${escapeHtml(rt.cluster_name || "Untitled Report")}</div>
          <div class="exec-meta">
            ${metaChip("ID", report.report_id, "platform")}
            ${metaChip("Platforms", listText(platformLabels(report.platforms)), "platform")}
            ${metaChip("Actor", rt.threat_actor || "Unknown", "ttp")}
            ${metaChip("IoCs", formatNumber((report.iocs?.items || []).length), "post")}
            ${metaChip("Posts", formatNumber(report.source_posts?.count), "post")}
            ${metaChip("Created", formatDate(report.created_at), "time")}
          </div>
        </div>
      </div>
      <div class="exec-body">
        <p class="exec-summary">${escapeHtml(rt.summary || "")}</p>
        <div class="report-screenshot-band">
          <div class="snapshot-column snapshot-findings"><div class="snapshot-title">Key Findings</div>${findingsHtml(rt.key_findings || [])}</div>
          <div class="snapshot-column snapshot-evidence"><div class="snapshot-evidence-grid">
            ${snapshotSectionHtml("CVEs", chipsHtml(rt.cves || []))}
            ${snapshotSectionHtml("MITRE ATT&CK TTPs", chipsHtml(rt.ttps || []))}
          </div></div>
        </div>
      </div>
    </div>
    ${sectionHtml("IoCs", iocsHtml(report.iocs?.items || []))}
    ${sectionHtml("Recommendations", recommendationsHtml(rt.recommendations || []))}
    ${sectionHtml("Attack Timeline", timelineHtml(rt.timeline || []))}
    ${sectionHtml("Sources", sourcesHtml(rt.sources || []))}
    ${sectionHtml("Source Posts", sourcePostsHtml(report.source_posts?.items || []))}`;
}
function renderEmptyDetail(title, message) {
  els.detailPanel.innerHTML = `<div class="section-card empty-state"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span></div>`;
}
els.prevPageButton.addEventListener("click", () => selectNextReport(-1));
els.nextPageButton.addEventListener("click", () => selectNextReport(1));
els.threatCategoryFilter.addEventListener("change", applyFilters);
els.threatSeverityFilter.addEventListener("change", applyFilters);
els.platformFilter.addEventListener("change", applyFilters);
els.reportIdInput.addEventListener("input", applyFilters);
document.addEventListener("keydown", (e) => { if (e.target.tagName !== "INPUT") { if (e.key === "ArrowRight") { selectNextReport(1); } if (e.key === "ArrowLeft") { selectNextReport(-1); } } });
applyPayload(window.TIBLENDER_DATASET);
