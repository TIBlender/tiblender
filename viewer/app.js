const SEVERITY_COLORS = {
  CRITICAL: "#7b241c",
  HIGH: "#e74c3c",
  MEDIUM: "#e67e22",
  LOW: "#27ae60",
  UNKNOWN: "#6b7a8d",
};

const state = {
  payload: null,
  allReports: [],
  reports: [],
  selectedIndex: 0,
};

const els = {
  metricReports: document.getElementById("metricReports"),
  metricCreated: document.getElementById("metricCreated"),
  pageSummary: document.getElementById("pageSummary"),
  detailPanel: document.getElementById("detailPanel"),
  prevPageButton: document.getElementById("prevPageButton"),
  nextPageButton: document.getElementById("nextPageButton"),
  threatCategoryFilter: document.getElementById("threatCategoryFilter"),
  threatSeverityFilter: document.getElementById("threatSeverityFilter"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function formatDate(value) {
  if (!value) {
    return "No date";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function listText(values, fallback = "Unspecified") {
  const items = Array.isArray(values) ? values.filter(Boolean) : [];
  return items.length ? items.join(", ") : fallback;
}

const SEVERITY_RANK = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

function severityRank(report) {
  return report.severity in SEVERITY_RANK ? SEVERITY_RANK[report.severity] : 4;
}

function sortReports(reports) {
  reports.sort((a, b) => {
    const rankA = Number(a.display_rank || 0);
    const rankB = Number(b.display_rank || 0);
    if (rankA || rankB) {
      return rankA - rankB;
    }
    return severityRank(a) - severityRank(b) || Number(a.report_index || 0) - Number(b.report_index || 0);
  });
}

function applyPayload(payload) {
  if (!payload || !Array.isArray(payload.reports)) {
    renderEmptyDetail("Dataset unavailable", "reports_viewer_data.js did not expose a reports array.");
    return;
  }
  state.payload = payload;
  state.allReports = [...payload.reports];
  sortReports(state.allReports);
  populateFilters();
  applyFilters();
}

function populateFilters() {
  setSelectOptions(
    els.threatCategoryFilter,
    uniqueValues(state.allReports.map((report) => report.category)),
    "All categories",
  );
  setSelectOptions(
    els.threatSeverityFilter,
    uniqueValues(state.allReports.map((report) => report.severity)),
    "All severities",
  );
}

function uniqueValues(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
}

function setSelectOptions(select, values, allLabel) {
  const currentValue = select.value;
  select.innerHTML = [`<option value="">${escapeHtml(allLabel)}</option>`]
    .concat(values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`))
    .join("");
  select.value = values.includes(currentValue) ? currentValue : "";
}

function applyFilters() {
  const category = els.threatCategoryFilter.value;
  const severity = els.threatSeverityFilter.value;
  state.reports = state.allReports.filter((report) => {
    const categoryMatches = !category || report.category === category;
    const severityMatches = !severity || report.severity === severity;
    return categoryMatches && severityMatches;
  });
  state.selectedIndex = 0;
  renderSummary();
  renderControls();
  renderDetail();
}

function selectedReport() {
  return state.reports[state.selectedIndex] || null;
}

function renderSummary() {
  const report = selectedReport();
  els.metricReports.textContent = `${formatNumber(state.reports.length)} / ${formatNumber(state.allReports.length)}`;
  els.metricCreated.textContent = formatDate(report?.created_at);
}

function renderControls() {
  const total = state.reports.length;
  state.selectedIndex = Math.min(Math.max(0, state.selectedIndex), Math.max(0, total - 1));
  els.pageSummary.textContent = total ? `${formatNumber(state.selectedIndex + 1)} of ${formatNumber(total)}` : "0 reports";
  els.prevPageButton.disabled = state.selectedIndex <= 0;
  els.nextPageButton.disabled = state.selectedIndex >= total - 1 || total === 0;
}

function selectNextReport(offset) {
  const total = state.reports.length;
  if (!total) {
    return;
  }
  state.selectedIndex = Math.min(Math.max(0, state.selectedIndex + offset), total - 1);
  renderSummary();
  renderControls();
  renderDetail();
}

function metaChip(label, value, modifier) {
  return `
    <span class="meta-chip metric-meta-chip metric-meta-chip--${escapeHtml(modifier)}">
      <span class="meta-chip-label">${escapeHtml(label)}</span>
      <span class="meta-chip-value">${escapeHtml(value)}</span>
    </span>
  `;
}

function renderDetail() {
  const report = selectedReport();
  if (!report) {
    renderEmptyDetail("No report", "No report records are bundled.");
    return;
  }

  const severity = report.severity || "UNKNOWN";
  const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.UNKNOWN;
  els.detailPanel.innerHTML = `
    <div class="exec-card">
      <div class="exec-header" style="border-bottom-color:${escapeHtml(color)}">
        <div class="exec-header-main">
          <div class="exec-badges">
            <span class="sev-badge" style="background:${escapeHtml(color)}">${escapeHtml(severity)}</span>
            <span class="type-badge">${escapeHtml(report.category || "Uncategorized")}</span>
          </div>
          <div class="exec-title">${escapeHtml(report.title || "Untitled Report")}</div>
          <div class="exec-meta">
            ${metaChip("PF", listText(report.platforms, "Unspecified"), "platform")}
            ${metaChip("TTP", formatNumber((report.ttp_ids || []).length), "ttp")}
            ${metaChip("Posts", formatNumber(report.source_posts?.count), "post")}
            ${metaChip("Created", formatDate(report.created_at), "time")}
          </div>
        </div>
      </div>
      <div class="exec-body">
        <p class="exec-summary">${escapeHtml(report.summary || "No summary is bundled for this report.")}</p>
        <div class="report-screenshot-band">
          <div class="snapshot-column snapshot-findings">
            <div class="snapshot-title">Key Findings</div>
            ${findingsHtml((report.report_text?.key_findings || []).slice(0, 3))}
          </div>
          <div class="snapshot-column snapshot-evidence">
            <div class="snapshot-evidence-grid">
              ${snapshotSectionHtml("IoCs", iocsHtml(report.iocs?.items || []))}
              ${snapshotSectionHtml("MITRE ATT&CK TTPs", ttpHtml(report.ttp_ids || []))}
            </div>
          </div>
        </div>
      </div>
    </div>
    ${sectionHtml("Recommendations", recommendationsHtml(report.report_text?.recommendations || {}))}
    ${sectionHtml("Infection Chain", infectionChainHtml(report.report_text?.infection_chain, "horizontal"))}
    ${sectionHtml("Detection Rules", detectionRulesHtml(report.report_text?.recommendations?.detection_rules || []))}
    ${sectionHtml("Source Context", sourcePostsHtml(report.source_posts?.items || []))}
  `;
}

function sectionHtml(title, body) {
  return `
    <div class="section-card">
      <div class="section-title">${escapeHtml(title)}</div>
      ${body || '<div class="empty-text">No data bundled.</div>'}
    </div>
  `;
}

function snapshotSectionHtml(title, body, modifier = "") {
  return `
    <div class="snapshot-evidence-block ${escapeHtml(modifier)}">
      <div class="snapshot-subtitle">${escapeHtml(title)}</div>
      ${body || '<div class="empty-text">No data bundled.</div>'}
    </div>
  `;
}

function paragraphHtml(value) {
  const text = String(value || "").trim();
  return text ? `<p class="section-paragraph">${escapeHtml(text)}</p>` : "";
}

function infectionChainHtml(value, layout = "vertical") {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }

  const markers = [...text.matchAll(/(?:^|\s)(\d+)[.)]\s+/g)];
  if (markers.length < 2) {
    return paragraphHtml(text);
  }

  const steps = markers
    .map((marker, index) => {
      const start = marker.index + marker[0].length;
      const end = index + 1 < markers.length ? markers[index + 1].index : text.length;
      return { number: marker[1], body: text.slice(start, end).trim() };
    })
    .filter((step) => step.body);

  if (steps.length < 2) {
    return paragraphHtml(text);
  }

  return `
    <div class="infection-chain-list ${layout === "horizontal" ? "infection-chain-list--horizontal" : ""}">
      ${steps
        .map(
          (step) => `
            <div class="infection-chain-item">
              <span class="infection-step-number">${escapeHtml(step.number)}</span>
              <p class="infection-step-text">${escapeHtml(step.body)}</p>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function findingsHtml(items) {
  if (!items.length) {
    return "";
  }
  return items
    .map(
      (item, index) => `
        <div class="finding-item">
          <span class="finding-num">${index + 1}</span>
          <p class="finding-text">${escapeHtml(item)}</p>
        </div>
      `,
    )
    .join("");
}

function iocsHtml(items) {
  if (!items.length) {
    return "";
  }
  return `
    <div class="ioc-table-wrap">
      <table class="ioc-table">
        <thead><tr><th>Type</th><th>Value</th><th>Extraction</th><th>Source platform</th></tr></thead>
        <tbody>
          ${items
            .slice(0, 80)
            .map(
              (item) => `
                <tr>
                  <td class="ioc-type">${escapeHtml(item.type || "unknown")}</td>
                  <td class="ioc-val">${escapeHtml(item.value || "")}</td>
                  <td>${escapeHtml(item.extraction_type || "")}</td>
                  <td>${escapeHtml((item.platforms || []).join(", "))}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
    ${items.length > 80 ? '<div class="section-note block-note">Showing first 80 IoCs.</div>' : ""}
  `;
}

function ttpHtml(items) {
  if (!items.length) {
    return "";
  }
  return `<div class="ttp-list">${items.map((item) => `<span class="ttp-chip">${escapeHtml(item)}</span>`).join("")}</div>`;
}

function recommendationsHtml(recommendations) {
  const groups = [
    ["Immediate Actions", recommendations.immediate_actions],
    ["Short-Term Actions", recommendations.short_term_actions],
    ["Long-Term Actions", recommendations.long_term_actions],
  ];
  return groups
    .map(([title, items]) => recommendationGroupHtml(title, items || []))
    .filter(Boolean)
    .join("");
}

function recommendationGroupHtml(title, items) {
  if (!items.length) {
    return "";
  }
  return `
    <div class="recommendation-group">
      <div class="judgement-subtitle">${escapeHtml(title)}</div>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
}

function detectionRulesHtml(rules) {
  if (!rules.length) {
    return "";
  }
  return rules
    .map(
      (rule) => `
        <details class="rule-block">
          <summary>
            ${escapeHtml(rule.rule_name || "Detection rule")}
            ${rule.rule_type ? `<span class="section-note">${escapeHtml(rule.rule_type)}</span>` : ""}
          </summary>
          ${rule.description ? `<p>${escapeHtml(rule.description)}</p>` : ""}
          ${rule.rule_content ? `<pre>${escapeHtml(rule.rule_content)}</pre>` : ""}
        </details>
      `,
    )
    .join("");
}

function sourcePostsHtml(items) {
  if (!items.length) {
    return "";
  }
  return `
    <div class="child-source-post-list">
      ${items
        .map(
          (item, index) => `
            <div class="child-source-post-row">
              <div class="child-source-post-url-cell">
                <div class="child-source-post-meta">
                  <span class="child-source-post-index">[${index + 1}]</span>
                  <span class="child-source-post-url-text">${escapeHtml(item.platform || "")} \u00b7 ${escapeHtml(formatDate(item.timestamp))}</span>
                </div>
                <div class="source-ref">${escapeHtml(item.source_ref || "")}</div>
              </div>
              <div class="child-source-post-body-cell">${escapeHtml(item.text || "")}</div>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderEmptyDetail(title, message) {
  els.detailPanel.innerHTML = `
    <div class="section-card empty-state">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(message)}</span>
    </div>
  `;
}

els.prevPageButton.addEventListener("click", () => selectNextReport(-1));
els.nextPageButton.addEventListener("click", () => selectNextReport(1));
els.threatCategoryFilter.addEventListener("change", applyFilters);
els.threatSeverityFilter.addEventListener("change", applyFilters);

applyPayload(window.TIBLENDER_ARTIFACT_DATA || window.TIBLENDER_RELEASE_DATA);
