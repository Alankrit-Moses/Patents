const DATA = window.RESULT_VIEWER_DATA;
const app = document.getElementById("app");

const routeState = {
  pattern: "All",
  model: "All",
  experiment: "All",
  setup: "All",
  score: "All",
  search: "",
};

const TARGETS = ["Q1", "Q2", "Q3-M", "Q3-T"];
const MAX_RESULTS = 120;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(2);
}

function avg(values) {
  const clean = values.filter((value) => value !== null && value !== undefined && !Number.isNaN(value));
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + Number(value), 0) / clean.length;
}

function compact(value, limit = 360) {
  const text = String(value ?? "").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trim()}...`;
}

function href(parts) {
  return `#${parts.map((part) => encodeURIComponent(part)).join("/")}`;
}

function routeParts() {
  const raw = window.location.hash.replace(/^#/, "") || "home";
  return raw.split("/").filter(Boolean).map((part) => decodeURIComponent(part));
}

function modelLabel(id) {
  return DATA.models.find((model) => model.id === id)?.label || id;
}

function setupLabel(id) {
  return DATA.setups.find((setup) => setup.id === id)?.label || id;
}

function expByLabel(label) {
  return Object.values(DATA.experiments).find((exp) => exp.id === label);
}

function expByNumber(number) {
  return DATA.experiments[String(number)] || null;
}

function expNumberFromLabel(label) {
  const found = Object.entries(DATA.experiments).find(([, exp]) => exp.id === label);
  return found ? found[0] : null;
}

function caseById(id) {
  return DATA.cases.find((item) => item.case_id === id);
}

function scoreClass(score) {
  if (score === null || score === undefined) return "";
  if (score < 2) return "score-low";
  if (score < 4) return "score-mid";
  return "score-high";
}

function allJudgements(cases = DATA.cases) {
  return cases.flatMap((item) =>
    item.judgements.map((judgement, index) => ({
      ...judgement,
      judgement_index: index,
      case_id: item.case_id,
      model: item.model,
      model_label: item.model_label,
      setup: item.setup,
      setup_label: item.setup_label,
      experiment: item.experiment,
      experiment_label: item.experiment_label,
      pattern: item.pattern,
      task_id: item.task_id,
    })),
  );
}

function filteredCases() {
  const needle = routeState.search.trim().toLowerCase();
  return DATA.cases.filter((item) => {
    if (routeState.pattern !== "All" && item.pattern !== routeState.pattern) return false;
    if (routeState.model !== "All" && item.model !== routeState.model) return false;
    if (routeState.experiment !== "All" && item.experiment_label !== routeState.experiment) return false;
    if (routeState.setup !== "All" && item.setup !== routeState.setup) return false;
    if (routeState.score !== "All") {
      const target = Number(routeState.score);
      if (!item.judgements.some((judgement) => Number(judgement.score) === target)) return false;
    }
    if (needle) {
      const haystack = [
        item.case_id,
        item.task_id,
        item.task_title,
        item.model_label,
        item.setup_label,
        item.pattern,
        item.experiment_label,
        JSON.stringify(item.parsed_output),
        item.raw_output,
      ].join(" ").toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    return true;
  });
}

function pageHeader(eyebrow, title, body, actions = "") {
  return `
    <section class="page-hero">
      <p class="eyebrow">${escapeHtml(eyebrow)}</p>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(body)}</p>
      ${actions ? `<div class="hero-actions">${actions}</div>` : ""}
    </section>
  `;
}

function statGrid(items) {
  return `
    <div class="stat-grid">
      ${items
        .map(
          ([label, value]) => `
          <article class="stat-card">
            <strong>${escapeHtml(value)}</strong>
            <span>${escapeHtml(label)}</span>
          </article>
        `,
        )
        .join("")}
    </div>
  `;
}

function codeBlock(value, className = "") {
  return `<pre class="code-block ${className}">${escapeHtml(value)}</pre>`;
}

function chip(label, kind = "") {
  return `<span class="chip ${kind}">${escapeHtml(label)}</span>`;
}

function renderHome() {
  const judgementCount = allJudgements().length;
  const meanScore = avg(DATA.cases.map((item) => item.score).filter((score) => score !== null));
  const actions = `
    <a class="button primary" href="#framework">Start with the framework</a>
    <a class="button" href="#results">Jump to results</a>
  `;
  return `
    ${pageHeader("Project overview", DATA.project.title, DATA.project.summary, actions)}
    ${statGrid([
      ["Patterns", DATA.patterns.length],
      ["Constructed tasks", Object.keys(DATA.tasks).length],
      ["Robustness samples", DATA.cases.length.toLocaleString()],
      ["Judged elements", judgementCount.toLocaleString()],
      ["Generator settings", DATA.models.length],
      ["Mean case score", fmt(meanScore)],
    ])}

    <section class="content-grid four">
      ${[
        ["Framework", "Understand D, P, M, T, E, typed queries, and the pattern cards.", "#framework"],
        ["Experiments", "Inspect Q1-Q3 and see how each query becomes concrete tasks.", "#experiments"],
        ["Pipelines", "Compare task-engineered prompts, structured prompts, and the agentic pipeline.", "#pipelines"],
        ["Results", "Browse stored samples, generated answers, scores, judge context, and traces.", "#results"],
      ]
        .map(
          ([title, body, link]) => `
          <a class="guide-card" href="${link}">
            <span>${escapeHtml(title)}</span>
            <p>${escapeHtml(body)}</p>
          </a>
        `,
        )
        .join("")}
    </section>

    <section class="panel-card">
      <div class="section-heading">
        <p class="eyebrow">What was tested</p>
        <h2>Three framework-derived queries across two pattern families</h2>
      </div>
      <div class="content-grid three">
        ${Object.entries(DATA.experiments)
          .map(
            ([number, exp]) => `
            <a class="query-tile small" href="${href(["experiment", exp.id])}">
              <strong>${escapeHtml(exp.id)}</strong>
              <span>${escapeHtml(exp.title)}</span>
              <em>${escapeHtml(exp.category)}</em>
              <p>${escapeHtml(exp.plain)}</p>
            </a>
          `,
          )
          .join("")}
      </div>
    </section>

    <section class="panel-card soft">
      <h2>How to read the viewer</h2>
      <p class="lead">${escapeHtml(DATA.project.takeaway)}</p>
    </section>
  `;
}

function renderFramework() {
  return `
    ${pageHeader(
      "Framework",
      "Typed queries over pattern and data components",
      "The framework describes pattern-mining tasks by marking which data and pattern components are available, absent, or targeted for production.",
      `<a class="button primary" href="#pattern/P1">Open Pattern P1</a><a class="button" href="#pattern/P2">Open Pattern P2</a>`,
    )}

    <section class="content-grid two">
      <article class="panel-card">
        <h2>Data components</h2>
        ${Object.entries(DATA.framework.data)
          .map(([slot, text]) => `<div class="slot-row"><strong>${escapeHtml(slot)}</strong><p>${escapeHtml(text)}</p></div>`)
          .join("")}
      </article>
      <article class="panel-card">
        <h2>Pattern components</h2>
        ${Object.entries(DATA.framework.pattern)
          .map(([slot, text]) => `<div class="slot-row"><strong>${escapeHtml(slot)}</strong><p>${escapeHtml(text)}</p></div>`)
          .join("")}
      </article>
    </section>

    <section class="panel-card">
      <div class="section-heading">
        <p class="eyebrow">Query markers</p>
        <h2>How a typed query marks component status</h2>
      </div>
      <div class="marker-grid">
        ${DATA.framework.markers
          .map(
            (item) => `
            <div class="marker-card">
              <strong>${escapeHtml(item.symbol)}</strong>
              <p>${escapeHtml(item.meaning)}</p>
            </div>
          `,
          )
          .join("")}
      </div>
      <div class="note-list">
        ${DATA.framework.notes.map((note) => `<p>${escapeHtml(note)}</p>`).join("")}
      </div>
    </section>

    <section class="panel-card">
      <div class="section-heading">
        <p class="eyebrow">Query catalog</p>
        <h2>Evaluated queries plus broader examples expressible in the framework</h2>
      </div>
      <div class="catalog">
        ${DATA.framework.query_catalog.map(renderCatalogRow).join("")}
      </div>
    </section>
  `;
}

function renderCatalogRow(item) {
  const link = item.evaluated ? href(["experiment", item.id]) : "#framework";
  return `
    <a class="catalog-row ${item.evaluated ? "is-evaluated" : ""}" href="${link}">
      <div>
        <strong>${escapeHtml(item.id)}</strong>
        <span>${escapeHtml(item.title)}</span>
      </div>
      ${codeBlock(item.query, "inline-code")}
      <p>${escapeHtml(item.description)}</p>
      <em>${escapeHtml(item.category)}${item.evaluated ? " / evaluated" : " / catalog example"}</em>
    </a>
  `;
}

function renderPattern(patternId, component = "overview", exampleId = "") {
  const pattern = DATA.patterns.find((item) => item.pattern_id === patternId) || DATA.patterns[0];
  if (!pattern) return renderNotFound("Pattern not found");
  const selectedExample = pattern.examples.find((example) => example.example_id === exampleId);
  const exampleList = pattern.examples
    .map(
      (example) => `
      <a class="example-pill ${example.example_id === exampleId ? "is-active" : ""}" href="${href(["pattern", pattern.pattern_id, "E", example.example_id])}">
        ${escapeHtml(example.example_id)}
        <span>${escapeHtml(example.report_id)}</span>
      </a>
    `,
    )
    .join("");
  const detail = selectedExample
    ? renderExampleDetail(selectedExample)
    : renderPatternComponent(pattern, component);
  return `
    ${pageHeader(
      "Pattern card",
      `${pattern.pattern_id}: ground-truth components`,
      "Pattern cards expose the same M, T, and paired E components used by the framework queries and evaluation protocol.",
      `<a class="button" href="#framework">Back to framework</a><a class="button" href="#experiments">Open experiments</a>`,
    )}
    <section class="pattern-layout">
      <aside class="pattern-nav panel-card">
        <h2>${escapeHtml(pattern.pattern_id)}</h2>
        <a href="${href(["pattern", pattern.pattern_id, "M"])}" class="${component === "M" ? "is-active" : ""}">M: mathematical definition</a>
        <a href="${href(["pattern", pattern.pattern_id, "T"])}" class="${component === "T" ? "is-active" : ""}">T: textual description</a>
        <a href="${href(["pattern", pattern.pattern_id, "E"])}" class="${component === "E" ? "is-active" : ""}">E: example pairs</a>
        <div class="example-list">${exampleList}</div>
      </aside>
      <section class="pattern-detail">${detail}</section>
    </section>
  `;
}

function renderPatternComponent(pattern, component) {
  if (component === "M") {
    return `<article class="panel-card"><p class="eyebrow">${escapeHtml(pattern.pattern_id)} / M</p><h2>Mathematical definition</h2>${textBlock(pattern.M)}</article>`;
  }
  if (component === "T") {
    return `<article class="panel-card"><p class="eyebrow">${escapeHtml(pattern.pattern_id)} / T</p><h2>Textual description</h2>${textBlock(pattern.T)}</article>`;
  }
  return `
    <article class="panel-card">
      <p class="eyebrow">${escapeHtml(pattern.pattern_id)} / E</p>
      <h2>Paired examples</h2>
      <p class="lead">Each example pair contains a report passage E.text and a compact tabular counterpart E.tab.</p>
      <div class="content-grid two example-cards">
        ${pattern.examples
          .map(
            (example) => `
            <a class="mini-card" href="${href(["pattern", pattern.pattern_id, "E", example.example_id])}">
              <strong>${escapeHtml(example.example_id)}</strong>
              <span>${escapeHtml(example.report_id)}</span>
              <p>${escapeHtml(compact(example.E_text, 180))}</p>
            </a>
          `,
          )
          .join("")}
      </div>
    </article>
  `;
}

function renderExampleDetail(example) {
  return `
    <article class="panel-card">
      <p class="eyebrow">${escapeHtml(example.pattern_id)} / ${escapeHtml(example.example_id)}</p>
      <h2>Example pair from ${escapeHtml(example.report_id)}</h2>
      <div class="content-grid two">
        <section>
          <h3>E.text</h3>
          ${textBlock(example.E_text || "(missing text example)")}
        </section>
        <section>
          <h3>E.tab</h3>
          ${codeBlock(example.E_tab || "(missing table example)")}
        </section>
      </div>
    </article>
  `;
}

function renderExperiments() {
  return `
    ${pageHeader(
      "Experiments",
      "Three evaluated framework queries",
      "Each query is shown as a typed signature. Hover a query card to see the natural-language task; click it to inspect task construction.",
    )}
    <section class="query-showcase">
      ${Object.entries(DATA.experiments)
        .map(([number, exp]) => renderQueryPoster(number, exp))
        .join("")}
    </section>
  `;
}

function renderQueryPoster(number, exp) {
  return `
    <a class="query-poster" href="${href(["experiment", exp.id])}">
      <div class="query-poster__front">
        <span>${escapeHtml(exp.id)}</span>
        <h2>${escapeHtml(exp.title)}</h2>
        ${codeBlock(exp.framework_query)}
      </div>
      <div class="query-poster__back">
        <span>${escapeHtml(exp.category)}</span>
        <p>${escapeHtml(exp.plain)}</p>
        <em>${escapeHtml(exp.construction)}</em>
      </div>
    </a>
  `;
}

function renderExperimentDetail(expLabel) {
  const exp = expByLabel(expLabel);
  const number = expNumberFromLabel(expLabel);
  if (!exp || !number) return renderNotFound("Experiment not found");
  const tasks = DATA.task_groups[number];
  return `
    ${pageHeader(
      `${exp.id} / ${exp.category}`,
      exp.title,
      exp.plain,
      `<a class="button" href="#experiments">All experiments</a><a class="button" href="#results">View results</a>`,
    )}
    <section class="content-grid two">
      <article class="panel-card">
        <h2>Typed query</h2>
        ${codeBlock(exp.framework_query)}
        <div class="chip-row">
          ${exp.known.map((item) => chip(`Known: ${item}`)).join("")}
          ${exp.target.map((item) => chip(`Target: ${item}`, "accent")).join("")}
        </div>
      </article>
      <article class="panel-card">
        <h2>Evaluation</h2>
        <p class="lead">${escapeHtml(exp.evaluation)}</p>
        <ul class="rubric">${exp.score_scale.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </article>
    </section>
    <section class="panel-card">
      <div class="section-heading">
        <p class="eyebrow">Task construction</p>
        <h2>${tasks.count} concrete tasks</h2>
      </div>
      <p class="lead">${escapeHtml(exp.construction)}</p>
      ${renderTaskConstruction(number, tasks)}
    </section>
  `;
}

function renderTaskConstruction(number, group) {
  if (number === "1") {
    const reports = group.reports;
    const patterns = group.patterns;
    return `
      <table class="task-matrix">
        <thead><tr><th>Pattern</th>${reports.map((report) => `<th>${escapeHtml(report)}</th>`).join("")}</tr></thead>
        <tbody>
          ${patterns
            .map(
              (pattern) => `
              <tr>
                <th><a href="${href(["pattern", pattern])}">${escapeHtml(pattern)}</a></th>
                ${reports
                  .map((report) => {
                    const task = group.tasks.find((item) => item.pattern === pattern && item.report === report);
                    return `<td>${task ? `<a href="${href(["task", task.task_id])}">${escapeHtml(task.task_id)}</a>` : "-"}</td>`;
                  })
                  .join("")}
              </tr>
            `,
            )
            .join("")}
        </tbody>
      </table>
    `;
  }
  if (number === "2") {
    return `
      <div class="content-grid two">
        ${group.patterns
          .map((pattern) => {
            const tasks = group.tasks.filter((task) => task.pattern === pattern);
            return `
              <article class="mini-panel">
                <h3><a href="${href(["pattern", pattern])}">${escapeHtml(pattern)}</a></h3>
                <div class="task-list">
                  ${tasks
                    .map(
                      (task) => `
                      <a href="${href(["task", task.task_id])}">
                        <strong>${escapeHtml(task.source_example_id)}</strong>
                        <span>${escapeHtml(task.report || "")}</span>
                      </a>
                    `,
                    )
                    .join("")}
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  }
  return `
    <div class="content-grid two">
      ${group.tasks
        .map(
          (task) => `
          <a class="fold-card" href="${href(["task", task.task_id])}">
            <strong>${escapeHtml(task.pattern)} fold ${escapeHtml(task.fold)}</strong>
            <p>Input pairs: ${escapeHtml(task.induction_examples.join(", "))}</p>
            <p>Held out: ${escapeHtml(task.heldout_examples.join(", "))}</p>
          </a>
        `,
        )
        .join("")}
    </div>
  `;
}

function renderTaskDetail(taskId) {
  const task = DATA.tasks[taskId];
  if (!task) return renderNotFound("Task not found");
  const exp = expByNumber(task.experiment);
  return `
    ${pageHeader(
      "Constructed task",
      task.title || task.task_id,
      `${exp.id}: ${exp.title}`,
      `<a class="button" href="${href(["experiment", exp.id])}">Back to ${exp.id}</a><a class="button" href="#results">Find samples</a>`,
    )}
    <section class="content-grid two">
      <article class="panel-card">
        <h2>Typed query</h2>
        ${codeBlock(task.display_framework_query || task.framework_query)}
        <div class="chip-row">
          ${chip(task.pattern_id)}
          ${task.report_id ? chip(task.report_id) : ""}
          ${task.source_example_id ? chip(task.source_example_id) : ""}
          ${task.fold ? chip(`fold ${task.fold}`) : ""}
        </div>
      </article>
      <article class="panel-card">
        <h2>Task evidence</h2>
        ${renderTaskEvidence(task, task.experiment)}
      </article>
    </section>
  `;
}

function renderPipelines() {
  return `
    ${pageHeader(
      "Pipelines",
      "Three ways of executing the same framework-derived queries",
      "The pipelines test manually adapted prompts, reusable structured prompts, and a framework-inspired agentic workflow.",
    )}
    <section class="setup-showcase">
      ${DATA.setups.map((setup) => renderSetupPoster(setup)).join("")}
    </section>
    <section class="panel-card">
      <div class="section-heading">
        <p class="eyebrow">Pipeline C</p>
        <h2>Agentic flow derived from framework components</h2>
      </div>
      ${renderAgentDiagram()}
    </section>
  `;
}

function renderSetupPoster(setup) {
  const pipeline = DATA.pipelines[setup.id];
  return `
    <a class="setup-poster" href="${href(["pipeline", setup.id])}">
      <div>
        <span>${escapeHtml(setup.id)}</span>
        <h2>${escapeHtml(pipeline.title)}</h2>
      </div>
      <p>${escapeHtml(pipeline.purpose)}</p>
    </a>
  `;
}

function renderPipelineDetail(setupId) {
  const pipeline = DATA.pipelines[setupId];
  if (!pipeline) return renderNotFound("Pipeline not found");
  return `
    ${pageHeader(
      setupId,
      pipeline.title,
      pipeline.purpose,
      `<a class="button" href="#pipelines">All pipelines</a><a class="button" href="#results">Browse samples</a>`,
    )}
    ${setupId === "C" ? `<section class="panel-card">${renderAgentDiagram()}</section>` : ""}
    <section class="panel-card">
      <div class="section-heading">
        <p class="eyebrow">Prompt skeletons by query</p>
        <h2>What the model sees structurally</h2>
      </div>
      <div class="prompt-stack">
        ${Object.entries(DATA.experiments)
          .map(([number, exp]) => {
            const skeleton = DATA.prompt_skeletons[setupId]?.[number] || "(prompt skeleton unavailable)";
            return `
              <details class="prompt-card" open>
                <summary>
                  <strong><a href="${href(["experiment", exp.id])}">${escapeHtml(exp.id)}: ${escapeHtml(exp.title)}</a></strong>
                  <span>${escapeHtml(exp.category)}</span>
                </summary>
                ${codeBlock(skeleton)}
              </details>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function renderAgentDiagram() {
  return `
    <div class="agent-map" aria-label="Pipeline C agent connection map">
      <div class="agent-legend">
        <span><i class="legend-invoke"></i>invoke</span>
        <span><i class="legend-write"></i>write</span>
        <span><i class="legend-read"></i>read</span>
      </div>

      <a class="map-node map-framework" href="${href(["agent", "FrameworkAgent"])}" title="${escapeHtml(DATA.agents.FrameworkAgent.role)}">
        <strong>FrameworkAgent</strong>
        <span>orchestrator</span>
      </a>

      <a class="map-node map-m" href="${href(["agent", "M-Inducer"])}" title="${escapeHtml(DATA.agents["M-Inducer"].role)}">
        <strong>M-Inducer</strong>
        <span>M / M_hat</span>
      </a>

      <a class="map-node map-t" href="${href(["agent", "T-Inducer"])}" title="${escapeHtml(DATA.agents["T-Inducer"].role)}">
        <strong>T-Inducer</strong>
        <span>T / T_hat</span>
      </a>

      <a class="map-node map-e" href="${href(["agent", "E.text-Finder"])}" title="${escapeHtml(DATA.agents["E.text-Finder"].role)}">
        <strong>E.text-Finder</strong>
        <span>candidate E.text</span>
      </a>

      <a class="map-node map-r" href="${href(["agent", "E.text-Reducer"])}" title="${escapeHtml(DATA.agents["E.text-Reducer"].role)}">
        <strong>E.text-Reducer</strong>
        <span>candidate selector</span>
      </a>

      <div class="map-node map-blackboard">
        <strong>Shared blackboard</strong>
        <span>working memory + trace</span>
      </div>

      <svg class="map-arrows" viewBox="0 0 1000 520" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <marker id="map-arrow-invoke" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#176d63"></path>
          </marker>
          <marker id="map-arrow-write" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#be643d"></path>
          </marker>
          <marker id="map-arrow-read" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 Z" fill="#6f4a22"></path>
          </marker>
        </defs>

        <path class="map-line invoke" d="M285 240 H360 V95 H535"></path>
        <path class="map-line invoke" d="M285 260 H535"></path>
        <path class="map-line invoke" d="M285 280 H360 V425 H535"></path>
        <path class="map-line invoke" d="M630 425 H680"></path>

        <path class="map-line write" d="M630 95 H750 V185 H780"></path>
        <path class="map-line write" d="M630 260 H780"></path>
        <path class="map-line write" d="M860 425 V267"></path>

        <path class="map-line read" d="M780 270 H700 V485 H285 V275"></path>
      </svg>
    </div>
    <p class="lead">
      FrameworkAgent invokes the three specialists. E.text-Finder invokes the reducer internally.
      Specialists write to the shared blackboard, and FrameworkAgent reads blackboard state before deciding the next action.
    </p>
  `;
}

function renderAgentDetail(agentId) {
  const agent = DATA.agents[agentId];
  if (!agent) return renderNotFound("Agent not found");
  return `
    ${pageHeader(
      "Agent page",
      `${agent.title}: ${agentId}`,
      agent.role,
      `<a class="button" href="#pipelines">Back to pipelines</a><a class="button" href="${href(["pipeline", "C"])}">Open Pipeline C</a>`,
    )}
    <section class="content-grid two">
      <article class="panel-card">
        <h2>Read/write contract</h2>
        <h3>Reads</h3>
        <div class="chip-row">${agent.reads.map((item) => chip(item)).join("")}</div>
        <h3>Writes</h3>
        <div class="chip-row">${agent.writes.map((item) => chip(item, "accent")).join("")}</div>
      </article>
      <article class="panel-card">
        <h2>Prompt skeleton</h2>
        ${codeBlock(agent.prompt_skeleton)}
      </article>
    </section>
  `;
}

function renderResults() {
  const cases = filteredCases();
  const visible = cases.slice(0, MAX_RESULTS);
  return `
    ${pageHeader(
      "Results",
      "Browse stored robustness samples",
      "The results browser mirrors the artifact directory structure while also letting you filter by pattern, model, query, pipeline, score, and text search.",
      `<a class="button" href="#stats">Open score charts</a><a class="button" href="#matrix">Open score matrix</a>`,
    )}
    <section class="results-layout">
      <aside class="panel-card result-controls">
        <h2>Filters</h2>
        ${renderFilters()}
      </aside>
      <section class="result-list">
        <div class="section-heading">
          <p class="eyebrow">Matching samples</p>
          <h2>${visible.length.toLocaleString()} of ${cases.length.toLocaleString()} shown</h2>
          <p class="lead">Click any sample card to open the full model output, task evidence, judge prompt skeleton, and stored score record.</p>
        </div>
        ${visible.length ? visible.map(renderResultCard).join("") : `<div class="empty">No samples match the current filters.</div>`}
      </section>
    </section>
  `;
}

function renderFilters() {
  return `
    <label class="filter-label">Pattern
      <select data-filter="pattern">${optionList(["All", ...DATA.patterns.map((pattern) => pattern.pattern_id)], routeState.pattern)}</select>
    </label>
    <label class="filter-label">Model
      <select data-filter="model">${optionList([{ value: "All", label: "All models" }, ...DATA.models.map((model) => ({ value: model.id, label: model.label }))], routeState.model)}</select>
    </label>
    <label class="filter-label">Query
      <select data-filter="experiment">${optionList([{ value: "All", label: "All queries" }, ...Object.values(DATA.experiments).map((exp) => ({ value: exp.id, label: `${exp.id}: ${exp.title}` }))], routeState.experiment)}</select>
    </label>
    <label class="filter-label">Pipeline
      <select data-filter="setup">${optionList([{ value: "All", label: "All pipelines" }, ...DATA.setups.map((setup) => ({ value: setup.id, label: setup.label }))], routeState.setup)}</select>
    </label>
    <label class="filter-label">Score
      <select data-filter="score">${optionList([{ value: "All", label: "Any score" }, ...[0, 1, 2, 3, 4, 5].map((score) => ({ value: String(score), label: `Score ${score}` }))], routeState.score)}</select>
    </label>
    <label class="filter-label">Search
      <input data-filter="search" value="${escapeHtml(routeState.search)}" placeholder="task, output, model..." />
    </label>
    <button class="button slim" data-reset-filters type="button">Reset filters</button>
  `;
}

function optionList(options, selected) {
  return options
    .map((option) => {
      const value = typeof option === "string" ? option : option.value;
      const label = typeof option === "string" ? option : option.label;
      return `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(label)}</option>`;
    })
    .join("");
}

function renderResultCard(item) {
  return `
    <article class="result-card" data-open-case="${escapeHtml(item.case_id)}" tabindex="0" role="link" aria-label="Open full sample detail for ${escapeHtml(item.task_title)}">
      <div class="result-card__head">
        <div>
          <h3>${escapeHtml(item.task_title)}</h3>
          <p>${escapeHtml(item.pattern)} / <a href="${href(["experiment", item.experiment_label])}">${escapeHtml(item.experiment_label)}</a> / ${escapeHtml(item.model_label)} / ${escapeHtml(item.setup_label)} / sample ${escapeHtml(Number(item.sample_index ?? 0) + 1)}</p>
        </div>
        <strong class="score-badge ${scoreClass(item.score)}">${fmt(item.score)}</strong>
      </div>
      <div class="answer-block">${renderAnswerOnly(item)}</div>
      <div class="grade-row">
        <a class="open-sample-link" href="${href(["case", item.case_id, "0"])}">Open full sample</a>
        ${item.judgements
          .map(
            (judgement, index) => `
            <a class="grade-pill ${scoreClass(judgement.score)}" href="${href(["case", item.case_id, String(index)])}">
              ${escapeHtml(judgement.target)}: ${escapeHtml(judgement.score)}
            </a>
          `,
          )
          .join("")}
      </div>
    </article>
  `;
}

function renderAnswerOnly(item, showScores = false) {
  if (!item.judgements.length) return textBlock(JSON.stringify(item.parsed_output, null, 2));
  return item.judgements
    .map(
      (judgement) => `
      <section>
        <div class="answer-head">
          <strong>${escapeHtml(judgement.target)}</strong>
          ${showScores ? `<span class="grade-pill ${scoreClass(judgement.score)}">${escapeHtml(judgement.score ?? "n/a")}</span>` : ""}
        </div>
        ${textBlock(judgement.prediction || "(empty output)")}
      </section>
    `,
    )
    .join("");
}

function renderCaseDetail(caseId, judgementIndex = "0") {
  const item = caseById(caseId);
  if (!item) return renderNotFound("Sample not found");
  const index = Number(judgementIndex) || 0;
  const judgement = item.judgements[index] || item.judgements[0] || {};
  const task = DATA.tasks[item.task_id] || {};
  const exp = expByNumber(item.experiment);
  return `
    ${pageHeader(
      "Evaluation detail",
      `${item.model_label} / ${item.setup_label}`,
      `${item.task_title} (${item.experiment_label}, ${item.pattern}, sample ${item.sample_index})`,
      `<a class="button" href="#results">Back to results</a><a class="button" href="${href(["task", item.task_id])}">Open task</a>`,
    )}
    <section class="content-grid two">
      <article class="panel-card">
        <h2>Navigation context</h2>
        <div class="chip-row">
          ${chip(item.model_label)}
          ${chip(item.setup_label)}
          ${chip(item.pattern)}
          ${chip(item.experiment_label)}
          ${chip(`score ${judgement.score ?? "n/a"}`, "accent")}
        </div>
        <p class="lead">
          Pattern: <a href="${href(["pattern", item.pattern])}">${escapeHtml(item.pattern)}</a><br />
          Query: <a href="${href(["experiment", item.experiment_label])}">${escapeHtml(item.experiment_label)}: ${escapeHtml(exp.title)}</a><br />
          Pipeline: <a href="${href(["pipeline", item.setup])}">${escapeHtml(item.setup_label)}</a>
        </p>
      </article>
      <article class="panel-card">
        <h2>Output evaluations</h2>
        <div class="big-grade ${scoreClass(judgement.score)}">${escapeHtml(judgement.score ?? "n/a")}</div>
        <p class="lead">${escapeHtml(judgement.target || "No judgement target")}</p>
        ${
          item.judgements.length > 1
            ? `<div class="grade-row">
                ${item.judgements
                  .map(
                    (entry, entryIndex) => `
                    <a class="grade-pill ${scoreClass(entry.score)} ${entryIndex === index ? "is-selected" : ""}" href="${href(["case", item.case_id, String(entryIndex)])}">
                      ${escapeHtml(entry.target)}: ${escapeHtml(entry.score)}
                    </a>
                  `,
                  )
                  .join("")}
              </div>`
            : ""
        }
      </article>
    </section>

    <section class="content-grid two">
      <article class="panel-card">
        <h2>Model output</h2>
        ${renderAnswerOnly(item, true)}
      </article>
      <article class="panel-card">
        <h2>Ground truth / task evidence</h2>
        ${renderTaskEvidence(task, item.experiment)}
      </article>
    </section>

    <section class="content-grid two">
      <article class="panel-card">
        <h2>Judge prompt skeleton</h2>
        <p class="lead">The judge was blind to ${escapeHtml(DATA.judge.blind_fields.join(", "))}.</p>
        ${codeBlock(DATA.judge.prompts[item.experiment] || "")}
      </article>
      <article class="panel-card">
        <h2>Stored judge output</h2>
        <p class="lead">${escapeHtml(DATA.judge.stored_output_note)}</p>
        ${codeBlock(JSON.stringify(item.evaluation_record, null, 2))}
      </article>
    </section>

    ${item.setup === "C" ? `<section class="panel-card"><h2>Agent trace</h2>${renderTrace(item)}</section>` : ""}
  `;
}

function renderTaskEvidence(task, experiment) {
  if (!task) return `<p class="lead">No task record found.</p>`;
  if (String(experiment) === "1") {
    const references = referenceExamplesForTask(task);
    return `
      <p class="lead">Inputs: report <strong>${escapeHtml(task.report_id || "")}</strong> and canonical M.</p>
      <h3>Canonical M</h3>
      ${textBlock(task.M || "")}
      <h3>Curated reference examples from this report</h3>
      ${
        references.length
          ? references
              .map(
                (example) => `
                <details class="nested-detail">
                  <summary>${escapeHtml(example.example_id)} / ${escapeHtml(example.report_id)}</summary>
                  <h4>E.text</h4>
                  ${textBlock(example.E_text || "")}
                  <h4>E.tab</h4>
                  ${codeBlock(example.E_tab || "")}
                </details>
              `,
              )
              .join("")
          : `<p class="lead">No curated same-report examples are available for this Q1 task. Q1 is judged against M rather than one exhaustive gold passage.</p>`
      }
    `;
  }
  if (String(experiment) === "2") {
    return `
      <p class="lead">Inputs: report <strong>${escapeHtml(task.report_id || "")}</strong> and compact E.tab. The paired E.text below is held-out diagnostic reference evidence for evaluation.</p>
      <h3>Provided E.tab</h3>
      ${codeBlock(task.E_tab || "")}
      <h3>Reference E.text</h3>
      ${textBlock(task.gold_E_text || "")}
    `;
  }
  const examples = (task.induction_examples || [])
    .map(
      (example) => `
      <details class="nested-detail">
        <summary>${escapeHtml(example.example_id)} / ${escapeHtml(example.report_id)}</summary>
        <h4>E.text</h4>
        ${textBlock(example.E_text || "")}
        <h4>E.tab</h4>
        ${codeBlock(example.E_tab || "")}
      </details>
    `,
    )
    .join("");
  return `
    <h3>Canonical M</h3>
    ${textBlock(task.M || "")}
    <h3>Canonical T</h3>
    ${textBlock(task.T || "")}
    <h3>Input example pairs</h3>
    ${examples}
  `;
}

function referenceExamplesForTask(task) {
  const pattern = DATA.patterns.find((item) => item.pattern_id === task.pattern_id);
  if (!pattern) return [];
  return pattern.examples.filter((example) => example.report_id === task.report_id);
}

function renderTrace(item) {
  if (!item.agent_trace.length) return `<p class="lead">No agent trace stored for this pipeline.</p>`;
  return `
    <div class="trace-stack">
      ${item.agent_trace
        .map((step) => {
          const payload = step.decision || step.agent_output || step.output || {};
          return `
            <details class="trace-step">
              <summary>Step ${escapeHtml(step.index)}: ${escapeHtml(step.action || payload.agent || payload.action || "trace")}</summary>
              ${step.decision?.reason ? `<p class="lead">${escapeHtml(step.decision.reason)}</p>` : ""}
              ${codeBlock(JSON.stringify(payload, null, 2))}
            </details>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderStats() {
  const cases = filteredCases();
  const judgements = allJudgements(cases);
  return `
    ${pageHeader(
      "Score statistics",
      "Score distributions by model, query, pattern, and pipeline",
      "These charts count judged elements. Q1 contributes up to three excerpt scores per sample, while Q3 contributes separate M and T scores.",
      `<a class="button" href="#results">Back to results</a><a class="button" href="#matrix">Open matrix</a>`,
    )}
    <section class="panel-card">
      ${renderFilters()}
    </section>
    ${renderScoreStats(judgements)}
  `;
}

function renderScoreStats(judgements) {
  if (!judgements.length) return `<div class="empty">No judged elements match the current filters.</div>`;
  const counts = scoreCounts(judgements);
  return `
    ${statGrid([
      ["Judged elements", judgements.length.toLocaleString()],
      ["Mean score", fmt(avg(judgements.map((item) => item.score)))],
      ["Score 4-5", `${Math.round((((counts[4] || 0) + (counts[5] || 0)) / judgements.length) * 100)}%`],
      ["Score 0", counts[0] || 0],
    ])}
    <section class="content-grid two">
      ${renderHistogram(judgements)}
      ${renderStacked("By query", Object.values(DATA.experiments).map((exp) => [exp.id, judgements.filter((item) => item.experiment_label === exp.id)]))}
    </section>
    <section class="content-grid two">
      ${renderStacked("By model", DATA.models.map((model) => [model.label, judgements.filter((item) => item.model === model.id)]))}
      ${renderStacked("By pipeline", DATA.setups.map((setup) => [setup.label, judgements.filter((item) => item.setup === setup.id)]))}
    </section>
  `;
}

function scoreCounts(items) {
  const counts = Object.fromEntries([0, 1, 2, 3, 4, 5].map((score) => [score, 0]));
  for (const item of items) counts[Number(item.score)] = (counts[Number(item.score)] || 0) + 1;
  return counts;
}

function renderHistogram(items) {
  const counts = scoreCounts(items);
  const max = Math.max(1, ...Object.values(counts));
  return `
    <article class="panel-card">
      <h2>Score buckets</h2>
      <div class="histogram">
        ${[0, 1, 2, 3, 4, 5]
          .map((score) => {
            const count = counts[score] || 0;
            const height = Math.max(8, (count / max) * 180);
            return `
              <button class="hist-bar" data-score-select="${score}" style="--h:${height}px">
                <span class="score-fill score-${score}"></span>
                <strong>${count}</strong>
                <em>${score}</em>
              </button>
            `;
          })
          .join("")}
      </div>
    </article>
  `;
}

function renderStacked(title, groups) {
  return `
    <article class="panel-card">
      <h2>${escapeHtml(title)}</h2>
      <div class="stack-list">
        ${groups
          .map(([label, items]) => {
            const counts = scoreCounts(items);
            const total = items.length || 1;
            return `
              <div class="stack-row">
                <div class="stack-label"><strong>${escapeHtml(label)}</strong><span>${items.length} elements / mean ${fmt(avg(items.map((item) => item.score)))}</span></div>
                <div class="stack-bar">
                  ${[0, 1, 2, 3, 4, 5]
                    .map((score) => {
                      const width = ((counts[score] || 0) / total) * 100;
                      return width ? `<span class="score-${score}" style="width:${width}%"></span>` : "";
                    })
                    .join("")}
                </div>
              </div>
            `;
          })
          .join("")}
      </div>
    </article>
  `;
}

function renderMatrix() {
  return `
    ${pageHeader(
      "Score matrix",
      "Robustness summary by pattern, model, pipeline, and query target",
      "Bold cells mark the best pipeline for each pattern-model-query target cell.",
      `<a class="button" href="#results">Back to results</a>`,
    )}
    ${["P1", "P2", "Avg"].map(renderMatrixPattern).join("")}
  `;
}

function renderMatrixPattern(pattern) {
  const matrix = DATA.summary_matrix[pattern];
  return `
    <section class="panel-card">
      <h2>${pattern === "Avg" ? "P1/P2 macro-average" : pattern}</h2>
      <div class="matrix-grid">
        ${DATA.models.map((model) => renderMatrixModel(pattern, model, matrix)).join("")}
      </div>
    </section>
  `;
}

function renderMatrixModel(pattern, model, matrix) {
  const winners = {};
  for (const target of TARGETS) {
    const values = DATA.setups.map((setup) => matrix[model.id]?.[setup.id]?.[target]).filter((value) => value !== null && value !== undefined);
    winners[target] = values.length ? Math.max(...values) : null;
  }
  return `
    <article class="matrix-card">
      <h3>${escapeHtml(model.label)}</h3>
      <table>
        <thead><tr><th>Pipeline</th>${TARGETS.map((target) => `<th>${target}</th>`).join("")}</tr></thead>
        <tbody>
          ${DATA.setups
            .map((setup) => {
              const cells = TARGETS
                .map((target) => {
                  const value = matrix[model.id]?.[setup.id]?.[target];
                  const best = value !== null && value !== undefined && value === winners[target];
                  return `<td class="${best ? "is-best" : ""}">${fmt(value)}</td>`;
                })
                .join("");
              return `<tr><th>${escapeHtml(setup.id)}</th>${cells}</tr>`;
            })
            .join("")}
        </tbody>
      </table>
    </article>
  `;
}

function textBlock(text) {
  return `<div class="text-block">${escapeHtml(text)}</div>`;
}

function renderNotFound(message) {
  return pageHeader("Not found", message, "The requested viewer page could not be found.", `<a class="button" href="#home">Go home</a>`);
}

function renderPage() {
  const parts = routeParts();
  const [page, a, b, c] = parts;
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const section = page === "pattern" ? "framework" : page === "experiment" || page === "task" ? "experiments" : page === "pipeline" || page === "agent" ? "pipelines" : page;
    link.classList.toggle("is-active", link.dataset.nav === section);
  });

  if (page === "home") app.innerHTML = renderHome();
  else if (page === "framework") app.innerHTML = renderFramework();
  else if (page === "pattern") app.innerHTML = renderPattern(a || "P1", b || "overview", c || "");
  else if (page === "experiments") app.innerHTML = renderExperiments();
  else if (page === "experiment") app.innerHTML = renderExperimentDetail(a || "Q1");
  else if (page === "task") app.innerHTML = renderTaskDetail(a || "");
  else if (page === "pipelines") app.innerHTML = renderPipelines();
  else if (page === "pipeline") app.innerHTML = renderPipelineDetail(a || "A");
  else if (page === "agent") app.innerHTML = renderAgentDetail(a || "FrameworkAgent");
  else if (page === "results") app.innerHTML = renderResults();
  else if (page === "stats") app.innerHTML = renderStats();
  else if (page === "matrix") app.innerHTML = renderMatrix();
  else if (page === "case") app.innerHTML = renderCaseDetail(a || "", b || "0");
  else app.innerHTML = renderNotFound("Unknown route");

  bindDynamicControls();
}

function bindDynamicControls() {
  app.querySelectorAll("[data-filter]").forEach((control) => {
    const key = control.dataset.filter;
    const eventName = key === "search" ? "input" : "change";
    control.addEventListener(eventName, (event) => {
      routeState[key] = event.target.value;
      renderPage();
    });
  });
  app.querySelectorAll("[data-reset-filters]").forEach((button) => {
    button.addEventListener("click", () => {
      Object.assign(routeState, {
        pattern: "All",
        model: "All",
        experiment: "All",
        setup: "All",
        score: "All",
        search: "",
      });
      renderPage();
    });
  });
  app.querySelectorAll("[data-score-select]").forEach((button) => {
    button.addEventListener("click", () => {
      routeState.score = String(button.dataset.scoreSelect);
      window.location.hash = "results";
      renderPage();
    });
  });
  app.querySelectorAll("[data-open-case]").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.closest("a, button, input, select, summary")) return;
      window.location.hash = href(["case", card.dataset.openCase, "0"]).slice(1);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      window.location.hash = href(["case", card.dataset.openCase, "0"]).slice(1);
    });
  });
}

window.addEventListener("hashchange", renderPage);
renderPage();
