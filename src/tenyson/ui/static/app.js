"use strict";

(function () {
  const chartColors = ["#1f6f5f", "#c56d3d", "#2e6aa2", "#b1831d", "#855c9e"];
  const metricPriority = {
    sft: ["train/loss", "eval/loss", "train/grad_norm", "train/learning_rate"],
    rl: [
      "reward",
      "kl",
      "loss",
      "train/loss",
      "train/reward",
      "profiling/Time taken: UnslothGRPOTrainer.transformers.generate",
      "profiling/Time taken: UnslothGRPOTrainer.reward_wordle_strict",
    ],
    eval: [
      "constraint_accuracy",
      "dict_accuracy",
      "format_accuracy",
      "total_samples",
      "avg_reward",
      "min_reward",
      "max_reward",
    ],
  };

  const state = {
    config: null,
    experiments: [],
    selectedExperimentId: null,
    experimentSnapshot: null,
    selectedRunKey: null,
    selectedRunSummary: null,
    runDetail: null,
    showFailuresOnly: false,
    rewardFilterMode: "any",
    refreshTimer: null,
    refreshInFlight: false,
    experimentRequestToken: 0,
    runRequestToken: 0,
    initialRoute: readRoute(),
  };

  const elements = {
    experimentSelect: document.getElementById("experiment-select"),
    refreshButton: document.getElementById("refresh-button"),
    projectSummary: document.getElementById("project-summary"),
    runCountBadge: document.getElementById("run-count-badge"),
    runList: document.getElementById("run-list"),
    heroCards: document.getElementById("hero-cards"),
    detailTitle: document.getElementById("detail-title"),
    detailStatus: document.getElementById("detail-status"),
    detailMeta: document.getElementById("detail-meta"),
    detailMetrics: document.getElementById("detail-metrics"),
    historySummary: document.getElementById("history-summary"),
    historyCharts: document.getElementById("history-charts"),
    evalSummary: document.getElementById("eval-summary"),
    evalTableBody: document.getElementById("eval-table-body"),
    rawPayload: document.getElementById("raw-payload"),
    showAllRows: document.getElementById("show-all-rows"),
    showFailedRows: document.getElementById("show-failed-rows"),
    rewardFilterButtons: Array.from(
      document.querySelectorAll("[data-reward-filter]")
    ),
    liveIndicator: document.getElementById("live-indicator"),
  };

  bindEvents();
  renderBootState();
  void refreshDashboard({ resetSelection: true });

  function bindEvents() {
    elements.experimentSelect.addEventListener("change", () => {
      state.selectedRunKey = null;
      state.selectedRunSummary = null;
      state.runDetail = null;
      void loadExperimentSnapshot(elements.experimentSelect.value, {
        preferredRunKey: null,
        preserveCurrentRun: false,
      });
    });

    elements.refreshButton.addEventListener("click", () => {
      void refreshDashboard();
    });

    elements.showAllRows.addEventListener("click", () => {
      state.showFailuresOnly = false;
      renderEvalSamples();
    });

    elements.showFailedRows.addEventListener("click", () => {
      state.showFailuresOnly = true;
      renderEvalSamples();
    });

    elements.rewardFilterButtons.forEach((button) => {
      button.addEventListener("click", () => {
        if (button.disabled) {
          return;
        }
        state.rewardFilterMode = String(button.dataset.rewardFilter || "any");
        renderEvalSamples();
      });
    });
  }

  async function refreshDashboard(options) {
    const settings = options || {};
    if (state.refreshInFlight) {
      return;
    }

    state.refreshInFlight = true;
    setRefreshBusy(true);

    try {
      const firstLoad = !state.config;
      const responses = firstLoad
        ? await Promise.all([fetchJson("/api/config"), fetchJson("/api/experiments")])
        : [state.config, await fetchJson("/api/experiments")];

      const config = responses[0];
      const experimentsPayload = responses[1];
      state.config = config;
      state.experiments = Array.isArray(experimentsPayload.experiments)
        ? experimentsPayload.experiments
        : [];

      scheduleAutoRefresh();

      const nextExperimentId = pickExperimentId({
        resetSelection: Boolean(settings.resetSelection),
        defaultExperimentId:
          experimentsPayload.default_experiment_id || config.default_experiment_id,
      });
      const previousExperimentId = state.selectedExperimentId;

      state.selectedExperimentId = nextExperimentId;
      renderExperimentSelect();

      if (!nextExperimentId) {
        state.selectedExperimentId = null;
        state.experimentSnapshot = null;
        state.selectedRunKey = null;
        state.selectedRunSummary = null;
        state.runDetail = null;
        renderNoExperiments();
        writeRoute();
        return;
      }

      await loadExperimentSnapshot(nextExperimentId, {
        preferredRunKey:
          settings.preferredRunKey || routeRunKeyForExperiment(nextExperimentId),
        preserveCurrentRun: !settings.resetSelection,
        showLoadingState:
          firstLoad ||
          Boolean(settings.resetSelection) ||
          previousExperimentId !== nextExperimentId,
      });
    } catch (error) {
      renderGlobalError(error);
    } finally {
      state.refreshInFlight = false;
      setRefreshBusy(false);
    }
  }

  function scheduleAutoRefresh() {
    if (state.refreshTimer !== null) {
      window.clearInterval(state.refreshTimer);
    }
    const refreshSeconds =
      state.config && Number.isFinite(state.config.refresh_seconds)
        ? Number(state.config.refresh_seconds)
        : 10;
    state.refreshTimer = window.setInterval(() => {
      void refreshDashboard();
    }, Math.max(2, refreshSeconds) * 1000);
  }

  function pickExperimentId(options) {
    const settings = options || {};
    const existingIds = new Set(
      state.experiments
        .map((item) => String(item.experiment_id || "").trim())
        .filter(Boolean)
    );
    const routeExperimentId = state.initialRoute.experimentId;

    if (
      !settings.resetSelection &&
      state.selectedExperimentId &&
      existingIds.has(state.selectedExperimentId)
    ) {
      return state.selectedExperimentId;
    }
    if (routeExperimentId && existingIds.has(routeExperimentId)) {
      return routeExperimentId;
    }
    if (
      settings.defaultExperimentId &&
      existingIds.has(String(settings.defaultExperimentId))
    ) {
      return String(settings.defaultExperimentId);
    }
    return state.experiments.length > 0
      ? String(state.experiments[0].experiment_id)
      : null;
  }

  async function loadExperimentSnapshot(experimentId, options) {
    const settings = options || {};
    const requestToken = ++state.experimentRequestToken;
    const targetExperimentId = String(experimentId || "").trim();
    if (!targetExperimentId) {
      renderNoExperiments();
      return;
    }

    const previousExperimentId = state.selectedExperimentId;
    const previousRunKey = state.selectedRunKey;
    state.selectedExperimentId = targetExperimentId;
    renderExperimentSelect();
    if (settings.showLoadingState !== false) {
      renderExperimentLoadingState();
    }

    try {
      const snapshot = await fetchJson(
        `/api/experiment?experiment_id=${encodeURIComponent(targetExperimentId)}`
      );
      if (requestToken !== state.experimentRequestToken) {
        return;
      }

      state.experimentSnapshot = snapshot;
      renderProjectSummary();
      renderHeroCards();

      const currentRunKey =
        settings.preserveCurrentRun && previousExperimentId === targetExperimentId
          ? state.selectedRunKey
          : null;
      const nextRun = pickRunSummary(snapshot.runs, [
        settings.preferredRunKey,
        currentRunKey,
        routeRunKeyForExperiment(targetExperimentId),
      ]);

      state.selectedRunSummary = nextRun;
      state.selectedRunKey = nextRun ? buildRunKey(nextRun) : null;
      renderRunList();
      writeRoute();

      if (nextRun) {
        await loadRunDetail(nextRun, {
          showLoadingState:
            settings.showLoadingState !== false ||
            previousRunKey !== state.selectedRunKey,
        });
        return;
      }

      clearRunDetail("No runs found for this experiment yet.");
    } catch (error) {
      renderExperimentError(error);
    }
  }

  async function loadRunDetail(runSummary, options) {
    const settings = options || {};
    const selectedKey = buildRunKey(runSummary);
    state.selectedRunSummary = runSummary;
    state.selectedRunKey = selectedKey;
    renderRunList();
    if (settings.showLoadingState !== false) {
      renderRunLoadingState(runSummary);
    }
    writeRoute();

    const requestToken = ++state.runRequestToken;

    try {
      const detail = await fetchJson(
        `/api/run?experiment_id=${encodeURIComponent(
          state.selectedExperimentId
        )}&phase=${encodeURIComponent(runSummary.phase)}&run_name=${encodeURIComponent(
          runSummary.run_name
        )}`
      );
      if (
        requestToken !== state.runRequestToken ||
        state.selectedRunKey !== selectedKey
      ) {
        return;
      }

      state.runDetail = detail;
      renderRunDetail();
      renderEvalSamples();
      renderRawPayload();
      updateDocumentTitle();
    } catch (error) {
      renderRunError(error, runSummary);
    }
  }

  function renderBootState() {
    appendEmptyState(elements.projectSummary, "Loading telemetry workspace...");
    appendEmptyState(elements.runList, "Loading runs...");
    appendEmptyState(elements.heroCards, "Loading experiment summary...");
    clearRunDetail("Select a run to inspect telemetry.");
    renderEvalSamples();
    renderRawPayload();
    updateIndicator("neutral", "Waiting for telemetry");
    updateDocumentTitle();
  }

  function renderNoExperiments() {
    renderProjectSummaryBase([
      metaItem("Backend", state.config ? state.config.backend_ref : "n/a"),
      metaItem(
        "Project",
        state.config && state.config.project_url
          ? buildLink("Open W&B project", state.config.project_url)
          : "n/a"
      ),
      metaItem(
        "Auto refresh",
        state.config ? `${state.config.refresh_seconds}s` : "n/a"
      ),
    ]);
    appendEmptyState(elements.heroCards, "No experiments found in this W&B project yet.");
    appendEmptyState(elements.runList, "No runs available.");
    elements.runCountBadge.textContent = "0";
    clearRunDetail("No experiment selected.");
    renderEvalSamples();
    renderRawPayload();
    updateIndicator("neutral", "No experiment selected");
    updateDocumentTitle();
  }

  function renderGlobalError(error) {
    const message = normalizeErrorMessage(error);
    renderProjectSummaryBase([
      metaItem("Backend", state.config ? state.config.backend_ref : "n/a"),
      metaItem("Status", message),
    ]);
    appendEmptyState(elements.heroCards, "Dashboard could not load telemetry.");
    appendEmptyState(elements.runList, message);
    elements.runCountBadge.textContent = "0";
    clearRunDetail(message);
    renderEvalSamples();
    renderRawPayload();
    updateIndicator("failed", message);
  }

  function renderExperimentLoadingState() {
    appendEmptyState(elements.runList, "Refreshing runs...");
    appendEmptyState(elements.heroCards, "Refreshing experiment summary...");
    elements.runCountBadge.textContent = "0";
    renderProjectSummary();
  }

  function renderExperimentError(error) {
    const message = normalizeErrorMessage(error);
    appendEmptyState(elements.runList, message);
    appendEmptyState(elements.heroCards, "Experiment snapshot could not be loaded.");
    elements.runCountBadge.textContent = "0";
    clearRunDetail(message);
    renderEvalSamples();
    renderRawPayload();
    updateIndicator("failed", message);
  }

  function renderExperimentSelect() {
    clearNode(elements.experimentSelect);

    if (!state.experiments.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No experiments";
      elements.experimentSelect.appendChild(option);
      elements.experimentSelect.disabled = true;
      return;
    }

    elements.experimentSelect.disabled = false;
    state.experiments.forEach((experiment) => {
      const option = document.createElement("option");
      option.value = String(experiment.experiment_id);
      option.selected = option.value === state.selectedExperimentId;
      option.textContent = `${option.value} (${experiment.run_count} runs)`;
      elements.experimentSelect.appendChild(option);
    });
  }

  function renderProjectSummary() {
    if (!state.config) {
      return;
    }

    if (!state.experimentSnapshot || !state.experimentSnapshot.experiment) {
      renderProjectSummaryBase([
        metaItem("Backend", state.config.backend_ref),
        metaItem(
          "Project",
          state.config.project_url
            ? buildLink("Open W&B project", state.config.project_url)
            : "n/a"
        ),
        metaItem("Auto refresh", `${state.config.refresh_seconds}s`),
      ]);
      return;
    }

    const experiment = state.experimentSnapshot.experiment;
    const runs = Array.isArray(state.experimentSnapshot.runs)
      ? state.experimentSnapshot.runs
      : [];
    const phaseSummary = summarizeCounts(countBy(runs, "phase"));
    const statusSummary = summarizeCounts(experiment.status_counts || {});
    const indicatorState = experimentTone(experiment);

    renderProjectSummaryBase([
      metaItem("Experiment", experiment.experiment_id),
      metaItem("Backend", experiment.backend_ref || state.config.backend_ref),
      metaItem(
        "Project",
        experiment.project_url
          ? buildLink("Open W&B project", experiment.project_url)
          : "n/a"
      ),
      metaItem("Auto refresh", `${state.config.refresh_seconds}s`),
      metaItem("Phase mix", phaseSummary || "No runs yet"),
      metaItem("Status mix", statusSummary || "No statuses yet"),
      metaItem(
        "Latest heartbeat",
        experiment.latest_heartbeat_at
          ? `${formatTimestamp(experiment.latest_heartbeat_at)} (${formatRelativeTime(
              experiment.latest_heartbeat_at
            )})`
          : "No heartbeat yet"
      ),
      metaItem("Refreshed", formatTimestamp(experiment.refreshed_at)),
    ]);

    updateIndicator(
      indicatorState,
      statusSummary || `${experiment.active_run_count} active runs`
    );
  }

  function renderProjectSummaryBase(items) {
    clearNode(elements.projectSummary);
    items.forEach((item) => {
      elements.projectSummary.appendChild(item);
    });
  }

  function renderHeroCards() {
    clearNode(elements.heroCards);

    if (!state.experimentSnapshot || !state.experimentSnapshot.experiment) {
      appendEmptyState(elements.heroCards, "No experiment summary loaded.");
      return;
    }

    const experiment = state.experimentSnapshot.experiment;
    const runs = Array.isArray(state.experimentSnapshot.runs)
      ? state.experimentSnapshot.runs
      : [];
    const phaseCounts = countBy(runs, "phase");
    const statusCounts = experiment.status_counts || {};

    const cards = [
      { label: "Total runs", value: experiment.run_count || 0 },
      { label: "Active now", value: experiment.active_run_count || 0 },
      { label: "Succeeded", value: statusCounts.success || 0 },
      { label: "Failed", value: statusCounts.failed || 0 },
      { label: "Stopped", value: statusCounts.stopped || 0 },
      { label: "Phases", value: Object.keys(phaseCounts).length || 0 },
    ];

    cards.forEach((card) => {
      const node = document.createElement("article");
      node.className = "hero-card";
      node.appendChild(createText("div", "metric-label", card.label));
      node.appendChild(
        createText("div", "hero-value", formatMetricValue(card.value, card.label))
      );
      elements.heroCards.appendChild(node);
    });
  }

  function renderRunList() {
    clearNode(elements.runList);

    const runs =
      state.experimentSnapshot && Array.isArray(state.experimentSnapshot.runs)
        ? state.experimentSnapshot.runs
        : [];

    elements.runCountBadge.textContent = String(runs.length);

    if (!runs.length) {
      appendEmptyState(elements.runList, "No stage runs found for this experiment.");
      return;
    }

    runs.forEach((run) => {
      const key = buildRunKey(run);
      const card = document.createElement("button");
      card.type = "button";
      card.className =
        "run-card" + (key === state.selectedRunKey ? " selected" : "");
      card.addEventListener("click", () => {
        void loadRunDetail(run);
      });

      const top = document.createElement("div");
      top.className = "run-card-top";
      top.appendChild(createText("div", "run-title", run.display_name || run.run_name));
      top.appendChild(buildStatusPill(run.status));

      const phase = createText(
        "div",
        "run-phase",
        [String(run.phase || "").toUpperCase(), run.provider].filter(Boolean).join(" · ")
      );

      const bottom = document.createElement("div");
      bottom.className = "run-card-bottom";
      bottom.appendChild(
        createText("span", "", pickRunPreview(run) || "No metrics logged yet")
      );
      bottom.appendChild(
        createText(
          "span",
          "",
          run.heartbeat_at
            ? formatRelativeTime(run.heartbeat_at)
            : run.created_at
            ? formatTimestamp(run.created_at)
            : "No timestamp"
        )
      );

      card.appendChild(top);
      card.appendChild(phase);
      card.appendChild(bottom);
      elements.runList.appendChild(card);
    });
  }

  function renderRunLoadingState(runSummary) {
    const title = runSummary.display_name || runSummary.run_name || "Loading run";
    elements.detailTitle.textContent = title;
    applyStatusPill(elements.detailStatus, runSummary.status || "loading");
    appendEmptyState(elements.detailMeta, "Loading run metadata...");
    appendEmptyState(elements.detailMetrics, "Loading summary metrics...");
    elements.historySummary.textContent = "Loading metric history...";
    appendEmptyState(elements.historyCharts, "Loading metric history...");
    elements.rawPayload.textContent = "Loading canonical payload...";
  }

  function clearRunDetail(message) {
    const detailMessage = message || "No run selected.";
    state.runDetail = null;
    elements.detailTitle.textContent = "Select a run";
    applyStatusPill(elements.detailStatus, "idle");
    appendEmptyState(elements.detailMeta, detailMessage);
    appendEmptyState(elements.detailMetrics, "Run-level metrics will appear here.");
    elements.historySummary.textContent = "No series loaded";
    appendEmptyState(elements.historyCharts, "Metric history will appear here.");
    elements.rawPayload.textContent = detailMessage;
    renderEvalSamples();
    updateDocumentTitle();
  }

  function renderRunError(error, runSummary) {
    const message = normalizeErrorMessage(error);
    elements.detailTitle.textContent =
      runSummary.display_name || runSummary.run_name || "Run detail";
    applyStatusPill(elements.detailStatus, "failed");
    appendEmptyState(elements.detailMeta, message);
    appendEmptyState(elements.detailMetrics, "No metrics available.");
    elements.historySummary.textContent = "No series loaded";
    appendEmptyState(elements.historyCharts, message);
    elements.rawPayload.textContent = message;
    state.runDetail = null;
    renderEvalSamples();
  }

  function renderRunDetail() {
    if (!state.runDetail) {
      clearRunDetail("No run detail loaded.");
      return;
    }

    const summary = asObject(state.runDetail.summary);
    const resultPayload = asObject(state.runDetail.result_payload);
    const jobResultPayload = asObject(state.runDetail.job_result_payload);
    const metrics = collectMetrics(state.runDetail);

    elements.detailTitle.textContent =
      summary.display_name || summary.run_name || "Run detail";
    applyStatusPill(elements.detailStatus, summary.status || "unknown");

    renderMetaGrid([
      metaItem("Phase", String(summary.phase || "unknown").toUpperCase()),
      metaItem("Provider", summary.provider || "n/a"),
      metaItem(
        "W&B run",
        summary.wandb_url ? buildLink("Open run", summary.wandb_url) : "n/a"
      ),
      metaItem(
        "HF adapter",
        summary.hf_repo_id
          ? `${summary.hf_repo_id}${summary.hf_revision ? ` @ ${summary.hf_revision}` : ""}`
          : "n/a"
      ),
      metaItem(
        "Runtime",
        Number.isFinite(Number(summary.total_time_seconds))
          ? formatDuration(Number(summary.total_time_seconds))
          : "n/a"
      ),
      metaItem(
        "Created",
        summary.created_at ? formatTimestamp(summary.created_at) : "n/a"
      ),
      metaItem(
        "Last heartbeat",
        summary.heartbeat_at
          ? `${formatTimestamp(summary.heartbeat_at)} (${formatRelativeTime(
              summary.heartbeat_at
            )})`
          : "n/a"
      ),
      metaItem(
        "Samples",
        summarizeSamples(summary.processed_samples, summary.expected_samples)
      ),
      metaItem("Attempt token", summary.attempt_token || "n/a"),
      metaItem(
        "Stop requested",
        summary.stop_requested
          ? summary.stop_requested_at
            ? `Yes at ${formatTimestamp(summary.stop_requested_at)}`
            : "Yes"
          : "No"
      ),
      metaItem(
        "Failure reason",
        summary.failure_reason || jobResultPayload.failure_reason || "n/a"
      ),
    ]);

    if (!Object.keys(metrics).length) {
      appendEmptyState(elements.detailMetrics, "No summary metrics were logged for this run.");
    } else {
      renderMetricCards(elements.detailMetrics, metrics);
    }

    renderHistoryCharts();
    renderRawPayload();
  }

  function renderMetaGrid(items) {
    clearNode(elements.detailMeta);
    items.forEach((item) => {
      elements.detailMeta.appendChild(item);
    });
  }

  function renderMetricCards(container, metrics) {
    clearNode(container);
    orderedMetricEntries(metrics).forEach(([key, value]) => {
      const card = document.createElement("article");
      card.className = "metric-card";
      card.appendChild(createText("div", "metric-label", prettifyKey(key)));
      card.appendChild(createText("div", "metric-value", formatMetricValue(value, key)));
      container.appendChild(card);
    });
  }

  function renderHistoryCharts() {
    clearNode(elements.historyCharts);

    const history =
      state.runDetail && state.runDetail.history ? asObject(state.runDetail.history) : {};
    const rows = Array.isArray(history.rows) ? history.rows : [];
    const allKeys = Array.isArray(history.keys) ? history.keys : [];
    const selectedKeys = pickChartKeys(history, state.selectedRunSummary);

    if (!rows.length || !selectedKeys.length) {
      elements.historySummary.textContent = "No numeric history logged for this run";
      appendEmptyState(elements.historyCharts, "No metric history available yet.");
      return;
    }

    selectedKeys.forEach((key, index) => {
      elements.historyCharts.appendChild(
        buildChartCard(rows, key, chartColors[index % chartColors.length])
      );
    });

    elements.historySummary.textContent = `Showing ${selectedKeys.length} of ${allKeys.length} numeric series across ${pluralize(rows.length, "point")}`;
  }

  function renderEvalSamples() {
    const resultPayload =
      state.runDetail && state.runDetail.result_payload
        ? asObject(state.runDetail.result_payload)
        : {};
    const detailedRows = Array.isArray(resultPayload.detailed_results)
      ? resultPayload.detailed_results
      : [];
    const normalizedRows = detailedRows.map(normalizeEvalRow);
    const metrics = Object.assign(
      {},
      asObject(resultPayload.metrics),
      deriveRewardMetrics(normalizedRows)
    );
    const fallbackMetrics = state.runDetail ? collectMetrics(state.runDetail) : {};
    const failureCount = normalizedRows.filter((row) => !row.outcome.passed).length;
    const rewardCounts = countRewardBuckets(normalizedRows);

    toggleSampleFilters(detailedRows.length > 0, rewardCounts, {
      totalCount: normalizedRows.length,
      failureCount: failureCount,
    });

    if (Object.keys(metrics).length) {
      renderMetricCards(elements.evalSummary, metrics);
    } else if (
      state.selectedRunSummary &&
      String(state.selectedRunSummary.phase || "").toLowerCase() === "eval" &&
      Object.keys(fallbackMetrics).length
    ) {
      renderMetricCards(elements.evalSummary, fallbackMetrics);
    } else {
      appendEmptyState(
        elements.evalSummary,
        "Detailed eval metrics appear here when the run logs a structured result payload."
      );
    }

    clearNode(elements.evalTableBody);

    if (!normalizedRows.length) {
      appendTableMessage("No detailed sample rows were logged for this run.");
      return;
    }

    const filteredRows = normalizedRows.filter((row) => {
      if (state.showFailuresOnly && row.outcome.passed) {
        return false;
      }
      return matchesRewardFilter(row.reward, state.rewardFilterMode);
    });

    if (!filteredRows.length) {
      appendTableMessage(
        state.showFailuresOnly || state.rewardFilterMode !== "any"
          ? "No samples match the current filters."
          : "All logged samples passed the current checks."
      );
      return;
    }

    filteredRows.forEach((row) => {
      const outcome = row.outcome;
      const tr = document.createElement("tr");

      const statusCell = document.createElement("td");
      statusCell.className = "status-cell";
      const badge = createText(
        "span",
        `eval-status ${outcome.passed ? "pass" : "fail"}`,
        outcome.passed ? "Pass" : "Needs review"
      );
      statusCell.appendChild(badge);
      if (outcome.reasons.length) {
        statusCell.appendChild(
          createText("div", "muted-copy", outcome.reasons.join(", "))
        );
      }

      const rewardCell = document.createElement("td");
      rewardCell.className = "reward-cell";
      rewardCell.appendChild(buildRewardCell(row.raw, row.reward));

      const guessCell = createText(
        "td",
        "guess-cell",
        row.raw && row.raw.parsed_guess ? String(row.raw.parsed_guess) : "-"
      );

      const completionCell = document.createElement("td");
      completionCell.className = "completion-cell";
      completionCell.appendChild(
        buildExpandableCopy(
          "completion-copy",
          row.raw && row.raw.completion ? String(row.raw.completion) : "",
          520
        )
      );

      const promptCell = document.createElement("td");
      promptCell.className = "prompt-cell";
      promptCell.appendChild(
        buildExpandableCopy(
          "prompt-copy",
          row.raw && row.raw.prompt ? String(row.raw.prompt) : "",
          420
        )
      );

      tr.appendChild(statusCell);
      tr.appendChild(rewardCell);
      tr.appendChild(guessCell);
      tr.appendChild(completionCell);
      tr.appendChild(promptCell);
      elements.evalTableBody.appendChild(tr);
    });
  }

  function renderRawPayload() {
    if (!state.runDetail) {
      elements.rawPayload.textContent = "No run selected.";
      return;
    }

    const history = asObject(state.runDetail.history);
    const payload = {
      summary: asObject(state.runDetail.summary),
      result_payload: asObject(state.runDetail.result_payload),
      job_result_payload: asObject(state.runDetail.job_result_payload),
      config: asObject(state.runDetail.config),
      history_preview: {
        keys: Array.isArray(history.keys) ? history.keys : [],
        default_keys: Array.isArray(history.default_keys) ? history.default_keys : [],
        row_count: Array.isArray(history.rows) ? history.rows.length : 0,
      },
    };
    elements.rawPayload.textContent = JSON.stringify(payload, null, 2);
  }

  function buildChartCard(rows, key, color) {
    const card = document.createElement("article");
    card.className = "chart-card";

    const titleRow = document.createElement("div");
    titleRow.className = "chart-title-row";
    titleRow.appendChild(createText("h3", "", prettifyKey(key)));
    const values = rows
      .map((row, index) => ({
        x: Number.isFinite(Number(row.step)) ? Number(row.step) : index,
        y: Number(row[key]),
      }))
      .filter((item) => Number.isFinite(item.y));
    const latestValue = values.length ? values[values.length - 1].y : null;
    titleRow.appendChild(
      createText("span", "muted-copy", formatMetricValue(latestValue, key))
    );
    card.appendChild(titleRow);

    if (!values.length) {
      card.appendChild(createText("div", "history-empty", "No values logged."));
      return card;
    }

    if (values.length === 1) {
      card.classList.add("single-point");
      card.appendChild(
        createText(
          "div",
          "chart-single-value",
          formatMetricValue(values[0].y, key)
        )
      );
      card.appendChild(
        createText(
          "div",
          "chart-note muted-copy",
          "Only one telemetry point is available so there is no trend line yet."
        )
      );
      const footer = document.createElement("div");
      footer.className = "chart-footer";
      footer.appendChild(createText("span", "", `min ${formatMetricValue(values[0].y, key)}`));
      footer.appendChild(createText("span", "", `max ${formatMetricValue(values[0].y, key)}`));
      footer.appendChild(createText("span", "", "1 point"));
      card.appendChild(footer);
      return card;
    }

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "chart-spark");
    svg.setAttribute("viewBox", "0 0 260 170");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${prettifyKey(key)} history`);

    const padding = 14;
    const width = 260;
    const height = 170;
    const baselineY = height - padding;
    const innerWidth = width - padding * 2;
    const innerHeight = height - padding * 2;
    const xValues = values.map((item) => item.x);
    const yValues = values.map((item) => item.y);
    const minX = Math.min.apply(null, xValues);
    const maxX = Math.max.apply(null, xValues);
    const minY = Math.min.apply(null, yValues);
    const maxY = Math.max.apply(null, yValues);
    const xSpan = maxX - minX || Math.max(values.length - 1, 1);
    const ySpan = maxY - minY;

    for (let index = 0; index < 4; index += 1) {
      const guide = document.createElementNS("http://www.w3.org/2000/svg", "line");
      const y = padding + (innerHeight * index) / 3;
      guide.setAttribute("x1", String(padding));
      guide.setAttribute("x2", String(width - padding));
      guide.setAttribute("y1", y.toFixed(2));
      guide.setAttribute("y2", y.toFixed(2));
      guide.setAttribute("stroke", "rgba(28, 45, 40, 0.08)");
      guide.setAttribute("stroke-width", "1");
      svg.appendChild(guide);
    }

    const coordinates = values.map((item, index) => {
      const x =
        padding +
        (((item.x - minX) || (values.length === 1 ? 0 : index)) / xSpan) * innerWidth;
      const y =
        ySpan === 0
          ? padding + innerHeight / 2
          : baselineY - ((item.y - minY) / ySpan) * innerHeight;
      return [x, y];
    });

    const linePath = pathFromCoordinates(coordinates);
    const areaPath = `${linePath} L ${coordinates[coordinates.length - 1][0].toFixed(
      2
    )} ${baselineY.toFixed(2)} L ${coordinates[0][0].toFixed(
      2
    )} ${baselineY.toFixed(2)} Z`;

    const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
    area.setAttribute("d", areaPath);
    area.setAttribute("fill", withAlpha(color, 0.16));
    svg.appendChild(area);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", linePath);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "2.5");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("stroke-linejoin", "round");
    svg.appendChild(line);

    const last = coordinates[coordinates.length - 1];
    const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    point.setAttribute("cx", last[0].toFixed(2));
    point.setAttribute("cy", last[1].toFixed(2));
    point.setAttribute("r", "4");
    point.setAttribute("fill", color);
    point.setAttribute("stroke", "#fffdfa");
    point.setAttribute("stroke-width", "2");
    svg.appendChild(point);

    card.appendChild(svg);

    const footer = document.createElement("div");
    footer.className = "chart-footer";
    footer.appendChild(
      createText("span", "", `min ${formatMetricValue(minY, key)}`)
    );
    footer.appendChild(
      createText("span", "", `max ${formatMetricValue(maxY, key)}`)
    );
    footer.appendChild(createText("span", "", `${values.length} points`));
    card.appendChild(footer);

    return card;
  }

  function pickChartKeys(history, runSummary) {
    const keys = Array.isArray(history.keys) ? history.keys : [];
    const defaults = Array.isArray(history.default_keys) ? history.default_keys : [];
    const phase = runSummary ? String(runSummary.phase || "").toLowerCase() : "";
    const preferred = metricPriority[phase] || [];
    const selected = [];

    defaults.forEach((key) => {
      if (keys.includes(key) && !selected.includes(key)) {
        selected.push(key);
      }
    });

    preferred.forEach((key) => {
      if (keys.includes(key) && !selected.includes(key)) {
        selected.push(key);
      }
    });

    keys.forEach((key) => {
      if (!selected.includes(key) && selected.length < 4) {
        selected.push(key);
      }
    });

    return selected.slice(0, 4);
  }

  function pickRunSummary(runs, candidateKeys) {
    const list = Array.isArray(runs) ? runs : [];
    const candidates = Array.isArray(candidateKeys)
      ? candidateKeys.filter(Boolean)
      : [];
    for (const candidate of candidates) {
      const match = list.find((run) => buildRunKey(run) === candidate);
      if (match) {
        return match;
      }
    }
    return list.length ? list[0] : null;
  }

  function pickRunPreview(run) {
    const metrics = asObject(run.metrics);
    const entries = orderedMetricEntries(metrics, run.phase);
    if (entries.length) {
      const entry = entries[0];
      return `${prettifyKey(entry[0])} ${formatMetricValue(entry[1], entry[0])}`;
    }
    if (Number.isFinite(Number(run.total_time_seconds))) {
      return formatDuration(Number(run.total_time_seconds));
    }
    if (run.hf_revision) {
      return `Adapter ${String(run.hf_revision).slice(0, 8)}`;
    }
    return "";
  }

  function collectMetrics(detail) {
    const summary = asObject(detail.summary);
    const summaryMetrics = asObject(summary.metrics);
    const resultMetrics = asObject(asObject(detail.result_payload).metrics);
    return Object.assign({}, resultMetrics, summaryMetrics);
  }

  function orderedMetricEntries(metrics, phaseOverride) {
    const entries = Object.entries(asObject(metrics)).filter((entry) =>
      isDisplayableMetricValue(entry[1])
    );
    const phase =
      phaseOverride ||
      (state.selectedRunSummary ? String(state.selectedRunSummary.phase || "") : "");
    const preferred = metricPriority[String(phase || "").toLowerCase()] || [];

    entries.sort((left, right) => {
      const leftIndex = preferred.indexOf(left[0]);
      const rightIndex = preferred.indexOf(right[0]);
      if (leftIndex !== -1 || rightIndex !== -1) {
        if (leftIndex === -1) {
          return 1;
        }
        if (rightIndex === -1) {
          return -1;
        }
        return leftIndex - rightIndex;
      }
      return left[0].localeCompare(right[0]);
    });
    return entries;
  }

  function isDisplayableMetricValue(value) {
    return (
      typeof value === "number" ||
      typeof value === "string" ||
      typeof value === "boolean"
    );
  }

  function appendEmptyState(container, message) {
    clearNode(container);
    container.appendChild(createText("div", "empty-state", message));
  }

  function appendTableMessage(message) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.appendChild(createText("div", "empty-state", message));
    row.appendChild(cell);
    elements.evalTableBody.appendChild(row);
  }

  function toggleSampleFilters(enabled, rewardCounts, summary) {
    const details = summary || {};
    const totalCount = Number(details.totalCount || 0);
    const failureCount = Number(details.failureCount || 0);
    const rewards = rewardCounts || {
      available: 0,
      positive: 0,
      zero: 0,
      negative: 0,
      nonpositive: 0,
    };
    elements.showAllRows.disabled = !enabled;
    elements.showFailedRows.disabled = !enabled;
    elements.showAllRows.textContent = enabled ? `All (${totalCount})` : "All";
    elements.showFailedRows.textContent = enabled
      ? `Failures (${failureCount})`
      : "Failures";
    elements.showAllRows.classList.toggle("active", !state.showFailuresOnly);
    elements.showFailedRows.classList.toggle("active", state.showFailuresOnly);
    elements.showAllRows.setAttribute("aria-pressed", String(!state.showFailuresOnly));
    elements.showFailedRows.setAttribute("aria-pressed", String(state.showFailuresOnly));

    if (!enabled || !rewards.available) {
      state.rewardFilterMode = "any";
    }
    updateRewardFilterButton("any", "Any reward", rewards.available || totalCount, {
      disabled: !enabled,
    });
    updateRewardFilterButton("positive", "Positive only", rewards.positive, {
      disabled: !enabled || !rewards.available,
    });
    updateRewardFilterButton("zero", "Zero only", rewards.zero, {
      disabled: !enabled || !rewards.available,
    });
    updateRewardFilterButton("negative", "Negative only", rewards.negative, {
      disabled: !enabled || !rewards.available,
    });
    updateRewardFilterButton("nonpositive", "Zero or below", rewards.nonpositive, {
      disabled: !enabled || !rewards.available,
    });
  }

  function buildExpandableCopy(className, text, previewLength) {
    const wrapper = document.createElement("div");
    wrapper.className = "expand-shell";
    const fullText = String(text || "");
    const previewText = summarizeText(fullText, previewLength);
    wrapper.appendChild(createText("div", `${className} copy-preview`, previewText || " "));

    if (previewText !== fullText) {
      const details = document.createElement("details");
      details.className = "expand-panel";
      const summary = createText("summary", "expand-toggle", "Show full");
      details.addEventListener("toggle", () => {
        summary.textContent = details.open ? "Show less" : "Show full";
      });
      details.appendChild(summary);
      details.appendChild(createText("div", className, fullText));
      wrapper.appendChild(details);
    }

    return wrapper;
  }

  function normalizeEvalRow(row) {
    const record = asObject(row);
    return {
      raw: record,
      reward: extractRowReward(record),
      outcome: sampleOutcome(record),
    };
  }

  function buildRewardCell(row, rewardValue) {
    const wrapper = document.createElement("div");
    wrapper.className = "reward-shell";
    if (!Number.isFinite(Number(rewardValue))) {
      wrapper.appendChild(createText("div", "muted-copy", "n/a"));
      return wrapper;
    }

    const chip = createText(
      "span",
      `reward-chip ${rewardTone(rewardValue)}`,
      formatSignedValue(rewardValue)
    );
    wrapper.appendChild(chip);

    const breakdown = summarizeRewardBreakdown(row);
    if (breakdown) {
      wrapper.appendChild(createText("div", "muted-copy reward-breakdown", breakdown));
    }
    return wrapper;
  }

  function deriveRewardMetrics(rows) {
    const values = rows
      .map((row) => row.reward)
      .filter((value) => Number.isFinite(Number(value)))
      .map((value) => Number(value));
    if (!values.length) {
      return {};
    }
    const sum = values.reduce((total, value) => total + value, 0);
    return {
      avg_reward: sum / values.length,
      min_reward: Math.min.apply(null, values),
      max_reward: Math.max.apply(null, values),
    };
  }

  function countRewardBuckets(rows) {
    return rows.reduce(
      (counts, row) => {
        if (!Number.isFinite(Number(row.reward))) {
          return counts;
        }
        const reward = Number(row.reward);
        counts.available += 1;
        if (Math.abs(reward) < 1e-9) {
          counts.zero += 1;
          counts.nonpositive += 1;
          return counts;
        }
        if (reward > 0) {
          counts.positive += 1;
          return counts;
        }
        counts.negative += 1;
        counts.nonpositive += 1;
        return counts;
      },
      { available: 0, positive: 0, zero: 0, negative: 0, nonpositive: 0 }
    );
  }

  function updateRewardFilterButton(value, label, count, options) {
    const settings = options || {};
    const button = elements.rewardFilterButtons.find(
      (candidate) => candidate.dataset.rewardFilter === value
    );
    if (!button) {
      return;
    }
    button.textContent = `${label} (${Number(count || 0)})`;
    button.disabled = Boolean(settings.disabled);
    const isActive = state.rewardFilterMode === value;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }

  function matchesRewardFilter(rewardValue, mode) {
    const filterMode = String(mode || "any");
    if (filterMode === "any") {
      return true;
    }
    if (!Number.isFinite(Number(rewardValue))) {
      return false;
    }
    const reward = Number(rewardValue);
    if (filterMode === "positive") {
      return reward > 0;
    }
    if (filterMode === "zero") {
      return Math.abs(reward) < 1e-9;
    }
    if (filterMode === "negative") {
      return reward < 0;
    }
    if (filterMode === "nonpositive") {
      return reward <= 0;
    }
    return true;
  }

  function extractRowReward(row) {
    const record = asObject(row);
    const rewardKeys = ["reward_total", "reward", "total_reward"];
    for (const key of rewardKeys) {
      if (Number.isFinite(Number(record[key]))) {
        return Number(record[key]);
      }
    }
    return null;
  }

  function summarizeRewardBreakdown(row) {
    const record = asObject(row);
    const rewardParts = [
      ["fmt", "reward_format"],
      ["dict", "reward_dict"],
      ["constraints", "reward_constraints"],
      ["repeat", "reward_repeat"],
      ["length", "reward_overlength"],
    ]
      .filter((entry) => Number.isFinite(Number(record[entry[1]])))
      .map((entry) => `${entry[0]} ${formatSignedValue(record[entry[1]])}`);
    if (rewardParts.length) {
      return rewardParts.join(" · ");
    }

    return Object.entries(asObject(record.reward_components))
      .filter((entry) => Number.isFinite(Number(entry[1])))
      .map((entry) => `${prettifyKey(entry[0])} ${formatSignedValue(entry[1])}`)
      .join(" · ");
  }

  function sampleOutcome(row) {
    const record = row && typeof row === "object" ? row : {};
    const explicitReasons = Array.isArray(record.failure_reasons)
      ? record.failure_reasons
          .map((reason) => prettifyKey(reason))
          .filter((reason) => reason && reason !== "n/a")
      : [];
    if (Object.prototype.hasOwnProperty.call(record, "passed")) {
      const passed = Boolean(record.passed);
      if (passed) {
        return { passed: true, reasons: [] };
      }
      if (explicitReasons.length) {
        return { passed: false, reasons: explicitReasons };
      }
    }

    const statusText = String(record.status || "").trim().toLowerCase();
    if (statusText) {
      if (["pass", "passed", "success", "ok"].includes(statusText)) {
        return { passed: true, reasons: [] };
      }
      if (["fail", "failed", "error"].includes(statusText)) {
        return { passed: false, reasons: ["status"] };
      }
    }

    const booleanEntries = Object.entries(record).filter(
      (entry) => typeof entry[1] === "boolean"
    );
    if (!booleanEntries.length) {
      return { passed: true, reasons: [] };
    }

    const reasons = booleanEntries
      .filter((entry) => !entry[1])
      .map((entry) => prettifyKey(entry[0]));
    return { passed: reasons.length === 0, reasons: reasons };
  }

  function metaItem(label, value) {
    const item = document.createElement("div");
    item.className = "meta-item";
    item.appendChild(createText("div", "meta-label", label));
    const valueNode = document.createElement("div");
    valueNode.className = "meta-value";
    if (value instanceof Node) {
      valueNode.appendChild(value);
    } else {
      valueNode.textContent = String(value);
    }
    item.appendChild(valueNode);
    return item;
  }

  function buildStatusPill(status) {
    const node = document.createElement("span");
    applyStatusPill(node, status);
    return node;
  }

  function applyStatusPill(node, status) {
    const tone = statusTone(status);
    node.className = `status-pill ${tone}`;
    node.textContent = String(status || "unknown");
  }

  function updateIndicator(tone, title) {
    elements.liveIndicator.className = `status-dot ${tone}`;
    elements.liveIndicator.title = title || "";
  }

  function setRefreshBusy(isBusy) {
    elements.refreshButton.disabled = isBusy;
    elements.refreshButton.textContent = isBusy ? "Refreshing..." : "Refresh";
  }

  function statusTone(status) {
    const value = String(status || "").trim().toLowerCase();
    if (!value || value === "idle" || value === "unknown") {
      return "neutral";
    }
    if (value.includes("success") || value.includes("complete")) {
      return "success";
    }
    if (value.includes("run") || value.includes("active")) {
      return "running";
    }
    if (value.includes("partial")) {
      return "partial";
    }
    if (value.includes("stop")) {
      return "stopped";
    }
    if (value.includes("fail") || value.includes("error")) {
      return "failed";
    }
    return "neutral";
  }

  function countBy(items, key) {
    return (Array.isArray(items) ? items : []).reduce((counts, item) => {
      const rawValue =
        item && Object.prototype.hasOwnProperty.call(item, key) ? item[key] : "unknown";
      const value = String(rawValue == null ? "unknown" : rawValue).trim() || "unknown";
      counts[value] = (counts[value] || 0) + 1;
      return counts;
    }, {});
  }

  function experimentTone(experiment) {
    const runCount = Number(experiment && experiment.run_count ? experiment.run_count : 0);
    const activeCount = Number(
      experiment && experiment.active_run_count ? experiment.active_run_count : 0
    );
    const statuses = asObject(experiment && experiment.status_counts);
    const failed = Number(statuses.failed || 0);
    const partial = Number(statuses.partial || 0);
    const stopped = Number(statuses.stopped || 0);
    const success = Number(statuses.success || 0);

    if (activeCount > 0) {
      return "running";
    }
    if (failed > 0) {
      return "failed";
    }
    if (partial > 0) {
      return "partial";
    }
    if (stopped > 0 && runCount > 0 && stopped === runCount) {
      return "stopped";
    }
    if (stopped > 0) {
      return "partial";
    }
    if (success > 0) {
      return "success";
    }
    return "neutral";
  }

  function summarizeCounts(counts) {
    return Object.entries(counts || {})
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .map((entry) => `${entry[0]} ${entry[1]}`)
      .join(" · ");
  }

  function summarizeSamples(processed, expected) {
    if (processed == null && expected == null) {
      return "n/a";
    }
    if (processed != null && expected != null) {
      return `${processed} / ${expected}`;
    }
    return processed != null ? String(processed) : `expected ${expected}`;
  }

  function buildRunKey(run) {
    return `${String(run.phase || "")}::${String(run.run_name || "")}`;
  }

  function pluralize(count, noun) {
    const numericCount = Number(count || 0);
    return `${numericCount} ${noun}${numericCount === 1 ? "" : "s"}`;
  }

  function routeRunKeyForExperiment(experimentId) {
    if (
      !state.initialRoute.experimentId ||
      state.initialRoute.experimentId !== experimentId
    ) {
      return null;
    }
    if (!state.initialRoute.phase || !state.initialRoute.runName) {
      return null;
    }
    return `${state.initialRoute.phase}::${state.initialRoute.runName}`;
  }

  function writeRoute() {
    const params = new URLSearchParams(window.location.search);
    if (state.selectedExperimentId) {
      params.set("experiment_id", state.selectedExperimentId);
    } else {
      params.delete("experiment_id");
    }

    if (state.selectedRunSummary) {
      params.set("phase", String(state.selectedRunSummary.phase || ""));
      params.set("run_name", String(state.selectedRunSummary.run_name || ""));
    } else {
      params.delete("phase");
      params.delete("run_name");
    }

    const nextQuery = params.toString();
    const nextUrl = nextQuery
      ? `${window.location.pathname}?${nextQuery}`
      : window.location.pathname;
    window.history.replaceState({}, "", nextUrl);
    state.initialRoute = readRoute();
  }

  function readRoute() {
    const params = new URLSearchParams(window.location.search);
    return {
      experimentId: trimOrNull(params.get("experiment_id")),
      phase: trimOrNull(params.get("phase")),
      runName: trimOrNull(params.get("run_name")),
    };
  }

  function updateDocumentTitle() {
    const experimentId = state.selectedExperimentId;
    const runName =
      state.selectedRunSummary &&
      String(state.selectedRunSummary.run_name || "").trim();
    if (experimentId && runName) {
      document.title = `Tenyson Dashboard · ${experimentId} · ${runName}`;
      return;
    }
    if (experimentId) {
      document.title = `Tenyson Dashboard · ${experimentId}`;
      return;
    }
    document.title = "Tenyson Dashboard";
  }

  function buildLink(label, href) {
    const link = document.createElement("a");
    link.href = String(href);
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = label;
    return link;
  }

  function createText(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    node.textContent = text == null ? "" : String(text);
    return node;
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (response.ok) {
      return response.json();
    }

    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      if (payload && payload.error) {
        message = String(payload.error);
      }
    } catch (_error) {
      try {
        const text = await response.text();
        if (text) {
          message = text;
        }
      } catch (_ignored) {
        // Ignore fallback parse errors and use the default message.
      }
    }
    throw new Error(message);
  }

  function normalizeErrorMessage(error) {
    if (!error) {
      return "Unknown dashboard error.";
    }
    return error instanceof Error && error.message ? error.message : String(error);
  }

  function formatMetricValue(value, key) {
    if (value == null || value === "") {
      return "n/a";
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    if (typeof value === "string") {
      return value;
    }
    if (!Number.isFinite(Number(value))) {
      return String(value);
    }

    const numeric = Number(value);
    const normalizedKey = String(key || "").toLowerCase();
    if (normalizedKey.includes("accuracy") || normalizedKey.endsWith("_rate")) {
      return `${(numeric * 100).toFixed(numeric >= 0.995 ? 0 : 1)}%`;
    }
    if (Number.isInteger(numeric) && Math.abs(numeric) < 1e9) {
      return new Intl.NumberFormat().format(numeric);
    }
    if (Math.abs(numeric) >= 1000) {
      return new Intl.NumberFormat(undefined, {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(numeric);
    }
    if (Math.abs(numeric) >= 100) {
      return numeric.toFixed(1);
    }
    if (Math.abs(numeric) >= 1) {
      return numeric.toFixed(3).replace(/\.?0+$/, "");
    }
    return numeric.toFixed(4).replace(/\.?0+$/, "");
  }

  function formatSignedValue(value) {
    if (!Number.isFinite(Number(value))) {
      return "n/a";
    }
    const numeric = Number(value);
    const formatted = formatMetricValue(numeric, "reward");
    return numeric > 0 ? `+${formatted}` : formatted;
  }

  function rewardTone(value) {
    if (!Number.isFinite(Number(value))) {
      return "neutral";
    }
    const numeric = Number(value);
    if (numeric > 0) {
      return "positive";
    }
    if (numeric < 0) {
      return "negative";
    }
    return "neutral";
  }

  function prettifyKey(key) {
    return String(key || "")
      .replace(/\//g, " / ")
      .replace(/_/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function formatTimestamp(value) {
    const date = toDate(value);
    if (!date) {
      return "n/a";
    }
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function formatRelativeTime(value) {
    const date = toDate(value);
    if (!date) {
      return "n/a";
    }

    const deltaMs = date.getTime() - Date.now();
    const absMs = Math.abs(deltaMs);
    const units = [
      ["day", 24 * 60 * 60 * 1000],
      ["hour", 60 * 60 * 1000],
      ["minute", 60 * 1000],
      ["second", 1000],
    ];
    const formatter =
      typeof Intl !== "undefined" && typeof Intl.RelativeTimeFormat === "function"
        ? new Intl.RelativeTimeFormat(undefined, { numeric: "auto" })
        : null;

    for (const unit of units) {
      const unitName = unit[0];
      const unitMs = unit[1];
      if (absMs >= unitMs || unitName === "second") {
        const amount = Math.round(deltaMs / unitMs);
        if (formatter) {
          return formatter.format(amount, unitName);
        }
        return `${Math.abs(amount)} ${unitName}${Math.abs(amount) === 1 ? "" : "s"} ago`;
      }
    }
    return "just now";
  }

  function formatDuration(totalSeconds) {
    if (!Number.isFinite(totalSeconds)) {
      return "n/a";
    }
    const rounded = Math.max(0, Math.round(totalSeconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const seconds = rounded % 60;
    const parts = [];
    if (hours) {
      parts.push(`${hours}h`);
    }
    if (minutes) {
      parts.push(`${minutes}m`);
    }
    if (seconds || !parts.length) {
      parts.push(`${seconds}s`);
    }
    return parts.join(" ");
  }

  function toDate(value) {
    if (!value) {
      return null;
    }
    const date = value instanceof Date ? value : new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function pathFromCoordinates(coordinates) {
    return coordinates
      .map((pair, index) => {
        const command = index === 0 ? "M" : "L";
        return `${command} ${pair[0].toFixed(2)} ${pair[1].toFixed(2)}`;
      })
      .join(" ");
  }

  function withAlpha(hex, alpha) {
    const cleaned = String(hex || "").replace("#", "");
    const normalized =
      cleaned.length === 3
        ? cleaned
            .split("")
            .map((part) => part + part)
            .join("")
        : cleaned;
    const red = parseInt(normalized.slice(0, 2), 16) || 0;
    const green = parseInt(normalized.slice(2, 4), 16) || 0;
    const blue = parseInt(normalized.slice(4, 6), 16) || 0;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function trimOrNull(value) {
    const text = String(value || "").trim();
    return text || null;
  }

  function summarizeText(text, maxChars) {
    const source = String(text || "");
    if (!source || source.length <= maxChars) {
      return source;
    }
    return `${source.slice(0, maxChars).trimEnd()}…`;
  }
})();
