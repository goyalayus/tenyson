"use strict";

(function () {
  const { useEffect, useRef, useState, useTransition } = React;
  const html = htm.bind(React.createElement);

  const chartColors = ["#1f6f5f", "#c56d3d", "#2e6aa2", "#b1831d", "#855c9e"];
  const metricPriority = {
    sft: [
      "train_loss",
      "train/loss",
      "epoch",
      "train/epoch",
      "global_step",
      "train/global_step",
      "train/grad_norm",
      "train/learning_rate",
      "train_steps_per_second",
      "train_samples_per_second",
    ],
    rl: [
      "train/reward",
      "train_reward",
      "train/kl",
      "train/loss",
      "train_loss",
      "global_step",
      "train/global_step",
      "train/reward_std",
      "train/completion_length",
      "train_steps_per_second",
      "train_samples_per_second",
      "train/learning_rate",
    ],
    eval: [
      "exact_match_accuracy",
      "format_accuracy",
      "parsed_answer_rate",
      "avg_abs_error",
      "total_samples",
    ],
  };
  const overviewMetricGroups = {
    sft: [
      { key: "train_loss", sources: ["train_loss", "train/loss"] },
      { key: "train/grad_norm", sources: ["train/grad_norm"] },
      { key: "train/learning_rate", sources: ["train/learning_rate"] },
      { key: "epoch", sources: ["epoch", "train/epoch"] },
      { key: "global_step", sources: ["global_step", "train/global_step"] },
      {
        key: "train_steps_per_second",
        sources: ["train_steps_per_second", "train/steps_per_second"],
      },
    ],
    rl: [
      { key: "train/reward", sources: ["train/reward", "train_reward", "reward"] },
      { key: "train/kl", sources: ["train/kl", "kl"] },
      { key: "train/loss", sources: ["train/loss", "train_loss", "loss"] },
      { key: "train/reward_std", sources: ["train/reward_std", "reward_std"] },
      {
        key: "train/completion_length",
        sources: ["train/completion_length", "completion_length"],
      },
      { key: "global_step", sources: ["global_step", "train/global_step"] },
    ],
    eval: [
      { key: "exact_match_accuracy", sources: ["exact_match_accuracy"] },
      { key: "avg_abs_error", sources: ["avg_abs_error"] },
      { key: "format_accuracy", sources: ["format_accuracy"] },
      { key: "parsed_answer_rate", sources: ["parsed_answer_rate"] },
      { key: "total_samples", sources: ["total_samples"] },
    ],
  };

  function DashboardApp() {
    const initialRouteRef = useRef(readRoute());
    const refreshTokenRef = useRef(0);
    const runTokenRef = useRef(0);
    const intervalRef = useRef(null);
    const detailCacheRef = useRef({});
    const detailLoadingRef = useRef({});
    const detailErrorRef = useRef({});
    const [config, setConfig] = useState(null);
    const [experiments, setExperiments] = useState([]);
    const [experimentsLoaded, setExperimentsLoaded] = useState(false);
    const [selectedExperimentId, setSelectedExperimentId] = useState(
      initialRouteRef.current.experimentId || null
    );
    const [experimentSnapshot, setExperimentSnapshot] = useState(null);
    const [snapshotLoading, setSnapshotLoading] = useState(true);
    const [selectedRunKey, setSelectedRunKey] = useState(
      routeRunKeyForExperiment(initialRouteRef.current, initialRouteRef.current.experimentId)
    );
    const [runDetailCache, setRunDetailCache] = useState({});
    const [runDetailLoadingByKey, setRunDetailLoadingByKey] = useState({});
    const [runDetailErrorByKey, setRunDetailErrorByKey] = useState({});
    const [selectedPanel, setSelectedPanel] = useState(
      initialRouteRef.current.panel || "overview"
    );
    const [showFailuresOnly, setShowFailuresOnly] = useState(false);
    const [rewardFilterMode, setRewardFilterMode] = useState("any");
    const [refreshInFlight, setRefreshInFlight] = useState(false);
    const [globalError, setGlobalError] = useState("");
    const [, startTransition] = useTransition();

    const experiment = asObject(experimentSnapshot && experimentSnapshot.experiment);
    const runs = Array.isArray(experimentSnapshot && experimentSnapshot.runs)
      ? experimentSnapshot.runs
      : [];
    const selectedRunSummary = pickRunSummary(runs, [selectedRunKey]);
    const selectedRunIdentity = selectedRunSummary
      ? buildRunKey(selectedRunSummary)
      : null;
    const selectedRunCacheKey =
      selectedRunSummary && selectedExperimentId
        ? buildDetailCacheKey(selectedExperimentId, selectedRunSummary)
        : null;
    const runDetail = selectedRunCacheKey ? runDetailCache[selectedRunCacheKey] || null : null;
    const runDetailLoading = selectedRunCacheKey
      ? Boolean(runDetailLoadingByKey[selectedRunCacheKey])
      : false;
    const runDetailError = selectedRunCacheKey
      ? runDetailErrorByKey[selectedRunCacheKey] || ""
      : "";
    const activePanel = resolveActivePanel(
      selectedPanel,
      selectedRunSummary,
      runDetail
    );
    const loadingDetail =
      runDetailLoading &&
      Boolean(selectedRunSummary) &&
      (!runDetail || buildRunKey(asObject(runDetail.summary)) !== selectedRunIdentity);
    const overviewMetrics = runDetail ? collectOverviewMetrics(runDetail) : {};
    const projectItems = buildProjectSummaryItems(config, experimentSnapshot, globalError);
    const heroCards = buildHeroCards(experimentSnapshot);
    const detailFacts = selectedRunSummary
      ? buildRunFacts(selectedRunSummary, overviewMetrics, loadingDetail)
      : [{ label: "Selection", value: "No run selected" }];
    const detailActions = selectedRunSummary
      ? buildRunActions(selectedRunSummary, config)
      : [];
    const detailMetaItems = selectedRunSummary
      ? buildRunMetaItems(
          selectedRunSummary,
          asObject(runDetail && runDetail.job_result_payload)
        )
      : [];
    const detailMetrics = orderedMetricEntries(overviewMetrics, selectedRunSummary && selectedRunSummary.phase);
    const detailStageNote = runDetailError
      ? "Detail failed to load"
      : loadingDetail
      ? "Loading deeper telemetry…"
      : selectedRunSummary
      ? "Detail ready"
      : "Snapshot ready";
    const detailStatus =
      runDetailError && selectedRunSummary
        ? "failed"
        : selectedRunSummary
        ? selectedRunSummary.status
        : "idle";
    const historyState = buildHistoryState(runDetail, selectedRunSummary, loadingDetail);
    const samplesState = buildSamplesState(
      runDetail,
      selectedRunSummary,
      loadingDetail,
      showFailuresOnly,
      rewardFilterMode
    );
    const rolloutState = buildRolloutState(runDetail, selectedRunSummary, loadingDetail);
    const payloadState = buildPayloadState(runDetail, loadingDetail);
    const refreshLabel = refreshInFlight
      ? "Syncing telemetry…"
      : config
      ? `Auto refresh every ${Math.max(2, Number(config.refresh_seconds) || 10)}s`
      : "Telemetry idle";
    const liveTone = globalError
      ? "failed"
      : experiment && Object.keys(experiment).length
      ? experimentTone(experiment)
      : "neutral";
    const liveTitle = globalError
      ? globalError
      : experiment && Object.keys(experiment).length
      ? summarizeCounts(asObject(experiment.status_counts)) ||
        `${experiment.active_run_count || 0} active runs`
      : "Waiting for telemetry";
    const experimentOptions = experiments.length
      ? experiments
      : selectedExperimentId
      ? [{ experiment_id: selectedExperimentId, run_count: 0, loading: true }]
      : [];

    useEffect(() => {
      detailCacheRef.current = runDetailCache;
    }, [runDetailCache]);

    useEffect(() => {
      detailLoadingRef.current = runDetailLoadingByKey;
    }, [runDetailLoadingByKey]);

    useEffect(() => {
      detailErrorRef.current = runDetailErrorByKey;
    }, [runDetailErrorByKey]);

    useEffect(() => {
      void refreshDashboard({
        resetSelection: true,
        preferredExperimentId: initialRouteRef.current.experimentId || null,
        preferredRunKey: routeRunKeyForExperiment(
          initialRouteRef.current,
          initialRouteRef.current.experimentId
        ),
      });
      return () => {
        if (intervalRef.current !== null) {
          window.clearInterval(intervalRef.current);
        }
      };
    }, []);

    useEffect(() => {
      if (config == null) {
        return;
      }
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
      }
      intervalRef.current = window.setInterval(() => {
        void refreshDashboard({ resetSelection: false });
      }, Math.max(2, Number(config.refresh_seconds) || 10) * 1000);
      return () => {
        if (intervalRef.current !== null) {
          window.clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }, [config, selectedExperimentId, selectedRunKey]);

    useEffect(() => {
      if (selectedRunSummary && selectedRunIdentity !== selectedRunKey) {
        setSelectedRunKey(selectedRunIdentity);
      }
      if (!selectedRunSummary && selectedRunKey) {
        setSelectedRunKey(null);
      }
    }, [selectedRunIdentity, selectedRunKey, selectedRunSummary]);

    useEffect(() => {
      if (
        activePanel !== selectedPanel &&
        !shouldHoldRequestedPanel(selectedPanel, selectedRunSummary)
      ) {
        setSelectedPanel(activePanel);
      }
    }, [activePanel, selectedPanel]);

    useEffect(() => {
      writeRoute({
        experimentId: selectedExperimentId,
        runSummary: selectedRunSummary,
        panel: activePanel,
      });
    }, [selectedExperimentId, selectedRunIdentity, activePanel]);

    useEffect(() => {
      updateDocumentTitle(selectedExperimentId, selectedRunSummary);
    }, [selectedExperimentId, selectedRunIdentity]);

    async function refreshDashboard(options) {
      const settings = options || {};
      const requestId = ++refreshTokenRef.current;
      setRefreshInFlight(true);
      setGlobalError("");

      try {
        const nextConfig = config || (await fetchJson("/api/config"));
        if (requestId !== refreshTokenRef.current) {
          return;
        }
        setConfig(nextConfig);

        let experimentsPayload = null;
        try {
          experimentsPayload = await fetchJson("/api/experiments");
          if (requestId !== refreshTokenRef.current) {
            return;
          }
          setExperiments(
            Array.isArray(experimentsPayload.experiments)
              ? experimentsPayload.experiments
              : []
          );
          setExperimentsLoaded(true);
        } catch (_error) {
          setExperimentsLoaded(true);
        }

        const targetExperimentId =
          settings.preferredExperimentId ||
          selectedExperimentId ||
          initialRouteRef.current.experimentId ||
          nextConfig.default_experiment_id ||
          firstExperimentId(experimentsPayload) ||
          firstExperimentId({ experiments: experiments });

        if (!targetExperimentId) {
          setSelectedExperimentId(null);
          setExperimentSnapshot(null);
          setSnapshotLoading(false);
          return;
        }

        setSnapshotLoading(true);
        const snapshot = await fetchJson(
          `/api/experiment?experiment_id=${encodeURIComponent(targetExperimentId)}`
        );
        if (requestId !== refreshTokenRef.current) {
          return;
        }

        const nextRuns = Array.isArray(snapshot.runs) ? snapshot.runs : [];
        const nextRun = pickRunSummary(nextRuns, [
          settings.preferredRunKey,
          settings.resetSelection ? null : selectedRunKey,
          routeRunKeyForExperiment(initialRouteRef.current, targetExperimentId),
        ]);

        setSelectedExperimentId(targetExperimentId);
        setExperimentSnapshot(snapshot);
        setSnapshotLoading(false);
        setSelectedRunKey(nextRun ? buildRunKey(nextRun) : null);
        setShowFailuresOnly(false);
        setRewardFilterMode("any");

        if (nextRun) {
          void loadRunDetail(nextRun, targetExperimentId, requestId, {
            background: false,
            prefetch: false,
          });
          void prefetchRunDetails(nextRuns, targetExperimentId, nextRun);
        }
      } catch (error) {
        if (requestId !== refreshTokenRef.current) {
          return;
        }
        setSnapshotLoading(false);
        setGlobalError(normalizeErrorMessage(error));
      } finally {
        if (requestId === refreshTokenRef.current) {
          setRefreshInFlight(false);
        }
      }
    }

    async function prefetchRunDetails(runsToWarm, experimentId, selectedRun) {
      const targetExperimentId = experimentId || selectedExperimentId;
      const items = (Array.isArray(runsToWarm) ? runsToWarm : [])
        .filter((run) => buildRunKey(run) !== buildRunKey(selectedRun))
        .slice(0, 6);

      for (const [index, run] of items.entries()) {
        window.setTimeout(() => {
          void loadRunDetail(run, targetExperimentId, undefined, {
            background: true,
            prefetch: true,
          });
        }, 180 * (index + 1));
      }
    }

    async function loadRunDetail(runSummary, experimentId, parentRequestId, options) {
      const settings = options || {};
      const targetExperimentId = experimentId || selectedExperimentId;
      if (!targetExperimentId || !runSummary) {
        return;
      }
      const cacheKey = buildDetailCacheKey(targetExperimentId, runSummary);
      const cachedDetail = detailCacheRef.current[cacheKey] || null;

      if (
        settings.prefetch &&
        (cachedDetail || detailLoadingRef.current[cacheKey])
      ) {
        return;
      }

      const runRequestId = settings.prefetch ? null : ++runTokenRef.current;

      setRunDetailLoadingByKey((current) =>
        Object.assign({}, current, { [cacheKey]: true })
      );
      setRunDetailErrorByKey((current) =>
        Object.assign({}, current, { [cacheKey]: "" })
      );

      try {
        const detail = await fetchJson(
          `/api/run?experiment_id=${encodeURIComponent(
            targetExperimentId
          )}&phase=${encodeURIComponent(
            String(runSummary.phase || "")
          )}&run_name=${encodeURIComponent(String(runSummary.run_name || ""))}`
        );
        if (
          (runRequestId && runRequestId !== runTokenRef.current) ||
          (parentRequestId && parentRequestId !== refreshTokenRef.current)
        ) {
          return;
        }
        setRunDetailCache((current) => Object.assign({}, current, { [cacheKey]: detail }));
        setRunDetailErrorByKey((current) =>
          Object.assign({}, current, { [cacheKey]: "" })
        );
      } catch (error) {
        if (runRequestId && runRequestId !== runTokenRef.current) {
          return;
        }
        setRunDetailErrorByKey((current) =>
          Object.assign({}, current, {
            [cacheKey]: normalizeErrorMessage(error),
          })
        );
      } finally {
        if (!runRequestId || runRequestId === runTokenRef.current) {
          setRunDetailLoadingByKey((current) =>
            Object.assign({}, current, { [cacheKey]: false })
          );
        }
      }
    }

    function handleRefresh() {
      void refreshDashboard({
        resetSelection: false,
        preferredExperimentId: selectedExperimentId,
        preferredRunKey: selectedRunIdentity,
      });
    }

    function handleExperimentSelect(event) {
      const nextExperimentId = trimOrNull(event.target.value);
      startTransition(() => {
        setSelectedExperimentId(nextExperimentId);
        setSelectedRunKey(null);
        setSelectedPanel("overview");
        setShowFailuresOnly(false);
        setRewardFilterMode("any");
      });
      void refreshDashboard({
        resetSelection: true,
        preferredExperimentId: nextExperimentId,
        preferredRunKey: null,
      });
    }

    function handleRunSelect(runSummary) {
      const nextKey = buildRunKey(runSummary);
      startTransition(() => {
        setSelectedRunKey(nextKey);
        setSelectedPanel(resolvePanelForRun(runSummary, activePanel));
        setShowFailuresOnly(false);
        setRewardFilterMode("any");
      });
      void loadRunDetail(runSummary, selectedExperimentId);
    }

    function handlePanelSelect(panel) {
      startTransition(() => {
        setSelectedPanel(panel);
      });
      if (panel !== "overview" && selectedRunSummary && !runDetail && !runDetailLoading) {
        void loadRunDetail(selectedRunSummary, selectedExperimentId);
      }
    }

    return html`
      <div className="app-shell">
        <${TopBar}
          options=${experimentOptions}
          selectedExperimentId=${selectedExperimentId}
          refreshLabel=${refreshLabel}
          refreshInFlight=${refreshInFlight}
          onRefresh=${handleRefresh}
          onExperimentChange=${handleExperimentSelect}
          experimentsLoaded=${experimentsLoaded}
        />
        <main className="layout">
          <${Sidebar}
            projectItems=${projectItems}
            liveTone=${liveTone}
            liveTitle=${liveTitle}
            runs=${runs}
            selectedRunKey=${selectedRunIdentity}
            onRunSelect=${handleRunSelect}
          />
          <section className="main-panel">
            <${HeroGrid} cards=${heroCards} loading=${snapshotLoading} />
            <${DetailShell}
              summary=${selectedRunSummary}
              status=${detailStatus}
              subtitle=${selectedRunSummary ? summarizeRunSubtitle(selectedRunSummary) : "Pick a run from the left to inspect it."}
              facts=${detailFacts}
              actions=${detailActions}
              panel=${activePanel}
              panels=${availablePanels(selectedRunSummary, runDetail)}
              onSelectPanel=${handlePanelSelect}
            />
            <${OverviewPanel}
              metaItems=${detailMetaItems}
              metrics=${detailMetrics}
              stageNote=${detailStageNote}
              emptyMessage=${selectedRunSummary ? detailMetricsEmptyState(selectedRunSummary, overviewMetrics) : "Run-level metrics will appear here."}
              error=${runDetailError}
            />
            ${activePanel === "history"
              ? html`<${HistoryPanel} state=${historyState} />`
              : null}
            ${activePanel === "samples" && samplesVisible(selectedRunSummary, runDetail)
              ? html`<${SamplesPanel}
                  state=${samplesState}
                  showFailuresOnly=${showFailuresOnly}
                  rewardFilterMode=${rewardFilterMode}
                  onShowAll=${() => setShowFailuresOnly(false)}
                  onShowFailures=${() => setShowFailuresOnly(true)}
                  onRewardFilter=${(mode) => setRewardFilterMode(mode)}
                />`
              : null}
            ${activePanel === "rollouts" && isRLRun(selectedRunSummary)
              ? html`<${RolloutsPanel} state=${rolloutState} />`
              : null}
            ${activePanel === "payload"
              ? html`<${PayloadPanel} state=${payloadState} />`
              : null}
          </section>
        </main>
      </div>
    `;
  }

  function TopBar(props) {
    const options = props.options.length
      ? props.options
      : [{ experiment_id: "", loading: true, run_count: 0 }];
    return html`
      <header className="topbar">
        <div className="title-stack">
          <p className="eyebrow">Telemetry Workspace</p>
          <h1>Tenyson Dashboard</h1>
          <p className="subtitle-copy">
            Browse recent experiment runs without waiting for the whole page to catch up.
          </p>
        </div>
        <div className="toolbar">
          <label className="select-wrap">
            <span>Experiment</span>
            <select
              value=${props.selectedExperimentId || ""}
              onChange=${props.onExperimentChange}
              disabled=${!props.options.length && !props.experimentsLoaded}
            >
              ${options.map((item) => {
                const value = String(item.experiment_id || "");
                return html`<option key=${value || "loading"} value=${value}>
                  ${experimentOptionLabel(item)}
                </option>`;
              })}
            </select>
          </label>
          <span className="refresh-label">${props.refreshLabel}</span>
          <button
            className="ghost-button"
            type="button"
            onClick=${props.onRefresh}
            disabled=${props.refreshInFlight}
          >
            ${props.refreshInFlight ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </header>
    `;
  }

  function Sidebar(props) {
    return html`
      <aside className="sidebar card-panel">
        <section className="sidebar-section">
          <div className="section-header">
            <h2>Project</h2>
            <span className=${`status-dot ${props.liveTone}`} title=${props.liveTitle}></span>
          </div>
          <div className="meta-stack">
            ${props.projectItems.map((item) => html`<${MetaCard} key=${item.label} item=${item} />`)}
          </div>
        </section>
        <section className="sidebar-section">
          <div className="section-header">
            <h2>Runs</h2>
            <span className="count-pill">${props.runs.length}</span>
          </div>
          <div className="run-list">
            ${props.runs.length
              ? props.runs.map(
                  (run) => html`<${RunCard}
                    key=${buildRunKey(run)}
                    run=${run}
                    selected=${buildRunKey(run) === props.selectedRunKey}
                    onSelect=${props.onRunSelect}
                  />`
                )
              : html`<div className="empty-state">No runs available.</div>`}
          </div>
        </section>
      </aside>
    `;
  }

  function HeroGrid(props) {
    if (!props.cards.length && props.loading) {
      return html`<div className="hero-grid">
        <div className="empty-state">Loading experiment summary...</div>
      </div>`;
    }
    if (!props.cards.length) {
      return html`<div className="hero-grid">
        <div className="empty-state">No experiment summary loaded.</div>
      </div>`;
    }
    return html`
      <div className="hero-grid">
        ${props.cards.map(
          (card) => html`<article key=${card.label} className="hero-card">
            <div className="metric-label">${card.label}</div>
            <div className="hero-value">${formatMetricValue(card.value, card.label)}</div>
          </article>`
        )}
      </div>
    `;
  }

  function DetailShell(props) {
    const title = props.summary
      ? props.summary.display_name || props.summary.run_name
      : "Select a run";
    return html`
      <section className="card-panel detail-shell">
        <div className="detail-header">
          <div className="detail-title-wrap">
            <p className="eyebrow detail-eyebrow">Selected Run</p>
            <div className="detail-title-row">
              <h2>${title}</h2>
              <${StatusPill} status=${props.status} />
            </div>
            <p className="muted-copy detail-subtitle">${props.subtitle}</p>
          </div>
          <div className="tab-row" role="tablist" aria-label="Run detail sections">
            ${props.panels.map(
              (panel) => html`<button
                key=${panel}
                className=${`tab-button ${props.panel === panel ? "active" : ""}`}
                type="button"
                role="tab"
                aria-selected=${String(props.panel === panel)}
                onClick=${() => props.onSelectPanel(panel)}
              >
                ${panelLabel(panel)}
              </button>`
            )}
          </div>
        </div>
        <div className="fact-row">
          ${props.facts.map(
            (item) => html`<div key=${item.label} className="fact-chip">
              <div className="fact-label">${item.label}</div>
              <div className="fact-value">${item.value}</div>
            </div>`
          )}
        </div>
        ${props.actions.length
          ? html`<div className="action-row">
              ${props.actions.map(
                (action) => html`<a
                  key=${action.label}
                  className="action-link"
                  href=${action.href}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  ${action.label}
                </a>`
              )}
            </div>`
          : null}
      </section>
    `;
  }

  function OverviewPanel(props) {
    const metaChildren = props.metaItems.length
      ? props.metaItems.map((item, index) =>
          React.createElement(MetaCard, {
            key: `meta-${item.label}-${index}`,
            item: item,
          })
        )
      : [
          React.createElement(
            "div",
            { key: "meta-empty", className: "empty-state" },
            props.emptyMessage
          ),
        ];

    const metricChildren = props.metrics.length
      ? props.metrics.map(([key, value], index) =>
          React.createElement(
            "article",
            { key: `metric-${key}-${index}`, className: "metric-card" },
            React.createElement("div", { className: "metric-label" }, prettifyKey(key)),
            React.createElement(
              "div",
              { className: "metric-value" },
              formatMetricValue(value, key)
            )
          )
        )
      : [
          React.createElement(
            "div",
            { key: "metric-empty", className: "empty-state" },
            props.emptyMessage
          ),
        ];

    if (props.error) {
      return React.createElement(
        "section",
        { className: "card-panel tab-panel detail-panel active" },
        React.createElement(
          "div",
          { className: "section-header" },
          React.createElement("h2", null, "Run Overview"),
          React.createElement("span", { className: "muted-copy" }, props.stageNote)
        ),
        React.createElement("div", { className: "empty-state" }, props.error)
      );
    }

    return React.createElement(
      "section",
      { className: "card-panel tab-panel detail-panel active" },
      React.createElement(
        "div",
        { className: "section-header" },
        React.createElement("h2", null, "Run Overview"),
        React.createElement("span", { className: "muted-copy" }, props.stageNote)
      ),
      React.createElement("div", { className: "meta-grid" }, metaChildren),
      React.createElement("div", { className: "metric-grid" }, metricChildren)
    );
  }

  function HistoryPanel(props) {
    return html`
      <section className="card-panel tab-panel detail-panel active">
        <div className="section-header">
          <h2>Metric History</h2>
          <span className="muted-copy">${props.state.summary}</span>
        </div>
        <div className="chart-grid">
          ${props.state.loading
            ? html`<div className="empty-state">Loading chart data for this run in the background.</div>`
            : props.state.error
            ? html`<div className="empty-state">${props.state.error}</div>`
            : props.state.cards.length
            ? props.state.cards.map(
                (card, index) => html`<${ChartCard}
                  key=${card.key}
                  card=${card}
                  color=${chartColors[index % chartColors.length]}
                />`
              )
            : html`<div className="empty-state">No metric history available yet.</div>`}
        </div>
      </section>
    `;
  }

  function SamplesPanel(props) {
    const rewardButtons = [
      { value: "any", label: "Any reward", count: props.state.rewardCounts.available || props.state.totalCount },
      { value: "positive", label: "Positive only", count: props.state.rewardCounts.positive },
      { value: "zero", label: "Zero only", count: props.state.rewardCounts.zero },
      { value: "negative", label: "Negative only", count: props.state.rewardCounts.negative },
      { value: "nonpositive", label: "Zero or below", count: props.state.rewardCounts.nonpositive },
    ];

    return html`
      <section className="card-panel tab-panel eval-panel active">
        <div className="section-header">
          <h2>Eval Samples</h2>
          <div className="toolbar-inline">
            <button
              className=${`small-button ${!props.showFailuresOnly ? "active" : ""}`}
              type="button"
              onClick=${props.onShowAll}
              disabled=${!props.state.enabled}
            >
              ${props.state.enabled ? `All (${props.state.totalCount})` : "All"}
            </button>
            <button
              className=${`small-button ${props.showFailuresOnly ? "active" : ""}`}
              type="button"
              onClick=${props.onShowFailures}
              disabled=${!props.state.enabled}
            >
              ${props.state.enabled ? `Failures (${props.state.failureCount})` : "Failures"}
            </button>
            <div className="filter-pill-group" aria-label="Reward filter">
              <span>Reward</span>
              <div className="filter-pill-row">
                ${rewardButtons.map(
                  (item) => html`<button
                    key=${item.value}
                    className=${`small-button reward-filter-button ${
                      props.rewardFilterMode === item.value ? "active" : ""
                    }`}
                    type="button"
                    onClick=${() => props.onRewardFilter(item.value)}
                    disabled=${!props.state.enabled || (!props.state.rewardCounts.available && item.value !== "any")}
                  >
                    ${item.label} (${item.count || 0})
                  </button>`
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="metric-grid compact">
          ${props.state.metrics.length
            ? props.state.metrics.map(
                ([key, value]) => html`<article key=${key} className="metric-card">
                  <div className="metric-label">${prettifyKey(key)}</div>
                  <div className="metric-value">${formatMetricValue(value, key)}</div>
                </article>`
              )
            : html`<div className="empty-state">${props.state.message}</div>`}
        </div>
        <div className="table-wrap">
          <table className="eval-table">
            <thead>
              <tr>
                ${props.state.columns.length
                  ? props.state.columns.map(
                      (column, index) => html`<th key=${`head-${column.key}-${index}`}>
                        ${column.label}
                      </th>`
                    )
                  : html`<th>Status</th><th>Prompt</th><th>Completion</th>`}
              </tr>
            </thead>
            <tbody>
              ${props.state.rows.length
                ? props.state.rows.map(
                    (row, index) => html`<tr key=${row.key || index}>
                      ${props.state.columns.map(
                        (column, columnIndex) => html`<td
                          key=${`cell-${row.key || index}-${column.key}-${columnIndex}`}
                          className=${column.className || ""}
                        >
                          ${renderColumnContent(column, row)}
                        </td>`
                      )}
                    </tr>`
                  )
                : html`<tr>
                    <td colSpan=${Math.max(1, props.state.columns.length || 3)}>
                      <div className="empty-state">${props.state.message}</div>
                    </td>
                  </tr>`}
            </tbody>
          </table>
        </div>
      </section>
    `;
  }

  function RolloutsPanel(props) {
    return html`
      <section className="card-panel tab-panel eval-panel active">
        <div className="section-header">
          <h2>RL Rollouts</h2>
          <span className="muted-copy">${props.state.summary}</span>
        </div>
        ${props.state.note
          ? html`<div className="empty-state">${props.state.note}</div>`
          : null}
        <div className="table-wrap">
          <table className="eval-table">
            <thead>
              <tr>
                ${props.state.columns.map(
                  (column, index) => html`<th key=${`rollout-head-${column.key}-${index}`}>
                    ${column.label}
                  </th>`
                )}
              </tr>
            </thead>
            <tbody>
              ${props.state.rows.length
                ? props.state.rows.map(
                    (row, index) => html`<tr key=${row.key || index}>
                      ${props.state.columns.map(
                        (column, columnIndex) => html`<td
                          key=${`rollout-cell-${row.key || index}-${column.key}-${columnIndex}`}
                          className=${column.className || ""}
                        >
                          ${renderColumnContent(column, row)}
                        </td>`
                      )}
                    </tr>`
                  )
                : html`<tr>
                    <td colSpan=${Math.max(1, props.state.columns.length || 1)}>
                      <div className="empty-state">${props.state.message}</div>
                    </td>
                  </tr>`}
            </tbody>
          </table>
        </div>
      </section>
    `;
  }

  function PayloadPanel(props) {
    return html`
      <section className="card-panel tab-panel raw-panel active">
        <div className="section-header">
          <h2>Canonical Payload</h2>
          <span className="muted-copy">Raw telemetry for debugging and review</span>
        </div>
        <pre className="raw-code">${props.state.text}</pre>
      </section>
    `;
  }

  function MetaCard(props) {
    return html`
      <div className="meta-item">
        <div className="meta-label">${props.item.label}</div>
        <div className="meta-value">
          ${props.item.href
            ? html`<a href=${props.item.href} target="_blank" rel="noreferrer noopener">
                ${props.item.linkLabel || props.item.value}
              </a>`
            : props.item.value}
        </div>
      </div>
    `;
  }

  function RunCard(props) {
    const run = props.run;
    return html`
      <button
        type="button"
        className=${`run-card ${props.selected ? "selected" : ""}`}
        onClick=${() => props.onSelect(run)}
      >
        <div className="run-card-top">
          <div className="run-title">${run.display_name || run.run_name}</div>
          <${StatusPill} status=${run.status} />
        </div>
        <div className="run-phase">
          ${[String(run.phase || "").toUpperCase(), run.provider].filter(Boolean).join(" · ")}
        </div>
        <div className="run-card-bottom">
          <span>${pickRunPreview(run) || runPreviewFallbackText(run)}</span>
          <span>
            ${run.heartbeat_at
              ? formatRelativeTime(run.heartbeat_at)
              : run.created_at
              ? formatTimestamp(run.created_at)
              : "No timestamp"}
          </span>
        </div>
      </button>
    `;
  }

  function StatusPill(props) {
    return html`<span className=${`status-pill ${statusTone(props.status)}`}>
      ${String(props.status || "unknown")}
    </span>`;
  }

  function ChartCard(props) {
    const values = props.card.values;
    const key = props.card.key;
    const latestValue = values.length ? values[values.length - 1].y : null;
    const color = props.color;

    if (!values.length) {
      return html`<article className="chart-card">
        <div className="chart-title-row">
          <h3>${prettifyKey(key)}</h3>
          <span className="muted-copy">n/a</span>
        </div>
        <div className="history-empty">No values logged.</div>
      </article>`;
    }

    if (values.length === 1) {
      return html`<article className="chart-card single-point">
        <div className="chart-title-row">
          <h3>${prettifyKey(key)}</h3>
          <span className="muted-copy">${formatMetricValue(latestValue, key)}</span>
        </div>
        <div className="chart-single-value">${formatMetricValue(values[0].y, key)}</div>
        <div className="chart-note muted-copy">
          Only one telemetry point is available so there is no trend line yet.
        </div>
        <div className="chart-footer">
          <span>min ${formatMetricValue(values[0].y, key)}</span>
          <span>max ${formatMetricValue(values[0].y, key)}</span>
          <span>1 point</span>
        </div>
      </article>`;
    }

    const geometry = buildChartGeometry(values);
    return html`
      <article className="chart-card">
        <div className="chart-title-row">
          <h3>${prettifyKey(key)}</h3>
          <span className="muted-copy">${formatMetricValue(latestValue, key)}</span>
        </div>
        <svg className="chart-spark" viewBox="0 0 260 170" role="img" aria-label=${`${prettifyKey(key)} history`}>
          ${[0, 1, 2, 3].map((index) => {
            const y = geometry.padding + (geometry.innerHeight * index) / 3;
            return html`<line
              key=${`guide-${index}`}
              x1=${geometry.padding}
              x2=${260 - geometry.padding}
              y1=${y.toFixed(2)}
              y2=${y.toFixed(2)}
              stroke="rgba(28, 45, 40, 0.08)"
              strokeWidth="1"
            />`;
          })}
          <path d=${geometry.areaPath} fill=${withAlpha(color, 0.16)} />
          <path
            d=${geometry.linePath}
            fill="none"
            stroke=${color}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle
            cx=${geometry.last[0].toFixed(2)}
            cy=${geometry.last[1].toFixed(2)}
            r="4"
            fill=${color}
            stroke="#fffdfa"
            strokeWidth="2"
          />
        </svg>
        <div className="chart-footer">
          <span>min ${formatMetricValue(geometry.minY, key)}</span>
          <span>max ${formatMetricValue(geometry.maxY, key)}</span>
          <span>${values.length} points</span>
        </div>
      </article>
    `;
  }

  function ExpandableText(props) {
    const fullText = String(props.text || "");
    const preview = summarizeText(fullText, props.previewLength || 200);
    if (!fullText.trim()) {
      return html`<div className=${`copy-inline muted-copy ${props.className || ""}`}>
        n/a
      </div>`;
    }
    if (preview === fullText) {
      return html`<div className=${`copy-inline ${props.className || ""}`}>${fullText}</div>`;
    }
    return html`
      <div className="expand-shell">
        <div className=${`${props.className || ""} copy-preview`}>${preview}</div>
        <details className="expand-panel">
          <summary className="expand-toggle">Show full</summary>
          <div className=${props.className || ""}>${fullText}</div>
        </details>
      </div>
    `;
  }

  function renderColumnContent(column, row) {
    if (column.type === "status") {
      return html`<div className="status-shell">
        <span className=${`eval-status ${row.outcome.passed ? "pass" : "fail"}`}>
          ${row.outcome.passed ? "Pass" : "Needs review"}
        </span>
        ${row.outcome.reasons.length
          ? html`<div className="muted-copy status-note">
              ${row.outcome.reasons.join(", ")}
            </div>`
          : null}
      </div>`;
    }

    if (column.type === "reward") {
      if (!Number.isFinite(Number(row.reward))) {
        return html`<div className="muted-copy">n/a</div>`;
      }
      const breakdown = summarizeRewardBreakdown(row.raw);
      return html`<div className="reward-shell">
        <span className=${`reward-chip ${rewardTone(row.reward)}`}>
          ${formatSignedValue(row.reward)}
        </span>
        ${breakdown
          ? html`<div className="muted-copy reward-breakdown">${breakdown}</div>`
          : null}
      </div>`;
    }

    if (column.type === "text") {
      return row[column.key] == null || row[column.key] === ""
        ? "n/a"
        : String(row[column.key]);
    }

    if (column.type === "longtext") {
      return html`<${ExpandableText}
        className=${column.className || "copy-inline"}
        text=${row[column.key] || ""}
        previewLength=${column.previewLength || 240}
      />`;
    }

    if (column.type === "completion") {
      return html`<${ExpandableText}
        className="completion-copy"
        text=${row.completion || ""}
        previewLength=${column.previewLength}
      />`;
    }

    if (column.type === "prompt") {
      return html`<${ExpandableText}
        className="prompt-copy"
        text=${row.prompt || ""}
        previewLength=${column.previewLength}
      />`;
    }

    return "n/a";
  }

  function buildProjectSummaryItems(config, snapshot, errorMessage) {
    if (errorMessage && !snapshot) {
      return [
        { label: "Backend", value: config ? config.backend_ref : "n/a" },
        { label: "Status", value: errorMessage },
      ];
    }

    const experiment = asObject(snapshot && snapshot.experiment);
    if (Object.keys(experiment).length) {
      const runs = Array.isArray(snapshot.runs) ? snapshot.runs : [];
      return [
        { label: "Experiment", value: experiment.experiment_id },
        { label: "Backend", value: experiment.backend_ref || (config ? config.backend_ref : "n/a") },
        {
          label: "Project",
          value: "Open W&B project",
          href: experiment.project_url || (config ? config.project_url : ""),
          linkLabel: "Open W&B project",
        },
        { label: "Auto refresh", value: `${config ? config.refresh_seconds : 10}s` },
        { label: "Phase mix", value: summarizeCounts(countBy(runs, "phase")) || "No runs yet" },
        {
          label: "Status mix",
          value: summarizeCounts(asObject(experiment.status_counts)) || "No statuses yet",
        },
        {
          label: "Latest heartbeat",
          value: experiment.latest_heartbeat_at
            ? `${formatTimestamp(experiment.latest_heartbeat_at)} (${formatRelativeTime(
                experiment.latest_heartbeat_at
              )})`
            : "No heartbeat yet",
        },
        { label: "Refreshed", value: formatTimestamp(experiment.refreshed_at) },
      ];
    }

    if (config) {
      return [
        { label: "Backend", value: config.backend_ref },
        {
          label: "Project",
          value: "Open W&B project",
          href: config.project_url,
          linkLabel: "Open W&B project",
        },
        { label: "Auto refresh", value: `${config.refresh_seconds}s` },
      ];
    }

    return [{ label: "Status", value: "Loading telemetry workspace..." }];
  }

  function buildHeroCards(snapshot) {
    const experiment = asObject(snapshot && snapshot.experiment);
    if (!Object.keys(experiment).length) {
      return [];
    }
    const runs = Array.isArray(snapshot.runs) ? snapshot.runs : [];
    const phaseCounts = countBy(runs, "phase");
    const statuses = asObject(experiment.status_counts);
    return [
      { label: "Total runs", value: experiment.run_count || 0 },
      { label: "Active now", value: experiment.active_run_count || 0 },
      { label: "Succeeded", value: statuses.success || 0 },
      { label: "Failed", value: statuses.failed || 0 },
      { label: "Partial", value: (statuses.partial || 0) + (statuses.stopped || 0) },
      { label: "Phases", value: Object.keys(phaseCounts).length || 0 },
    ];
  }

  function buildRunActions(summary, config) {
    const actions = [];
    if (summary.wandb_url) {
      actions.push({ label: "Open W&B run", href: summary.wandb_url });
    }
    if (summary.hf_repo_id) {
      actions.push({
        label: "Open HF artifact",
        href: buildHfUrl(summary.hf_repo_id, summary.hf_revision),
      });
    }
    if (config && config.project_url) {
      actions.push({ label: "Open project", href: config.project_url });
    }
    return actions;
  }

  function buildRunMetaItems(summary, jobResultPayload) {
    const items = [
      { label: "Phase", value: String(summary.phase || "unknown").toUpperCase() },
      { label: "Provider", value: summary.provider || "n/a" },
      {
        label: "Runtime",
        value: Number.isFinite(Number(summary.total_time_seconds))
          ? formatDuration(Number(summary.total_time_seconds))
          : "n/a",
      },
      { label: "Created", value: summary.created_at ? formatTimestamp(summary.created_at) : "n/a" },
      {
        label: "Last heartbeat",
        value: summary.heartbeat_at
          ? `${formatTimestamp(summary.heartbeat_at)} (${formatRelativeTime(summary.heartbeat_at)})`
          : "n/a",
      },
    ];

    if (hasMeaningfulSamples(summary)) {
      items.push({
        label: "Samples",
        value: summarizeSamples(summary.processed_samples, summary.expected_samples),
      });
    }
    if (summary.stop_requested) {
      items.push({
        label: "Stop requested",
        value: summary.stop_requested_at
          ? `Yes at ${formatTimestamp(summary.stop_requested_at)}`
          : "Yes",
      });
    }
    const failureReason = summary.failure_reason || jobResultPayload.failure_reason;
    if (failureReason) {
      items.push({ label: "Failure reason", value: failureReason });
    }
    return items;
  }

  function summarizeRunSubtitle(summary) {
    const parts = [
      String(summary.phase || "").toUpperCase(),
      summary.provider,
      pickRunPreview(summary) || runPreviewFallbackText(summary),
    ].filter(Boolean);
    return parts.length ? parts.join(" · ") : "Pick a run from the left to inspect it.";
  }

  function buildRunFacts(summary, metrics, loadingDetail) {
    const items = [
      { label: "Phase", value: String(summary.phase || "unknown").toUpperCase() },
      { label: "Provider", value: summary.provider || "n/a" },
    ];
    if (hasMeaningfulSamples(summary)) {
      items.push({
        label: "Samples",
        value: summarizeSamples(summary.processed_samples, summary.expected_samples),
      });
    }

    orderedMetricEntries(metrics, summary.phase)
      .filter((entry) => !isNoisyOverviewMetricKey(entry[0]))
      .slice(0, hasMeaningfulSamples(summary) ? 1 : 2)
      .forEach(([key, value]) => {
        items.push({ label: prettifyKey(key), value: formatMetricValue(value, key) });
      });

    if (items.length < 4) {
      items.push({
        label: "Runtime",
        value: Number.isFinite(Number(summary.total_time_seconds))
          ? formatDuration(Number(summary.total_time_seconds))
          : "n/a",
      });
    }

    items.push({ label: "Detail", value: loadingDetail ? "Syncing" : "Ready" });
    return items;
  }

  function availablePanels(summary, detail) {
    const panels = ["overview", "history"];
    if (samplesVisible(summary, detail)) {
      panels.push("samples");
    }
    if (rolloutsVisible(summary, detail)) {
      panels.push("rollouts");
    }
    panels.push("payload");
    return panels;
  }

  function samplesVisible(summary, detail) {
    return (
      isEvalLikeRun(summary) ||
      Boolean(
        detailDetailedResults(detail).length && isEvalLikeRun(summary)
      )
    );
  }

  function rolloutsVisible(summary, detail) {
    return isRLRun(summary);
  }

  function buildHistoryState(detail, summary, loadingDetail) {
    if (loadingDetail && !detail) {
      return {
        loading: true,
        error: "",
        summary: "Loading metric history…",
        cards: [],
      };
    }
    if (!detail) {
      return {
        loading: false,
        error: "",
        summary: "No series loaded",
        cards: [],
      };
    }
    const history = asObject(detail.history);
    const rows = Array.isArray(history.rows) ? history.rows : [];
    const keys = Array.isArray(history.keys) ? history.keys : [];
    const selectedKeys = pickChartKeys(history, summary);
    if (!rows.length || !selectedKeys.length) {
      return {
        loading: false,
        error: "",
        summary: "No numeric history logged for this run",
        cards: [],
      };
    }
    return {
      loading: false,
      error: "",
      summary: `Showing ${selectedKeys.length} key series from ${keys.length} logged metrics across ${pluralize(
        rows.length,
        "point"
      )}`,
      cards: selectedKeys.map((key) => ({
        key: key,
        values: rows
          .map((row, index) => ({
            x: Number.isFinite(Number(row.step)) ? Number(row.step) : index,
            y: Number(row[key]),
          }))
          .filter((item) => Number.isFinite(item.y)),
      })),
    };
  }

  function buildSamplesState(detail, summary, loadingDetail, showFailuresOnly, rewardFilterMode) {
    if (loadingDetail && !detail) {
      return {
        enabled: false,
        totalCount: 0,
        failureCount: 0,
        rewardCounts: {
          available: 0,
          positive: 0,
          zero: 0,
          negative: 0,
          nonpositive: 0,
        },
        metrics: [],
        columns: [],
        rows: [],
        message: isEvalLikeRun(summary)
          ? "Loading eval samples…"
          : "This run does not expose eval sample rows.",
      };
    }

    const resultPayload = preferredResultPayload(detail);
    const detailedRows = detailDetailedResults(detail);
    const normalizedRows = detailedRows.map(normalizeEvalRow);
    const metrics = Object.assign(
      {},
      asObject(resultPayload.metrics),
      deriveRewardMetrics(normalizedRows)
    );
    const rewardCounts = countRewardBuckets(normalizedRows);
    const failureCount = normalizedRows.filter((row) => !row.outcome.passed).length;

    if (!normalizedRows.length) {
      return {
        enabled: false,
        totalCount: 0,
        failureCount: 0,
        rewardCounts: rewardCounts,
        metrics: orderedMetricEntries(metrics, summary && summary.phase),
        columns: [],
        rows: [],
        message: isEvalLikeRun(summary)
          ? "No detailed sample rows were logged for this run."
          : "This run does not expose eval sample rows.",
      };
    }

    const filteredRows = normalizedRows
      .filter((row) => {
        if (showFailuresOnly && row.outcome.passed) {
          return false;
        }
        return matchesRewardFilter(row.reward, rewardFilterMode);
      })
      .map((row, index) => normalizeEvalDisplayRow(row, index));

    return {
      enabled: true,
      totalCount: normalizedRows.length,
      failureCount: failureCount,
      rewardCounts: rewardCounts,
      metrics: orderedMetricEntries(metrics, summary && summary.phase),
      columns: buildEvalColumns(filteredRows, rewardCounts.available > 0),
      rows: filteredRows,
      message:
        filteredRows.length === 0
          ? showFailuresOnly || rewardFilterMode !== "any"
            ? "No samples match the current filters."
            : "All logged samples passed the current checks."
          : "",
    };
  }

  function buildRolloutState(detail, summary, loadingDetail) {
    if (loadingDetail && !detail) {
      return {
        columns: [],
        rows: [],
        summary: "Loading rollout telemetry…",
        message: "Loading rollout telemetry…",
        note: "",
      };
    }

    const resultPayload = preferredResultPayload(detail);
    const detailedRows = detailDetailedResults(detail);

    if (detailedRows.length) {
      return {
        columns: buildRolloutColumns("raw"),
        rows: detailedRows.map(normalizeRolloutDetailRow),
        summary: `Showing ${pluralize(detailedRows.length, "logged rollout")}`,
        message: "No rollout rows were logged.",
        note: "",
      };
    }

    return {
      columns: buildRolloutColumns("raw"),
      rows: [],
      summary: "No rollout samples logged",
      message: "This RL run did not persist prompt/completion/reward samples.",
      note:
        "This historical run has step metrics, but not the actual rollout rows. New runs after the logging fix will show prompt, completion, and reward here.",
    };
  }

  function buildPayloadState(detail, loadingDetail) {
    if (loadingDetail && !detail) {
      return { text: "Loading canonical payload…" };
    }
    if (!detail) {
      return { text: "No run selected." };
    }
    const history = asObject(detail.history);
    const payload = {
      summary: asObject(detail.summary),
      result_payload: asObject(detail.result_payload),
      job_result_payload: asObject(detail.job_result_payload),
      config: asObject(detail.config),
      history_preview: {
        keys: Array.isArray(history.keys) ? history.keys : [],
        default_keys: Array.isArray(history.default_keys) ? history.default_keys : [],
        row_count: Array.isArray(history.rows) ? history.rows.length : 0,
      },
    };
    return { text: JSON.stringify(payload, null, 2) };
  }

  function buildEvalColumns(rows, includeReward) {
    const first = rows.length ? rows[0].raw : {};
    if (isArithmeticEvalRow(first)) {
      return [
        { key: "status", label: "Status", type: "status", className: "status-cell cell-status" },
        { key: "problem", label: "Problem", type: "text", className: "problem-cell cell-problem sample-primary" },
        { key: "expected_answer", label: "Expected", type: "text", className: "expected-cell cell-expected sample-mono" },
        { key: "parsed_value", label: "Parsed answer", type: "text", className: "guess-cell cell-parsed sample-mono" },
        { key: "completion", label: "Completion", type: "completion", className: "completion-cell cell-completion", previewLength: 180 },
        { key: "absolute_error", label: "Absolute error", type: "text", className: "numeric-cell sample-mono" },
      ];
    }

    const columns = [{ key: "status", label: "Status", type: "status", className: "status-cell cell-status" }];
    if (includeReward) {
      columns.push({ key: "reward", label: "Reward", type: "reward", className: "reward-cell cell-reward" });
    }
    columns.push(
      { key: "parsed_value", label: "Parsed", type: "text", className: "guess-cell cell-parsed sample-mono" },
      { key: "completion", label: "Completion", type: "completion", className: "completion-cell cell-completion", previewLength: 420 },
      { key: "prompt", label: "Prompt", type: "prompt", className: "prompt-cell cell-prompt", previewLength: 360 }
    );
    return columns;
  }

  function buildRolloutColumns(mode) {
    if (mode === "raw") {
      return [
        { key: "rollout_step", label: "Step", type: "text", className: "numeric-cell sample-mono" },
        { key: "rollout_batch_id", label: "Batch", type: "text", className: "numeric-cell sample-mono" },
        { key: "reward", label: "Reward", type: "reward", className: "reward-cell cell-reward" },
        { key: "components", label: "Components", type: "longtext", className: "copy-inline", previewLength: 200 },
        { key: "prompt", label: "Prompt", type: "prompt", className: "prompt-cell cell-prompt", previewLength: 280 },
        { key: "completion", label: "Completion", type: "completion", className: "completion-cell cell-completion", previewLength: 180 },
      ];
    }
    return [];
  }

  function normalizeEvalDisplayRow(row, index) {
    const raw = asObject(row.raw);
    return {
      key: raw.id != null ? `sample-${raw.id}` : `sample-${index}`,
      raw: raw,
      outcome: row.outcome,
      reward: row.reward,
      problem: raw.left == null || raw.right == null ? "n/a" : `${raw.left} + ${raw.right}`,
      expected_answer:
        raw.expected_answer != null ? String(raw.expected_answer) : "n/a",
      parsed_value: findParsedValue(raw),
      completion: raw.completion ? String(raw.completion) : "",
      prompt: resolvePromptPreview(raw),
      absolute_error:
        raw.absolute_error == null
          ? "n/a"
          : formatMetricValue(raw.absolute_error, "absolute_error"),
    };
  }

  function normalizeRolloutDetailRow(row, index) {
    const raw = asObject(row);
    return {
      key: raw.id != null ? `rollout-${raw.id}` : `rollout-${index}`,
      reward: raw.reward_total != null ? raw.reward_total : raw.reward,
      rollout_step: raw.rollout_step != null ? String(raw.rollout_step) : "n/a",
      rollout_batch_id:
        raw.rollout_batch_id != null ? String(raw.rollout_batch_id) : "n/a",
      components: summarizeRewardComponents(raw.reward_components),
      prompt: raw.prompt ? String(raw.prompt) : "",
      completion: raw.completion ? String(raw.completion) : "",
    };
  }

  function panelLabel(panel) {
    if (panel === "overview") {
      return "Overview";
    }
    if (panel === "history") {
      return "History";
    }
    if (panel === "samples") {
      return "Samples";
    }
    if (panel === "rollouts") {
      return "Rollouts";
    }
    return "Payload";
  }

  function firstExperimentId(payload) {
    const experiments = Array.isArray(payload && payload.experiments)
      ? payload.experiments
      : [];
    return experiments.length ? String(experiments[0].experiment_id || "") : null;
  }

  function experimentOptionLabel(item) {
    if (item.loading) {
      return `${item.experiment_id} (loading...)`;
    }
    if (item.run_count) {
      return `${item.experiment_id} (${item.run_count} runs)`;
    }
    return String(item.experiment_id || "");
  }

  function buildChartGeometry(values) {
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
    return {
      padding: padding,
      innerHeight: innerHeight,
      linePath: linePath,
      areaPath: areaPath,
      last: coordinates[coordinates.length - 1],
      minY: minY,
      maxY: maxY,
    };
  }

  function pathFromCoordinates(points) {
    return points
      .map((point, index) => `${index === 0 ? "M" : "L"} ${point[0].toFixed(2)} ${point[1].toFixed(2)}`)
      .join(" ");
  }

  function withAlpha(color, alpha) {
    const hex = String(color || "").replace("#", "");
    if (hex.length !== 6) {
      return color;
    }
    const red = parseInt(hex.slice(0, 2), 16);
    const green = parseInt(hex.slice(2, 4), 16);
    const blue = parseInt(hex.slice(4, 6), 16);
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function summarizeText(text, previewLength) {
    const fullText = String(text || "");
    if (!fullText.trim()) {
      return "";
    }
    const normalized = fullText.replace(/\s+/g, " ").trim();
    if (normalized.length <= previewLength) {
      return normalized;
    }
    return `${normalized.slice(0, Math.max(40, previewLength - 1)).trimEnd()}…`;
  }

  function pickChartKeys(history, runSummary) {
    const keys = Array.isArray(history.keys) ? history.keys : [];
    const defaults = Array.isArray(history.default_keys) ? history.default_keys : [];
    const phase = runSummary ? String(runSummary.phase || "").toLowerCase() : "";
    const preferred = metricPriority[phase] || [];
    const selected = [];

    [
      preferred.filter((key) => !isNoisyHistoryMetricKey(key)),
      defaults.filter((key) => !isNoisyHistoryMetricKey(key)),
      keys.filter((key) => !isNoisyHistoryMetricKey(key)),
      preferred.filter((key) => isNoisyHistoryMetricKey(key)),
      defaults.filter((key) => isNoisyHistoryMetricKey(key)),
      keys.filter((key) => isNoisyHistoryMetricKey(key)),
    ].forEach((pool) => {
      pool.forEach((key) => {
        if (keys.includes(key) && !selected.includes(key) && selected.length < 4) {
          selected.push(key);
        }
      });
    });

    return selected.slice(0, 4);
  }

  function pickRunSummary(runs, candidateKeys) {
    const list = Array.isArray(runs) ? runs : [];
    const candidates = Array.isArray(candidateKeys) ? candidateKeys.filter(Boolean) : [];
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
      const [key, value] = entries[0];
      return `${prettifyKey(key)} ${formatMetricValue(value, key)}`;
    }
    if (Number.isFinite(Number(run.total_time_seconds))) {
      return formatDuration(Number(run.total_time_seconds));
    }
    if (run.hf_revision) {
      return `Artifact ${String(run.hf_revision).slice(0, 8)}`;
    }
    return "";
  }

  function runPreviewFallbackText(run) {
    if (hasPendingTerminalMetrics(run, run.metrics)) {
      return "Live run; final metrics pending";
    }
    return "No metrics logged yet";
  }

  function detailMetricsEmptyState(summary, metrics) {
    if (hasPendingTerminalMetrics(summary, metrics)) {
      return "This run is still live. Final summary metrics are not available yet.";
    }
    return "No summary metrics were logged for this run.";
  }

  function collectMetrics(detail) {
    const summary = asObject(detail.summary);
    const summaryMetrics = asObject(summary.metrics);
    const resultMetrics = asObject(asObject(detail.result_payload).metrics);
    const historyMetrics = latestHistoryMetricValues(asObject(detail.history));
    return Object.assign({}, historyMetrics, resultMetrics, summaryMetrics);
  }

  function collectOverviewMetrics(detail) {
    const summary = asObject(detail.summary);
    const phase = String(summary.phase || "").toLowerCase();
    const sourceMetrics = collectMetrics(detail);
    const groups = overviewMetricGroups[phase] || [];
    const selected = {};

    groups.forEach((group) => {
      const match = findFirstMetricValue(sourceMetrics, group.sources);
      if (match) {
        selected[group.key] = match.value;
      }
    });

    if (Object.keys(selected).length) {
      return selected;
    }

    return Object.fromEntries(
      orderedMetricEntries(sourceMetrics, phase)
        .filter((entry) => !isNoisyOverviewMetricKey(entry[0]))
        .slice(0, 6)
    );
  }

  function latestHistoryMetricValues(history) {
    const rows = Array.isArray(history.rows) ? history.rows : [];
    const latest = {};
    rows.forEach((row) => {
      Object.entries(asObject(row)).forEach(([key, value]) => {
        if (
          key !== "step" &&
          key !== "timestamp" &&
          key !== "runtime" &&
          typeof value === "number" &&
          Number.isFinite(value)
        ) {
          latest[key] = value;
        }
      });
    });
    return latest;
  }

  function findFirstMetricValue(metrics, keys) {
    return keys.reduce((match, key) => {
      if (match) {
        return match;
      }
      const value = asObject(metrics)[key];
      return isDisplayableMetricValue(value) ? { key: key, value: value } : null;
    }, null);
  }

  function hasPendingTerminalMetrics(run, metrics) {
    const summary = asObject(run);
    if (!summary.is_active || summary.job_result_present) {
      return false;
    }
    return !Object.keys(asObject(metrics)).length;
  }

  function orderedMetricEntries(metrics, phaseOverride) {
    const entries = Object.entries(asObject(metrics)).filter((entry) =>
      isDisplayableMetricValue(entry[1])
    );
    const phase = String(phaseOverride || "").toLowerCase();
    const preferred = metricPriority[phase] || [];

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

  function normalizeEvalRow(row) {
    const record = asObject(row);
    return {
      raw: record,
      reward: extractRowReward(record),
      outcome: sampleOutcome(record),
    };
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
    for (const key of ["reward_total", "reward", "total_reward"]) {
      if (Number.isFinite(Number(record[key]))) {
        return Number(record[key]);
      }
    }
    return null;
  }

  function summarizeRewardBreakdown(row) {
    const record = asObject(row);
    const parts = [
      ["fmt", "reward_format"],
      ["dict", "reward_dict"],
      ["constraints", "reward_constraints"],
      ["repeat", "reward_repeat"],
      ["length", "reward_overlength"],
    ]
      .filter((entry) => Number.isFinite(Number(record[entry[1]])))
      .map((entry) => `${entry[0]} ${formatSignedValue(record[entry[1]])}`);

    if (parts.length) {
      return parts.join(" · ");
    }

    return Object.entries(asObject(record.reward_components))
      .filter((entry) => Number.isFinite(Number(entry[1])))
      .map((entry) => `${prettifyKey(entry[0])} ${formatSignedValue(entry[1])}`)
      .join(" · ");
  }

  function sampleOutcome(row) {
    const record = asObject(row);
    const explicitReasons = Array.isArray(record.failure_reasons)
      ? record.failure_reasons
          .map((reason) => prettifyKey(reason))
          .filter((reason) => reason && reason !== "n/a")
      : [];
    if (Object.prototype.hasOwnProperty.call(record, "passed")) {
      if (record.passed) {
        return { passed: true, reasons: [] };
      }
      return { passed: false, reasons: explicitReasons.length ? explicitReasons : ["status"] };
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

    const reasons = Object.entries(record)
      .filter((entry) => typeof entry[1] === "boolean" && !entry[1])
      .map((entry) => prettifyKey(entry[0]));
    return { passed: reasons.length === 0, reasons: reasons };
  }

  function resolvePromptPreview(rawRow) {
    const record = asObject(rawRow);
    if (record.prompt) {
      return String(record.prompt);
    }
    if (isArithmeticEvalRow(record)) {
      return `${record.left} + ${record.right} = ?`;
    }
    return "";
  }

  function isArithmeticEvalRow(rawRow) {
    const record = asObject(rawRow);
    return (
      record.left != null &&
      record.right != null &&
      Object.prototype.hasOwnProperty.call(record, "expected_answer")
    );
  }

  function findParsedValue(rawRow) {
    const record = asObject(rawRow);
    if (record.parsed_guess != null && record.parsed_guess !== "") {
      return String(record.parsed_guess);
    }
    if (record.parsed_answer != null && record.parsed_answer !== "") {
      return String(record.parsed_answer);
    }
    return "n/a";
  }

  function buildHfUrl(repoId, revision) {
    const encodedRepo = String(repoId || "").trim();
    if (!encodedRepo) {
      return "https://huggingface.co";
    }
    const base = `https://huggingface.co/${encodedRepo}`;
    return revision ? `${base}/tree/${encodeURIComponent(String(revision))}` : base;
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
    if (activeCount > 0) {
      return "running";
    }
    if (Number(statuses.failed || 0) > 0) {
      return "failed";
    }
    if (Number(statuses.partial || 0) > 0 || Number(statuses.stopped || 0) > 0) {
      return "partial";
    }
    if (Number(statuses.success || 0) > 0 || runCount > 0) {
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

  function buildDetailCacheKey(experimentId, run) {
    return `${String(experimentId || "")}::${buildRunKey(run)}`;
  }

  function isEvalLikeRun(run) {
    const summary = asObject(run);
    return String(summary.phase || "").toLowerCase() === "eval";
  }

  function isRLRun(run) {
    const summary = asObject(run);
    return String(summary.phase || "").toLowerCase() === "rl";
  }

  function resolvePanelForRun(runSummary, currentPanel) {
    if (currentPanel === "samples" && !isEvalLikeRun(runSummary)) {
      return "overview";
    }
    if (currentPanel === "rollouts" && !isRLRun(runSummary)) {
      return "overview";
    }
    return currentPanel;
  }

  function resolveActivePanel(selectedPanel, runSummary, detail) {
    const panels = availablePanels(runSummary, detail);
    if (panels.includes(selectedPanel)) {
      return selectedPanel;
    }
    if (shouldHoldRequestedPanel(selectedPanel, runSummary)) {
      return selectedPanel;
    }
    return "overview";
  }

  function shouldHoldRequestedPanel(selectedPanel, runSummary) {
    if (!runSummary && (selectedPanel === "samples" || selectedPanel === "rollouts")) {
      return true;
    }
    if (selectedPanel === "samples" && isEvalLikeRun(runSummary)) {
      return true;
    }
    if (selectedPanel === "rollouts" && isRLRun(runSummary)) {
      return true;
    }
    return false;
  }

  function hasMeaningfulSamples(run) {
    const summary = asObject(run);
    return summary.processed_samples != null || summary.expected_samples != null;
  }

  function isNoisyHistoryMetricKey(key) {
    const text = String(key || "").toLowerCase();
    return (
      text.startsWith("profiling/") ||
      text.startsWith("train/clip_ratio/") ||
      text === "train/completions/clipped_ratio" ||
      text === "train/frac_reward_zero_std"
    );
  }

  function isNoisyOverviewMetricKey(key) {
    const text = String(key || "").toLowerCase();
    return (
      text === "total_flos" ||
      text === "train_runtime" ||
      text === "train_samples_per_second" ||
      text === "train_steps_per_second"
    );
  }

  function preferredResultPayload(detail) {
    const resultPayload = asObject(detail && detail.result_payload);
    const jobPayload = asObject(detail && detail.job_result_payload);
    if (Array.isArray(resultPayload.detailed_results) && resultPayload.detailed_results.length) {
      return resultPayload;
    }
    if (Array.isArray(jobPayload.detailed_results) && jobPayload.detailed_results.length) {
      return jobPayload;
    }
    return Object.keys(resultPayload).length ? resultPayload : jobPayload;
  }

  function detailDetailedResults(detail) {
    const payload = preferredResultPayload(detail);
    return Array.isArray(payload.detailed_results) ? payload.detailed_results : [];
  }

  function summarizeRewardComponents(components) {
    return Object.entries(asObject(components))
      .filter((entry) => Number.isFinite(Number(entry[1])))
      .map((entry) => `${prettifyKey(entry[0])} ${formatSignedValue(entry[1])}`)
      .join(" · ");
  }

  function pluralize(count, noun) {
    const numericCount = Number(count || 0);
    return `${numericCount} ${noun}${numericCount === 1 ? "" : "s"}`;
  }

  function routeRunKeyForExperiment(route, experimentId) {
    if (!route || !route.experimentId || route.experimentId !== experimentId) {
      return null;
    }
    if (!route.phase || !route.runName) {
      return null;
    }
    return `${route.phase}::${route.runName}`;
  }

  function writeRoute(state) {
    const params = new URLSearchParams(window.location.search);
    if (state.experimentId) {
      params.set("experiment_id", state.experimentId);
    } else {
      params.delete("experiment_id");
    }
    if (state.runSummary) {
      params.set("phase", String(state.runSummary.phase || ""));
      params.set("run_name", String(state.runSummary.run_name || ""));
    } else {
      params.delete("phase");
      params.delete("run_name");
    }
    if (state.panel && state.panel !== "overview") {
      params.set("panel", state.panel);
    } else {
      params.delete("panel");
    }
    const query = params.toString();
    window.history.replaceState({}, "", query ? `${window.location.pathname}?${query}` : window.location.pathname);
  }

  function readRoute() {
    const params = new URLSearchParams(window.location.search);
    return {
      experimentId: trimOrNull(params.get("experiment_id")),
      phase: trimOrNull(params.get("phase")),
      runName: trimOrNull(params.get("run_name")),
      panel: trimOrNull(params.get("panel")),
    };
  }

  function updateDocumentTitle(experimentId, runSummary) {
    if (experimentId && runSummary) {
      document.title = `Tenyson Dashboard · ${experimentId} · ${runSummary.run_name}`;
      return;
    }
    if (experimentId) {
      document.title = `Tenyson Dashboard · ${experimentId}`;
      return;
    }
    document.title = "Tenyson Dashboard";
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
        // Ignore parse fallback issues.
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
    if (normalizedKey.includes("learning_rate")) {
      return `${(numeric * 100).toFixed(1)}%`;
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
      return numeric.toFixed(4).replace(/\.?0+$/, "");
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
    if (Number(value) > 0) {
      return "positive";
    }
    if (Number(value) < 0) {
      return "negative";
    }
    return "neutral";
  }

  function prettifyKey(key) {
    const labelKey = String(key || "");
    const overrides = {
      exact_match_accuracy: "Exact match",
      format_accuracy: "Format accuracy",
      parsed_answer_rate: "Parsed answer rate",
      avg_abs_error: "Avg absolute error",
      total_samples: "Total samples",
      train_loss: "Train loss",
      "train/loss": "Loss",
      "train/reward": "Reward",
      train_reward: "Reward",
      "train/kl": "KL",
      "train/reward_std": "Reward std",
      "train/completion_length": "Completion length",
      "train/grad_norm": "Grad norm",
      "train/learning_rate": "Learning rate",
      train_steps_per_second: "Steps / sec",
      train_samples_per_second: "Samples / sec",
      global_step: "Global step",
      "train/global_step": "Global step",
      epoch: "Epoch",
      "train/epoch": "Epoch",
    };
    if (Object.prototype.hasOwnProperty.call(overrides, labelKey)) {
      return overrides[labelKey];
    }
    return labelKey
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
    const formatter =
      typeof Intl !== "undefined" && typeof Intl.RelativeTimeFormat === "function"
        ? new Intl.RelativeTimeFormat(undefined, { numeric: "auto" })
        : null;
    for (const [unit, ms] of [
      ["day", 24 * 60 * 60 * 1000],
      ["hour", 60 * 60 * 1000],
      ["minute", 60 * 1000],
      ["second", 1000],
    ]) {
      if (absMs >= ms || unit === "second") {
        const amount = Math.round(deltaMs / ms);
        return formatter
          ? formatter.format(amount, unit)
          : `${Math.abs(amount)} ${unit}${Math.abs(amount) === 1 ? "" : "s"} ago`;
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
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    }
    return `${seconds}s`;
  }

  function toDate(value) {
    if (!value) {
      return null;
    }
    const date = value instanceof Date ? value : new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function trimOrNull(value) {
    const trimmed = String(value || "").trim();
    return trimmed ? trimmed : null;
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  ReactDOM.createRoot(document.getElementById("app")).render(html`<${DashboardApp} />`);
})();
