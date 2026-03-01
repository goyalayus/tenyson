# Goal
Fix the four identified code-level blockers that would prevent reliable experiment execution: RL runtime NameError, cloud runner package-path issues, fragile task loading spec in cloud managers, and invalid example import path.

# Context
- RL code uses `Path` without import in `tenyson/src/tenyson/jobs/rl.py`.
- AWS/Modal invoke `python -m tenyson.runner` from remote workspace without guaranteeing `src` package visibility.
- AWS/Modal derive `task_spec` solely from `task.__class__.__module__`, which breaks for file-loaded task plugins.
- Wordle example imports `tenyson.examples.wordle.wordle_task`, but examples are not packaged under `src/tenyson`.

# Constraints
- Keep diffs minimal and focused on bugs 1–4 only.
- Preserve existing behavior for currently working flows.
- No redesign of orchestration patterns in this task.
- Verify syntactic correctness and local static sanity checks after edits.

# Proposed changes
- `tenyson/src/tenyson/jobs/rl.py`: add missing `Path` import.
- `tenyson/src/tenyson/cloud/aws.py`: make remote project root resolution robust for src-layout and nested checkout; make task spec resolution robust with file-path fallback.
- `tenyson/src/tenyson/cloud/modal.py`: same task spec robustness and runner working-directory/PYTHONPATH handling.
- `tenyson/examples/wordle/experiment.py`: remove broken package import for task and load plugin from file path.

# Steps
1. Add missing import in RL job.
2. Patch AWS manager to:
   - detect remote project root (`~/workspace` vs `~/workspace/tenyson`) where `pyproject.toml` exists,
   - export `PYTHONPATH=src` before running runner,
   - resolve task to `module:Class` when importable or to task file path relative to repo root when not.
3. Patch Modal manager with equivalent task resolution and runtime path setup.
4. Patch Wordle example task construction to use file-path loading.
5. Run verification:
   - `python3 -m py_compile` on changed files,
   - targeted path-resolution smoke checks (non-GPU, non-cloud execution only).
6. Update plan status log, then update `ARCHITECTURE.md` to reflect these path/task-resolution decisions.
7. Commit changes with a focused message.

# Acceptance criteria
- RLJob module no longer raises NameError for `Path` usage.
- AWS and Modal cloud managers can execute runner with src-layout packaging assumptions.
- Task plugin resolution in cloud managers works for both importable module tasks and file-loaded tasks.
- Example script no longer references nonexistent `tenyson.examples` package path.
- Commit created with all above changes and docs updates.

# Risks / rollback
- Risk: path-detection shell logic could choose wrong remote directory for unusual layouts.
  - Mitigation: keep fallback deterministic (`~/workspace` then `~/workspace/tenyson`).
- Risk: task-file fallback path may fail when task file is outside synced root.
  - Mitigation: explicit error raised if task is outside repo root; keep module-spec path as first option.
- Rollback: revert this single commit to restore previous behavior.

# Status log
- 2026-03-01T19:02:00Z: Task started; created required planning artifact and architecture bootstrap file.
- 2026-03-01T19:08:00Z: Implemented bug fixes in code:
  - Added `Path` import in `tenyson/src/tenyson/jobs/rl.py`.
  - Hardened AWS cloud manager with local project root detection for src-layout, task-spec fallback to file path when module import is not stable, remote `PYTHONPATH=src`, and output/result path alignment when invoked from parent directory.
  - Hardened Modal cloud manager with same project-root/task-spec handling and remote `PYTHONPATH=src` setup.
  - Updated `tenyson/examples/wordle/experiment.py` to load the task plugin from file path and resolve config/report paths relative to the script location.
- 2026-03-01T19:09:00Z: Verification run:
  - `python3 -m py_compile` succeeded for all modified files.
  - Structural grep checks confirm helper methods/imports/path handling are present.
  - Note: runtime cloud integration test was not executed in this environment (no AWS/Modal credentials/session; local env also lacks `boto3` for import-time smoke scripts).
- 2026-03-01T19:11:00Z: Updated `ARCHITECTURE.md` to fully reflect current runtime/package/cloud/task-resolution behavior and new bug-fix decisions.
