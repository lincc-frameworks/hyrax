# VisualizeV2 Improvement Plan

## Goals

1. **No forced re-runs** — the user must never have to re-execute a notebook cell just to
   recover the UI.  For large datasets, the startup cost (UMAP coordinate load) can be
   many minutes, so losing the UI to a recoverable error is unacceptable.
2. **Appear more responsive** — show partial data as soon as it arrives rather than
   blocking until everything is ready.
3. **Fix known bugs** — silent failures, resource leaks, and subtle edge-case errors.
4. **Simplify** — reduce duplication and flatten the monolithic `run()` into testable
   pieces.

---

## Bugs

### B1 — Colormap matching silently fails for mixed-case names (line 167)
```python
# Current — "Blues" in config will never match "blues"
(k for k, v in _cmap_entries.items() if v == cmap.lower())

# Fix
(k for k, v in _cmap_entries.items() if v.lower() == cmap.lower())
```

### B2 — `ThreadPoolExecutor` is never shut down
`_pool` (line 314) is created every time `run()` is called but `shutdown()` is never
called.  Each notebook cell re-execution abandons a pool of 8 threads.

**Fix:** store the pool on `self._pool`; shut it down at the top of `_build_panel` before
creating a new one.  Also register `pn.state.on_session_destroyed` when available.

### B3 — No per-row exception handling in `as_completed` loop (line 369)
`future.result()` re-raises.  One corrupted record silently kills the entire table update.

**Fix:** wrap in `try/except`, log the error, and substitute a fallback row
`{"row_index": idx}`.

### B4 — `_do_export_all` has no generation/abort check
If the user clicks "Export all" on a 500 k selection and then makes a new selection, the
old export thread runs to completion and overwrites `selected_points` with stale data.

**Fix:** pass the current `_gen[0]` value to `_do_export_all`; bail out if `_gen[0]`
changes while iterating.

### B5 — Empty-selection path enables export buttons (lines 692-694)
After clearing a selection both export buttons are re-enabled, but there is nothing to
export.

**Fix:** leave both buttons disabled when `sel.empty`.

---

## Reliability — No Forced Re-runs

### R1 — Split data loading from UI construction (highest priority)

**Root cause:** `run()` does everything.  Lines 87-115 (dataset setup + UMAP load, the
expensive part) are fused with lines 139+ (widget construction, the cheap part).  The
user cannot restart the UI without reloading the data.

**Fix:** split into two methods:

| Method | Does | When called |
|---|---|---|
| `_load_data()` | `setup_dataset`, `__get_all__`, DataFrame build | Once per verb instance |
| `_build_panel(container, **kwargs)` | All widget/stream construction | On first display **and** on Reset |

`run()` becomes a thin orchestrator:
1. Show a loading spinner in a `pn.Column` container.
2. `display(container)` — the cell returns immediately.
3. Background thread: call `_load_data()` (skip if already done), then `_build_panel()`,
   then replace `container.objects`.

### R2 — Add a "Reset UI" button

A small **↺ Reset UI** button in the top-right corner.  Clicking it calls
`self._build_panel(container, **kwargs)` — reconstructs all widgets and streams from
scratch using the already-loaded `self.df` / `self.datasets`.  No data reload, no
notebook cell re-execution.

This handles: stuck progress bars, broken stream state, Bokeh callback errors, and any
other widget-level corruption.

### R3 — Isolate detail-pane exceptions

Wrap each `pane.object = _make_detail_fig(...)` assignment in `try/except` so one
failing pane (bad image, missing file) does not block the rest.  Fall back to
`_make_placeholder_fig()` and log the error.

### R4 — Surface errors inside the UI

The `selection_overlay` exception handler (line 611) logs to the Python logger but shows
nothing to the user — they see "nothing happened."  Write a small status label or
`pn.pane.Alert` that appears when a background error occurs, so the user knows to hit
Reset rather than assuming the tool is frozen.

---

## Perceived Performance

### P1 — Async initial data load with progress feedback

Instead of blocking the cell, show the spinner immediately and populate the hexbin once
data is ready.  The user sees the widget skeleton (and can read the spinner message)
while UMAP coordinates load.

### P2 — Async detail-pane rendering

`_render_page()` currently blocks on all `dataset.display()` calls before updating any
pane.  For 6 image panes at 200 ms each, the UI freezes for ~1.2 s per page turn.

**Fix:** set all panes to placeholder immediately, then render each figure in a single
background thread (placeholders → real figures one by one).  The user sees the page
switch happen instantly; images pop in as they load.

### P3 — Partial table streaming

`_update_table` collects all futures into `rows` before calling
`selection_table.value = display_df`.  For 1 000-row selections the table is blank until
100 % complete.

**Fix:** update `selection_table.value` every `_STREAM_EVERY` (≈ 50) completed futures.
Rows appear in their original order as they arrive.

### P4 — Shared row-fetcher (also a simplification)

`_fetch_row` inside `_update_table` and `_fetch_row_all` inside `_do_export_all` are
nearly identical.  Extract `_make_row_fetcher(active_cols)` factory used by both.
This also makes the column-selector change path reuse the same code path.

### P5 — Debounce / throttle `RangeXY` (future work)

HoloViews streams support a `throttle` parameter (ms) that batches rapid events.
Adding `RangeXY(throttle=80)` would reduce rasterization calls during smooth pan/zoom.
**This is not implemented here** because the exact API varies across HoloViews versions;
verify compatibility before adding.

---

## Simplification

### S1 — Monolithic `run()` → `_load_data()` + `_build_panel()`

Covered by R1.  The 876-line single function becomes three focused methods.

### S2 — Shared IPython namespace injection

```python
# Before (duplicated verbatim)
try:
    _ipy = get_ipython()
    _ipy.user_ns["selected_points"] = exported
except NameError:
    pass

# After (one helper)
def _export_to_namespace(name: str, df: pd.DataFrame) -> None:
    try:
        get_ipython().user_ns[name] = df
    except (NameError, AttributeError):
        pass
```

### S3 — Config/constant ambiguity

Five config keys are commented "This doesn't need to be a config."  Either embrace them
as user-tunable configs (and remove the comments) or demote them to module-level
constants.  The current state is confusing.  **This plan keeps them as configs** (the
simpler path) and removes the misleading comments.

---

## Implementation Priority

| # | Item | Category | Effort |
|---|---|---|---|
| 1 | B1 — colormap `.lower()` fix | Bug | Trivial |
| 2 | B2 — pool shutdown | Bug | Small |
| 3 | B3 — per-row exception handling | Bug | Small |
| 4 | B4 — export-all abort check | Bug | Small |
| 5 | B5 — empty-selection export buttons | Bug | Trivial |
| 6 | R1 — split data / UI | Reliability | Medium |
| 7 | R2 — Reset UI button | Reliability | Small |
| 8 | R3 — isolate detail-pane exceptions | Reliability | Small |
| 9 | S2 — shared row fetcher + IPython helper | Simplification | Small |
| 10 | P2 — async detail panes | Performance | Small |
| 11 | P3 — partial table streaming | Performance | Small |
