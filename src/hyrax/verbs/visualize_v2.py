import logging
from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING

from .verb_registry import Verb, hyrax_verb

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _disable_axis_zoom(plot, element):
    """Bokeh hook: disable axis-only zoom and align tick marks inward."""
    from bokeh.models import WheelZoomTool

    for tool in plot.state.tools:
        if isinstance(tool, WheelZoomTool):
            tool.zoom_on_axis = False
    plot.state.match_aspect = True
    for axis in [*plot.state.xaxis, *plot.state.yaxis]:
        axis.major_tick_in = 6
        axis.major_tick_out = 0
        axis.minor_tick_in = 3
        axis.minor_tick_out = 0
        axis.major_label_standoff = -20


@hyrax_verb
class VisualizeV2(Verb):
    """Verb to create a hexbin visualization of a 2D latent space."""

    cli_name = "visualize_v2"
    add_parser_kwargs = {}

    REQUIRED_DATA_GROUPS = ("visualize",)
    OPTIONAL_DATA_GROUPS = ()

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """CLI not implemented for this verb"""
        pass

    def run_cli(self, args: Namespace | None = None):
        """CLI not implemented for this verb"""
        logger.error("Running visualize_v2 from the cli is unimplemented")

    def run(
        self,
        **kwargs,
    ):
        """Generate an interactive hexbin visualization of a latent space projected to 2D.

        Uses HoloViews HexTiles with datashader for adaptive hexbin aggregation,
        box/lasso selection, a metadata table, and tabbed detail plots.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments passed as HexTiles opts overrides.

        Returns
        -------
        VisualizeV2
            This verb instance. Use it to call ``restart_ui()`` or ``get_selected_df()``
            after the UI has been displayed.
        """
        import panel as pn

        # pn.extension must be called before displaying any Panel widget.
        # Keep it here (in addition to _build_ui) so the loading indicator works on first run.
        pn.extension("tabulator")

        def _status(dataset_done: bool, ui_done: bool) -> str:
            dataset_icon = "✅" if dataset_done else "⏳"
            ui_icon = "✅" if ui_done else "⏳"
            return f"- {dataset_icon} Loading dataset\n- {ui_icon} Rendering UI"

        _loading = pn.pane.Markdown(_status(False, False))
        ipy_display = clear_output = None
        try:
            from IPython.display import clear_output
            from IPython.display import display as ipy_display

            ipy_display(_loading)
        except ImportError:
            pass

        self._load_data()
        _loading.object = _status(True, False)

        # Wipe the loading indicator so the full UI replaces it cleanly in the cell output.
        if clear_output is not None:
            clear_output(wait=True)

        self._build_ui(**kwargs)  # Build the UI once to cache any heavy operations before showing it.
        return self

    def restart_ui(self, **kwargs):
        """Rebuild and re-display the Panel UI without reloading data.

        Call this after a Jupyter websocket disconnect instead of re-running the cell.
        The expensive data-loading step is skipped — only the widgets are rebuilt.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments passed as HexTiles opts overrides.

        Returns
        -------
        VisualizeV2
            This verb instance. Use it to call ``restart_ui()`` or ``get_selected_df()``
            after the UI has been displayed.
        """
        if not getattr(self, "_data_loaded", False):
            raise RuntimeError("No data loaded yet. Call run() first.")
        self._build_ui(**kwargs)
        return self

    def _load_data(self):
        """Load dataset and build the points DataFrame.

        Guards with a ``_data_loaded`` sentinel so the expensive steps only run
        once per verb instance. Safe to call multiple times.
        """
        if getattr(self, "_data_loaded", False):
            return

        import numpy as np
        import pandas as pd

        from hyrax.pytorch_ignite import setup_dataset

        # ── Build DataProvider for metadata access ────────────────────────────
        self.datasets = setup_dataset(self.config)
        if not set(VisualizeV2.REQUIRED_DATA_GROUPS).intersection(set(self.datasets.keys())):
            required_keys = ", ".join(sorted(VisualizeV2.REQUIRED_DATA_GROUPS))
            available_keys = ", ".join(sorted(self.datasets.keys())) or "<none>"
            raise RuntimeError(
                f"VisualizeV2 requires dataset entries {required_keys} in the data request. "
                f"Available: {available_keys}"
            )

        # The primary dataset (identified by primary_id_field) holds the UMAP coordinates.
        _primary_ds_name = self.datasets["visualize"].primary_dataset
        if _primary_ds_name is None:
            raise RuntimeError(
                "No sub-dataset in data_request.visualize has a 'primary_id_field'. "
                "Set 'primary_id_field' on the dataset that contains the UMAP 2D coordinates."
            )
        reduced_dim_dataset = self.datasets["visualize"].prepped_datasets[_primary_ds_name]
        reduced_dim_results = reduced_dim_dataset.__get_all__()

        # ── Build DataFrame from UMAP 2D results ─────────────────────────────
        points_array = np.array([np.asarray(pt) for pt in reduced_dim_results])
        df = pd.DataFrame(
            {"x": points_array[:, 0].astype(np.float32), "y": points_array[:, 1].astype(np.float32)}
        )

        # Store references on self for downstream use
        self.df = df
        self.reduced_dim_results = reduced_dim_results
        self._n_points = len(df)

        # ── Probe available scalar fields ─────────────────────────────────────
        # Call DataProvider[0] to sample the structure and filter to scalar fields only.
        # This drops large array/tensor fields (e.g. image, data) automatically.
        _sample = self.datasets["visualize"][0]
        self._dataset_getters = self.datasets["visualize"].dataset_getters
        _scalar_col_options: list[str] = []
        _scalar_types = (int, float, str, bool, np.integer, np.floating)
        if "object_id" in _sample and isinstance(_sample["object_id"], _scalar_types):
            _scalar_col_options.append("object_id")
        for _fn, _field_dict in _sample.items():
            if _fn == "object_id" or not isinstance(_field_dict, dict):
                continue
            for _field, _val in _field_dict.items():
                if isinstance(_val, _scalar_types):
                    _scalar_col_options.append(f"{_fn}.{_field}")
        self._scalar_col_options = _scalar_col_options
        # Warn if any dataset sub-config has a 'fields' restriction
        self._fields_restricted = any(
            bool(ds_conf.get("fields"))
            for ds_conf in self.config.get("data_request", {}).get("visualize", {}).values()
            if isinstance(ds_conf, dict)
        )

        self._data_loaded = True

    def _build_ui(self, **kwargs):
        """Build and display the Panel UI using data already loaded by ``_load_data()``."""
        import math
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import datashader as ds
        import holoviews as hv
        import matplotlib.axes
        import matplotlib.figure as mpl_figure
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import panel as pn
        from holoviews import DynamicMap, Polygons, Rectangles, extension
        from holoviews.element.stats import HexTiles
        from holoviews.operation.datashader import rasterize
        from holoviews.streams import BoundsXY, Lasso, Params, RangeXY
        from IPython import get_ipython
        from matplotlib.path import Path as MplPath

        # ── Local aliases for data loaded by _load_data() ─────────────────────
        df = self.df
        n_points = self._n_points
        _scalar_col_options = self._scalar_col_options
        _dataset_getters = self._dataset_getters
        _fields_restricted = self._fields_restricted

        # ── Config ────────────────────────────────────────────────────────────
        viz_config = self.config["visualize_v2"]

        target_bins = viz_config["target_bins"]
        buffer_factor = viz_config["buffer_factor"]
        plot_width = viz_config["plot_width"]
        plot_height = viz_config["plot_height"]
        cmap = viz_config["cmap"]  # Sets the initial colormap; user can change it via the UI slider.
        max_table_rows = viz_config["max_table_rows"]
        num_detail_plots = viz_config["num_detail_plots"]

        # ── HoloViews / Panel init ────────────────────────────────────────────
        # pn.extension must come before hv.extension so Panel can patch HoloViews'
        # comm machinery before the Bokeh backend registers its own callbacks.
        pn.extension("tabulator")
        extension("bokeh")

        # ── Colormap selector ─────────────────────────────────────────────────
        # Keys are display names, values are matplotlib colormap identifiers.
        _cmap_entries = {
            "Viridis": "viridis",
            "Inferno": "inferno",
            "Plasma": "plasma",
            "Magma": "magma",
            "Cividis": "cividis",
            "Turbo": "turbo",
            "Blues": "Blues",
            "Greens": "Greens",
            "Purples": "Purples",
            "Reds": "Reds",
            "YlOrRd": "YlOrRd",
            "YlGnBu": "YlGnBu",
            "Spring": "spring",
            "Summer": "summer",
            "Autumn": "autumn",
            "Winter": "winter",
        }
        _cmap_display_names = list(_cmap_entries.keys())
        _cmap_initial = next(
            (k for k, v in _cmap_entries.items() if v == cmap.lower()), _cmap_display_names[0]
        )
        _cmap_slider = pn.widgets.IntSlider(
            name=f"Colormap: {_cmap_initial}",
            start=0,
            end=len(_cmap_display_names) - 1,
            value=_cmap_display_names.index(_cmap_initial),
            width=plot_width,
        )

        # ── Determine initial range from data ─────────────────────────────────
        x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
        y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
        x_pad = (x_hi - x_lo) * 0.05
        y_pad = (y_hi - y_lo) * 0.05
        initial_x_range = (x_lo - x_pad, x_hi + x_pad)
        initial_y_range = (y_lo - y_pad, y_hi + y_pad)

        # ── Adaptive hexbin callback ──────────────────────────────────────────
        range_xy = RangeXY()
        _cmap_stream = Params(_cmap_slider, ["value"], rename={"value": "cmap_idx"})

        # Cache the last rasterized element so colormap changes skip re-rasterization.
        _hex_cache: dict = {"range_key": None, "element": None}

        def make_hexbin(x_range, y_range, cmap_idx=0):
            xr = x_range if (x_range is not None and None not in x_range) else initial_x_range
            yr = y_range if (y_range is not None and None not in y_range) else initial_y_range

            range_key = (xr, yr)
            if _hex_cache["range_key"] != range_key or _hex_cache["element"] is None:
                x_pad_buf = (xr[1] - xr[0]) * buffer_factor
                y_pad_buf = (yr[1] - yr[0]) * buffer_factor
                mask = (
                    (df["x"] >= xr[0] - x_pad_buf)
                    & (df["x"] <= xr[1] + x_pad_buf)
                    & (df["y"] >= yr[0] - y_pad_buf)
                    & (df["y"] <= yr[1] + y_pad_buf)
                )
                df_view = df[mask]

                bin_size = (xr[1] - xr[0]) / target_bins

                hexbin = rasterize(
                    HexTiles(df_view, kdims=["x", "y"]).redim.range(x=xr, y=yr),
                    aggregator=ds.count(),
                    x_sampling=bin_size,
                    y_sampling=bin_size,
                    dynamic=False,
                )

                if hexbin.vdims:
                    count_col = hexbin.vdims[0].name
                    hexbin = hexbin.clone(hexbin.data.assign(**{count_col: np.log1p(hexbin.data[count_col])}))

                _hex_cache["range_key"] = range_key
                _hex_cache["element"] = hexbin

            # Cheap: apply colormap to cached element (no re-rasterization)
            return _hex_cache["element"].opts(
                hv.opts.HexTiles(cmap=_cmap_entries[_cmap_display_names[cmap_idx]])
            )

        dmap = DynamicMap(make_hexbin, streams=[range_xy, _cmap_stream])

        # ── Plot opts ─────────────────────────────────────────────────────────
        plot_opts = {
            "cnorm": "linear",
            "colorbar": True,
            "colorbar_position": "bottom",
            "colorbar_opts": {"title": "log(Count + 1)"},
            "width": plot_width,
            "height": plot_height,
            "xlabel": "",
            "ylabel": "",
            "title": f"Hexbin — {n_points:,} samples  |  ~{target_bins} bins across",
            "toolbar": "above",
            "tools": ["hover", "pan", "wheel_zoom", "reset", "box_select", "lasso_select"],
            "line_color": "white",
            "line_width": 0.5,
            "nonselection_alpha": 1.0,
            "nonselection_line_alpha": 1.0,
        }
        plot_opts.update(kwargs)

        plot = dmap.opts(hv.opts.HexTiles(**plot_opts, hooks=[_disable_axis_zoom]))

        # ── Selection streams ─────────────────────────────────────────────────
        bounds_stream = BoundsXY(source=dmap, bounds=(0, 0, 0, 0))
        lasso_stream = Lasso(source=dmap)

        # ── Column selector ───────────────────────────────────────────────────
        col_selector_title = pn.pane.Markdown("### Columns", margin=(0, 0))
        _default_cols = ["object_id"] if "object_id" in _scalar_col_options else []
        col_selector = pn.widgets.CheckBoxGroup(
            options=_scalar_col_options,
            value=_default_cols,
            inline=False,
            stylesheets=["label { font-size: 16px !important; }"],
        )
        _fields_alert = (
            pn.pane.Alert(
                "Additional fields may be available. Remove `fields` from the data request to see them.",
                alert_type="info",
                margin=(0, 0, 6, 0),
            )
            if _fields_restricted
            else None
        )

        # ── Selection table ───────────────────────────────────────────────────
        _empty = pd.DataFrame(columns=["row_index"] + _default_cols)

        # 21 data rows × 35px + 30px header + 35px footer/pagination bar ≈ one row of breathing room
        _table_height = 21 * 35 + 30 + 35

        selection_table = pn.widgets.Tabulator(
            _empty.copy(),
            pagination="remote",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_width",
            height=_table_height,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": True}},
            disabled=True,
        )
        table_title = pn.pane.Markdown("### Selected Points", margin=(0, 0))

        # Selected subsets, accessible via verb instance
        self.selected_box = pd.DataFrame()
        self.selected_lasso = pd.DataFrame()
        _table_df: list[pd.DataFrame] = [pd.DataFrame()]  # authoritative copy unaffected by user edits
        _pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))

        def _make_fetcher(active_cols: list[str]):
            """Return a row-fetching callable for the given active columns."""
            top_level = [c for c in active_cols if c == "object_id"]
            nested = [(c.split(".", 1)[0], c.split(".", 1)[1]) for c in active_cols if "." in c]

            def _fetch_row(idx):
                row: dict = {"row_index": idx}
                if top_level:
                    row["object_id"] = self.datasets["visualize"][idx].get("object_id")
                for fn, field in nested:
                    row[f"{fn}.{field}"] = _dataset_getters[fn][field](idx)
                return row

            return _fetch_row

        def _update_table(
            sel: pd.DataFrame,
            progress_callback=None,
            should_abort=None,
        ) -> None:
            active_cols: list[str] = col_selector.value
            if sel.empty:
                computed = pd.DataFrame(columns=["row_index"] + [c for c in active_cols])
                _table_df[0] = computed.copy()
                selection_table.value = computed
                table_title.object = "### Selected Points"
                return

            capped_indices = list(sel.index[:max_table_rows])

            if not active_cols:
                computed = pd.DataFrame({"row_index": capped_indices})
                _table_df[0] = computed.copy()
                selection_table.value = computed
                _truncated = len(sel) > max_table_rows
                table_title.object = (
                    f"### Selected Points — {len(sel):,} (showing first {max_table_rows:,} — export to see all)"  # noqa: E501
                    if _truncated
                    else f"### Selected Points — {len(sel):,}"
                )
                return

            _fetch_row = _make_fetcher(active_cols)

            total = len(capped_indices)
            _progress_step = max(1, total // 100)

            # Bail out before submitting any futures if already stale
            if should_abort and should_abort():
                return

            futures = {_pool.submit(_fetch_row, idx): i for i, idx in enumerate(capped_indices)}
            rows: list = [None] * total
            for done, future in enumerate(as_completed(futures)):
                if should_abort and should_abort():
                    # Cancel all queued-but-not-started futures to drain the pool backlog
                    for f in futures:
                        f.cancel()
                    return
                rows[futures[future]] = future.result()
                if progress_callback and (done % _progress_step == 0 or done == total - 1):
                    progress_callback(int((done + 1) / total * 100))

            display_df = pd.DataFrame(rows).reset_index(drop=True)
            _table_df[0] = display_df.copy()
            selection_table.value = display_df
            _truncated = len(sel) > max_table_rows
            table_title.object = (
                f"### Selected Points — {len(sel):,} (showing first {max_table_rows:,} — export to see all)"  # noqa: E501
                if _truncated
                else f"### Selected Points — {len(sel):,}"
            )

        # Re-fetch columns whenever the selector changes
        def _on_col_selector_change(_e):
            sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
            if sel.empty:
                _update_table(sel)
                return
            _gen[0] += 1
            my_gen = _gen[0]

            def _run():
                _progress_bar.value = 0
                _update_table(
                    sel,
                    progress_callback=lambda pct: setattr(_progress_bar, "value", pct),
                    should_abort=lambda: _gen[0] != my_gen,
                )
                _progress_bar.value = 0

            threading.Thread(target=_run, daemon=True).start()

        col_selector.param.watch(_on_col_selector_change, "value")

        # Update slider label whenever the colormap changes
        def _on_cmap_slide(event):
            _cmap_slider.name = f"Colormap: {_cmap_display_names[event.new]}"

        _cmap_slider.param.watch(_on_cmap_slide, "value")

        # ── Detail panes ─────────────────────────────────────────────────────
        _total_width = plot_width
        _detail_pane_width = (_total_width - (20 * (num_detail_plots - 1))) // num_detail_plots
        _prepped_datasets = self.datasets["visualize"].prepped_datasets
        _tab_names = list(_prepped_datasets.keys())

        # Consistent subplot margins applied to every figure shown in a detail pane.
        # Using fixed subplots_adjust (instead of tight_layout) together with tight=False
        # in the Matplotlib pane ensures all plots occupy identical canvas geometry
        # regardless of content type.
        _detail_layout = dict(left=0.03, right=0.97, bottom=0.03, top=0.9)

        def _make_placeholder_fig():
            fig = mpl_figure.Figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5,
                0.5,
                "No selection",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#aaaaaa",
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
            fig.subplots_adjust(**_detail_layout)
            return fig

        def _make_text_fig(text, index):
            fig = mpl_figure.Figure(figsize=(3, 3))
            ax = fig.add_subplot(111)
            ax.set_facecolor("#f8f8f8")
            ax.text(
                0.05,
                0.95,
                text,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=12,
                family="monospace",
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Index: {index}" if index is not None else "No selection")
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
            fig.subplots_adjust(**_detail_layout)
            return fig

        def _make_pane_row():
            return [
                pn.pane.Matplotlib(
                    _make_placeholder_fig(),
                    tight=False,
                    format="png",
                    width=_detail_pane_width,
                    height=_detail_pane_width,
                )
                for _ in range(num_detail_plots)
            ]

        detail_panes = [_make_pane_row() for _ in _tab_names]

        _tab_active_css = """
        .bk-tab.bk-active { background-color: #d6eaf8 !important; color: #000000 !important; }
        """
        detail_tabs = pn.Tabs(
            *[(name, pn.Row(*panes, margin=0)) for name, panes in zip(_tab_names, detail_panes)],
            width=_total_width,
            stylesheets=[_tab_active_css],
        )

        def _make_detail_fig(index, tab_index):
            """Create a matplotlib figure for one detail pane.

            Accepts whatever ``dataset.display()`` returns and normalises it to a
            ``matplotlib.figure.Figure`` before handing it to the Matplotlib pane:

            - ``Figure``        → used directly
            - ``Axes``          → parent figure extracted
            - ``numpy.ndarray`` → imshow'd into a new Figure
            - anything else     → rendered as text via ``_make_text_fig``

            Any pyplot-managed figures accidentally created inside ``display()`` are
            closed after the call so they don't leak into the notebook output.
            """
            if index is None:
                return _make_placeholder_fig()
            dataset_name = _tab_names[tab_index]
            dataset = _prepped_datasets[dataset_name]
            if not callable(getattr(dataset, "display", None)):
                fig = _make_text_fig(str(dataset[index]), index)
            else:
                _fignums_before = set(plt.get_fignums())
                try:
                    result = dataset.display(index)
                finally:
                    # Close any pyplot figures created as a side-effect of display()
                    for _fn in set(plt.get_fignums()) - _fignums_before:
                        plt.close(_fn)

                if isinstance(result, mpl_figure.Figure):
                    fig = result
                elif isinstance(result, matplotlib.axes.Axes):
                    fig = result.figure
                elif isinstance(result, np.ndarray):
                    fig = mpl_figure.Figure(figsize=(3, 3))
                    ax = fig.add_subplot(111)
                    ax.imshow(result)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Index: {index}")
                else:
                    fig = _make_text_fig(str(result), index)

            fig.set_size_inches(3, 3)
            fig.subplots_adjust(**_detail_layout)
            return fig

        def _update_detail_panes(indices):
            def _apply():
                for tab_index, tab_panes in enumerate(detail_panes):
                    for i, pane in enumerate(tab_panes):
                        pane.object = _make_detail_fig(indices[i] if i < len(indices) else None, tab_index)

            if pn.state.curdoc is not None:
                pn.state.execute(_apply)
            else:
                _apply()

        # ── Pagination controls ───────────────────────────────────────────────
        _page_state: dict = {"page": 0, "indices": []}

        btn_first = pn.widgets.Button(name="|◀", width=44, button_type="default", disabled=True)
        btn_prev = pn.widgets.Button(name="◀", width=44, button_type="default", disabled=True)
        page_numbers_row = pn.Row(margin=(0, 4))
        btn_next = pn.widgets.Button(name="▶", width=44, button_type="default", disabled=True)
        btn_last = pn.widgets.Button(name="▶|", width=44, button_type="default", disabled=True)

        def _total_pages() -> int:
            return max(1, math.ceil(len(_page_state["indices"]) / num_detail_plots))

        def _refresh_pagination_widgets() -> None:
            page = _page_state["page"]
            total = _total_pages()
            window_start = max(0, page - 3)
            window_end = min(total - 1, page + 3)
            page_btns = []
            for p in range(window_start, window_end + 1):
                is_current = p == page
                btn = pn.widgets.Button(
                    name=str(p + 1),
                    width=44,
                    button_type="primary" if is_current else "default",
                    disabled=is_current,
                )
                if not is_current:
                    btn.on_click(lambda _e, _p=p: _go_to_page(_p))
                page_btns.append(btn)
            page_numbers_row.objects = page_btns
            btn_first.disabled = page == 0
            btn_prev.disabled = page == 0
            btn_next.disabled = page >= total - 1
            btn_last.disabled = page >= total - 1

        def _render_page() -> None:
            page = _page_state["page"]
            start = page * num_detail_plots
            _update_detail_panes(list(_page_state["indices"][start : start + num_detail_plots]))
            _refresh_pagination_widgets()

        def _go_to_page(new_page: int) -> None:
            _page_state["page"] = max(0, min(new_page, _total_pages() - 1))
            _render_page()

        btn_first.on_click(lambda _e: _go_to_page(0))
        btn_prev.on_click(lambda _e: _go_to_page(_page_state["page"] - 1))
        btn_next.on_click(lambda _e: _go_to_page(_page_state["page"] + 1))
        btn_last.on_click(lambda _e: _go_to_page(_total_pages() - 1))

        pagination_row = pn.Row(
            pn.Spacer(),
            btn_first,
            btn_prev,
            page_numbers_row,
            btn_next,
            btn_last,
            pn.Spacer(),
            width=_total_width,
            align="center",
        )
        _refresh_pagination_widgets()  # initialise the page-number buttons

        # ── Selection overlay callback ────────────────────────────────────────
        _progress_bar = pn.indicators.Progress(width=_total_width, value=0, max=100, bar_color="info")
        _gen: list[int] = [0]

        _prev = {"bounds": (0, 0, 0, 0), "geometry": None}

        def selection_overlay(bounds, geometry):
            try:
                return _selection_overlay_impl(bounds, geometry)
            except Exception:
                logger.error("selection_overlay raised an exception", exc_info=True)
                return Rectangles([]).opts(apply_ranges=False) * Polygons([]).opts(apply_ranges=False)

        def _selection_overlay_impl(bounds, geometry):
            bounds_changed = bounds is not None and bounds != _prev["bounds"]
            geometry_changed = geometry is not _prev["geometry"]
            if bounds is not None:
                _prev["bounds"] = bounds
            _prev["geometry"] = geometry

            box_el = Rectangles([]).opts(apply_ranges=False)
            lasso_el = Polygons([]).opts(apply_ranges=False)

            if bounds_changed or (geometry_changed and geometry is not None):
                export_btn.icon = ""
                export_btn.name = "Export table to selected_points"
                export_btn.disabled = True
                export_all_btn.icon = ""
                export_all_btn.name = "Export all to selected_points"
                export_all_btn.disabled = True

            if bounds_changed:
                self.selected_lasso = pd.DataFrame()
                x0, y0, x1, y1 = bounds
                if x0 != x1 and y0 != y1:
                    mask = (df["x"] >= x0) & (df["x"] <= x1) & (df["y"] >= y0) & (df["y"] <= y1)
                    self.selected_box = df[mask]
                    if not self.selected_box.empty:
                        box_el = Rectangles([(x0, y0, x1, y1)]).opts(
                            fill_alpha=0.1,
                            fill_color="cyan",
                            line_color="cyan",
                            line_width=1.5,
                            apply_ranges=False,
                        )
                else:
                    self.selected_box = pd.DataFrame()

            elif geometry_changed and geometry is not None:
                self.selected_box = pd.DataFrame()

                if isinstance(geometry, dict):
                    coords = np.array(geometry["coordinates"][0])
                else:
                    coords = np.asarray(geometry)

                if len(coords) >= 3:
                    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
                    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
                    bbox_mask = (
                        (df["x"] >= x_min) & (df["x"] <= x_max) & (df["y"] >= y_min) & (df["y"] <= y_max)
                    )
                    df_candidates = df[bbox_mask]
                    path = MplPath(coords)
                    inside = path.contains_points(df_candidates[["x", "y"]].values)
                    self.selected_lasso = df_candidates[inside]
                    if not self.selected_lasso.empty:
                        lasso_el = Polygons([{("x", "y"): coords}]).opts(
                            fill_alpha=0.1,
                            fill_color="cyan",
                            line_color="cyan",
                            line_width=1.5,
                            apply_ranges=False,
                        )
                else:
                    self.selected_lasso = pd.DataFrame()

            # Spawn background thread for slow work; return overlay immediately
            if bounds_changed or (geometry_changed and geometry is not None):
                sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
                _gen[0] += 1
                my_gen = _gen[0]
                if sel.empty:
                    # Empty selection is fast — handle inline to avoid thread overhead
                    _update_table(sel)
                    _page_state["page"] = 0
                    _page_state["indices"] = []
                    _render_page()
                    export_btn.disabled = False
                    export_all_btn.disabled = False
                else:
                    threading.Thread(target=_do_selection_work, args=(sel, my_gen), daemon=True).start()

            return box_el * lasso_el

        def _do_selection_work(sel: "pd.DataFrame", my_gen: int) -> None:
            try:
                _progress_bar.value = 0
                export_btn.name = "Loading selected data..."

                if _gen[0] != my_gen:
                    return

                # Render detail panes immediately — indices are available now,
                # before the (slower) table fetch begins.
                try:
                    _page_state["page"] = 0
                    _page_state["indices"] = list(sel.index)
                    _render_page()
                except Exception:
                    pass  # never let detail-pane errors break status cleanup

                if _gen[0] != my_gen:
                    return

                _update_table(
                    sel,
                    progress_callback=lambda pct: setattr(_progress_bar, "value", pct),
                    should_abort=lambda: _gen[0] != my_gen,
                )

                export_btn.name = "Export table to selected_points"
                export_btn.disabled = False
                export_all_btn.disabled = False
            finally:
                _progress_bar.value = 0

        selection_dmap = DynamicMap(selection_overlay, streams=[bounds_stream, lasso_stream])

        # ── Export button ─────────────────────────────────────────────────────
        export_btn = pn.widgets.Button(
            name="Export table to selected_points",
            button_type="primary",
            width=300,
            icon="",
        )

        def _on_export(event):
            sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
            if sel.empty:
                return
            exported = _table_df[0] if not _table_df[0].empty else sel
            if (_ipy := get_ipython()) is not None:
                _ipy.user_ns["selected_points"] = exported
            export_btn.icon = "check-lg"
            export_btn.name = f"Exported {len(exported):,} rows to selected_points"

        export_btn.on_click(_on_export)

        # ── Export All button ──────────────────────────────────────────
        export_all_btn = pn.widgets.Button(
            name="Export all to selected_points",
            button_type="warning",
            width=300,
            icon="",
            disabled=True,
        )

        def _do_export_all(sel: "pd.DataFrame") -> None:
            active_cols: list[str] = col_selector.value
            if not active_cols:
                if (_ipy := get_ipython()) is not None:
                    _ipy.user_ns["selected_points"] = pd.DataFrame({"row_index": list(sel.index)})
                export_all_btn.icon = "check-lg"
                export_all_btn.name = f"Exported {len(sel):,} rows to selected_points"
                export_all_btn.disabled = False
                _progress_bar.value = 0
                return

            _fetch_row = _make_fetcher(active_cols)

            all_indices = list(sel.index)
            total = len(all_indices)
            _progress_step = max(1, total // 100)
            futures = {_pool.submit(_fetch_row, idx): i for i, idx in enumerate(all_indices)}
            rows: list = [None] * total
            for done, future in enumerate(as_completed(futures)):
                rows[futures[future]] = future.result()
                if done % _progress_step == 0 or done == total - 1:
                    _progress_bar.value = int((done + 1) / total * 100)

            full_df = pd.DataFrame(rows).reset_index(drop=True)
            if (_ipy := get_ipython()) is not None:
                _ipy.user_ns["selected_points"] = full_df
            export_all_btn.icon = "check-lg"
            export_all_btn.name = f"Exported {len(full_df):,} rows to selected_points"
            export_all_btn.disabled = False
            _progress_bar.value = 0

        def _on_export_all(event):
            sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
            if sel.empty:
                return
            export_all_btn.disabled = True
            export_all_btn.icon = ""
            export_all_btn.name = f"Fetching all {len(sel):,} rows..."
            _progress_bar.value = 0
            threading.Thread(target=_do_export_all, args=(sel,), daemon=True).start()

        export_all_btn.on_click(_on_export_all)

        # ── Layout ────────────────────────────────────────────────────────────
        combined = (plot * selection_dmap).opts(
            hv.opts.Rectangles(apply_ranges=False),
            hv.opts.Polygons(apply_ranges=False),
        )

        _col_selector_col = pn.Column(
            col_selector_title,
            *([_fields_alert] if _fields_alert else []),
            pn.Column(
                col_selector,
                height=_table_height // 2,
                scroll=True,
            ),
            width=280,
        )
        _table_col = pn.Column(
            table_title,
            selection_table,
            pn.Spacer(height=6),
            pn.Row(export_btn, pn.Spacer(width=10), export_all_btn),
            sizing_mode="stretch_width",
        )

        pane = pn.Column(
            combined,
            _cmap_slider,
            _progress_bar,
            pn.Row(
                _col_selector_col,
                pn.Spacer(width=20),
                _table_col,
                width=_total_width,
            ),
            pn.Spacer(height=10),
            detail_tabs,
            pagination_row,
        )

        try:
            from IPython.display import display

            display(pane)
        except ImportError:
            logger.warning("Couldn't find IPython display environment. Skipping display step.")

    def get_selected_df(self):
        """Return the current selection as a DataFrame."""
        import pandas as pd

        sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
        if sel.empty:
            return pd.DataFrame()
        return sel
