import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .verb_registry import Verb, hyrax_verb

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@hyrax_verb
class VisualizeV2(Verb):
    """Verb to create a hexbin visualization of a 2D latent space."""

    cli_name = "visualize_v2"
    add_parser_kwargs = {}

    REQUIRED_DATA_GROUPS = ("infer",)
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
        input_dir: Union[Path, str] | None = None,
        **kwargs,
    ):
        """Generate an interactive hexbin visualization of a latent space projected to 2D.

        Uses HoloViews HexTiles with datashader for adaptive hexbin aggregation,
        box/lasso selection, a metadata table, and tabbed detail plots.

        Parameters
        ----------
        input_dir : Optional[Union[Path, str]], optional
            Directory holding output from the 'umap' verb. When not provided, reads
            from ``[results][inference_dir]`` in config.
        kwargs :
            Additional keyword arguments passed as HexTiles opts overrides.

        Returns
        -------
        panel.Column
            The assembled Panel layout.
        tuple of (panel.Column, VisualizeV2)
            If return_verb is True.
        """
        import math
        import os
        from concurrent.futures import ThreadPoolExecutor

        import datashader as ds
        import holoviews as hv
        import matplotlib.figure as mpl_figure
        import numpy as np
        import pandas as pd
        import panel as pn
        from holoviews import DynamicMap, Polygons, Rectangles, extension
        from holoviews.element.stats import HexTiles
        from holoviews.operation.datashader import rasterize
        from holoviews.streams import BoundsXY, Lasso, RangeXY
        from IPython import get_ipython
        from matplotlib.path import Path as MplPath

        from hyrax.pytorch_ignite import setup_dataset

        # ── Config ────────────────────────────────────────────────────────────
        viz_config = self.config["visualize_v2"]

        target_bins = viz_config["target_bins"]
        buffer_factor = viz_config["buffer_factor"]  # This doesn't need to be a config
        plot_width = viz_config["plot_width"]  # This doesn't need to be a config
        plot_height = viz_config["plot_height"]  # This doesn't need to be a config
        cmap = viz_config["cmap"]  # We should turn this into a UI widget.
        max_table_rows = viz_config["max_table_rows"]  # This doesn't need to be a config
        num_detail_plots = viz_config["num_detail_plots"]  # This doesn't need to be a config

        # ── Load UMAP results ────────────────────────────────────────────────
        if input_dir is None:
            input_dir = self.config["data_request"]["infer"]["results"]["data_location"]

        # umap_results = load_results_dataset(self.config, results_dir=input_dir, verb="umap")

        # ── Build DataProvider for metadata access ────────────────────────────
        self.datasets = setup_dataset(self.config)
        if not set(VisualizeV2.REQUIRED_DATA_GROUPS).intersection(set(self.datasets.keys())):
            required_keys = ", ".join(sorted(VisualizeV2.REQUIRED_DATA_GROUPS))
            available_keys = ", ".join(sorted(self.datasets.keys())) or "<none>"
            raise RuntimeError(
                f"VisualizeV2 requires dataset entries {required_keys} in the data request. "
                f"Available: {available_keys}"
            )

        reduced_dim_dataset = self.datasets["infer"].prepped_datasets["results"]
        reduced_dim_results = reduced_dim_dataset.__getitem__(range(0, len(reduced_dim_dataset)))

        # ── Build DataFrame from UMAP 2D results ─────────────────────────────
        points_array = np.array([np.asarray(pt) for pt in reduced_dim_results])
        df = pd.DataFrame(
            {"x": points_array[:, 0].astype(np.float32), "y": points_array[:, 1].astype(np.float32)}
        )
        n_points = len(df)

        # Store references on self for downstream use
        self.df = df
        self.reduced_dim_results = reduced_dim_results

        # ── Probe available scalar fields ─────────────────────────────────────
        # Call DataProvider[0] to sample the structure and filter to scalar fields only.
        # This drops large array/tensor fields (e.g. image, data) automatically.
        _sample = self.datasets["infer"][0]
        _dataset_getters = self.datasets["infer"].dataset_getters
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
        # Warn if any dataset sub-config has a 'fields' restriction
        _fields_restricted = any(
            bool(ds_conf.get("fields"))
            for ds_conf in self.config.get("data_request", {}).get("infer", {}).values()
            if isinstance(ds_conf, dict)
        )

        # ── HoloViews / Panel init ────────────────────────────────────────────
        # pn.extension must come before hv.extension so Panel can patch HoloViews'
        # comm machinery before the Bokeh backend registers its own callbacks.
        pn.extension("tabulator")
        extension("bokeh")

        # ── Determine initial range from data ─────────────────────────────────
        x_lo, x_hi = float(df["x"].min()), float(df["x"].max())
        y_lo, y_hi = float(df["y"].min()), float(df["y"].max())
        x_pad = (x_hi - x_lo) * 0.05
        y_pad = (y_hi - y_lo) * 0.05
        initial_x_range = (x_lo - x_pad, x_hi + x_pad)
        initial_y_range = (y_lo - y_pad, y_hi + y_pad)

        tiles = HexTiles(df, kdims=["x", "y"]).redim.range(x=initial_x_range, y=initial_y_range)  # noqa: F841

        # ── Adaptive hexbin callback ──────────────────────────────────────────
        range_xy = RangeXY()

        def make_hexbin(x_range, y_range):
            xr = x_range if (x_range is not None and None not in x_range) else initial_x_range
            yr = y_range if (y_range is not None and None not in y_range) else initial_y_range

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
                data = hexbin.data.copy()
                data[count_col] = np.log1p(data[count_col])
                hexbin = hexbin.clone(data)

            return hexbin

        dmap = DynamicMap(make_hexbin, streams=[range_xy])

        # ── Plot opts ─────────────────────────────────────────────────────────
        plot_opts = {
            "cmap": cmap,
            "cnorm": "linear",
            "colorbar": True,
            "colorbar_opts": {"title": "log(Count + 1)"},
            "width": plot_width,
            "height": plot_height,
            "xlabel": "x",
            "ylabel": "y",
            "title": f"Hexbin — {n_points:,} samples  |  ~{target_bins} bins across",
            "toolbar": "above",
            "tools": ["hover", "pan", "wheel_zoom", "reset", "box_select", "lasso_select"],
            "line_color": "white",
            "line_width": 0.5,
            "nonselection_alpha": 1.0,
            "nonselection_line_alpha": 1.0,
        }
        plot_opts.update(kwargs)
        plot = dmap.opts(hv.opts.HexTiles(**plot_opts))

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
            width=340,
        )
        _fields_alert = (
            pn.pane.Alert(
                "Additional fields may be available. Remove `fields` from the data request to see them.",
                alert_type="info",
                width=340,
                margin=(0, 0, 6, 0),
            )
            if _fields_restricted
            else None
        )

        # ── Selection table ───────────────────────────────────────────────────
        _empty = pd.DataFrame(columns=["row_index"])

        selection_table = pn.widgets.Tabulator(
            _empty.copy(),
            pagination="remote",
            page_size=25,
            show_index=False,
            sizing_mode="stretch_height",
            width=340,
            header_align="right",
            configuration={"columnDefaults": {"headerSort": True}},
            disabled=True,
        )
        table_title = pn.pane.Markdown("### Selected Points", margin=(0, 0))

        # Selected subsets, accessible via verb instance
        self.selected_box = pd.DataFrame()
        self.selected_lasso = pd.DataFrame()
        _table_df: list[pd.DataFrame] = [pd.DataFrame()]  # authoritative copy unaffected by user edits

        def _update_table(sel: pd.DataFrame) -> None:
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
                table_title.object = f"### Selected Points — {len(sel):,}"
                return

            _top_level = [c for c in active_cols if c == "object_id"]
            _nested = [(c.split(".", 1)[0], c.split(".", 1)[1]) for c in active_cols if "." in c]

            def _fetch_row(idx):
                row: dict = {"row_index": idx}
                if _top_level:
                    row["object_id"] = self.datasets["infer"][idx].get("object_id")
                for fn, field in _nested:
                    row[f"{fn}.{field}"] = _dataset_getters[fn][field](idx)
                return row

            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                rows = list(pool.map(_fetch_row, capped_indices))

            display_df = pd.DataFrame(rows).reset_index(drop=True)
            _table_df[0] = display_df.copy()
            selection_table.value = display_df
            table_title.object = f"### Selected Points — {len(sel):,}"

        # Re-fetch columns whenever the selector changes
        col_selector.param.watch(
            lambda _e: _update_table(
                self.selected_box if not self.selected_box.empty else self.selected_lasso
            ),
            "value",
        )

        # ── Detail panes ─────────────────────────────────────────────────────
        _total_width = plot_width + 340
        _detail_pane_width = (_total_width - (20 * (num_detail_plots - 1))) // num_detail_plots
        _prepped_datasets = self.datasets["infer"].prepped_datasets
        _tab_names = list(_prepped_datasets.keys())

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
            fig.tight_layout()
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
                fontsize=7,
                family="monospace",
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Index: {index}" if index is not None else "No selection")
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
            fig.tight_layout()
            return fig

        def _make_pane_row():
            return [
                pn.pane.Matplotlib(
                    _make_placeholder_fig(),
                    tight=True,
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

            Override this method or supply custom render functions via config
            to customise the per-object detail view.
            """
            if index is None:
                return _make_placeholder_fig()
            dataset_name = _tab_names[tab_index]
            dataset = _prepped_datasets[dataset_name]
            if not callable(getattr(dataset, "display", None)):
                return _make_text_fig(str(dataset[index]), index)
            return dataset.display(index)

        def _update_detail_panes(indices):
            for tab_index, tab_panes in enumerate(detail_panes):
                for i, pane in enumerate(tab_panes):
                    pane.object = _make_detail_fig(indices[i] if i < len(indices) else None, tab_index)

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
                export_btn.name = "Export to selected_points"

            if bounds_changed:
                self.selected_lasso = pd.DataFrame()
                x0, y0, x1, y1 = bounds
                if x0 != x1 and y0 != y1:
                    mask = (df["x"] >= x0) & (df["x"] <= x1) & (df["y"] >= y0) & (df["y"] <= y1)
                    self.selected_box = df[mask]
                    _update_table(self.selected_box)
                    box_el = Rectangles([(x0, y0, x1, y1)]).opts(
                        fill_alpha=0.1,
                        fill_color="cyan",
                        line_color="cyan",
                        line_width=1.5,
                        apply_ranges=False,
                    )
                else:
                    self.selected_box = pd.DataFrame()
                    _update_table(pd.DataFrame())

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
                    _update_table(self.selected_lasso)
                    lasso_el = Polygons([{("x", "y"): coords}]).opts(
                        fill_alpha=0.1,
                        fill_color="orange",
                        line_color="orange",
                        line_width=1.5,
                        apply_ranges=False,
                    )
                else:
                    self.selected_lasso = pd.DataFrame()
                    _update_table(pd.DataFrame())

            # Update detail plots
            if bounds_changed or (geometry_changed and geometry is not None):
                try:
                    sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
                    _page_state["page"] = 0
                    _page_state["indices"] = list(sel.index)
                    _render_page()
                except Exception:
                    pass  # never let detail-pane errors break the overlay return

            return box_el * lasso_el

        selection_dmap = DynamicMap(selection_overlay, streams=[bounds_stream, lasso_stream])

        # ── Export button ─────────────────────────────────────────────────────
        export_btn = pn.widgets.Button(
            name="Export to selected_points",
            button_type="primary",
            width=340,
            icon="",
        )

        def _on_export(event):
            sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
            if sel.empty:
                return
            exported = _table_df[0].copy() if not _table_df[0].empty else sel
            try:
                _ipy = get_ipython()  # noqa: F821
                _ipy.user_ns["selected_points"] = exported
            except NameError:
                pass
            export_btn.icon = "check-lg"
            export_btn.name = f"Exported {len(exported):,} rows to selected_points"

        export_btn.on_click(_on_export)

        # ── Layout ────────────────────────────────────────────────────────────
        combined = (plot * selection_dmap).opts(
            hv.opts.Rectangles(apply_ranges=False),
            hv.opts.Polygons(apply_ranges=False),
        )

        pane = pn.Column(
            pn.Row(
                combined,
                pn.Column(
                    *([_fields_alert] if _fields_alert else []),
                    col_selector_title,
                    col_selector,
                    pn.Spacer(height=6),
                    table_title,
                    selection_table,
                    pn.Spacer(height=10),
                    export_btn,
                    sizing_mode="stretch_height",
                ),
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

        # if return_verb:
        #     return pane, self
        # return pane

    def get_selected_df(self) -> "pd.DataFrame":
        """Return the current selection as a DataFrame."""
        import pandas as pd

        sel = self.selected_box if not self.selected_box.empty else self.selected_lasso
        if sel.empty:
            return pd.DataFrame()
        return sel
