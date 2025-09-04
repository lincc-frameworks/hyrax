import logging
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy.typing as npt
import torch
from matplotlib.colors import LogNorm

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Visualize(Verb):
    """Verb to create a visualization"""

    cli_name = "visualize"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """CLI not implemented for this verb"""
        pass

    def run_cli(self, args: Optional[Namespace] = None):
        """CLI not implemented for this verb"""
        logger.error("Running visualize from the cli is unimplemented")

    def run(
        self,
        input_dir: Optional[Union[Path, str]] = None,
        *,
        return_verb: bool = False,
        make_lupton_rgb_opts: Optional[dict] = None,
        **kwargs,
    ):
        """Generate an interactive notebook visualization of a latent space that has been umapped down to 2d.

        The plot contains two holoviews objects, a scatter plot of the latent space, and a table of objects
        which can be populated by selecting from the scatter plot.

        Parameters
        ----------
        input_dir : Optional[Union[Path, str]], optional
            Directory holding the output from the 'umap' verb, by default None. When not provided, we use
            [results][inference_dir] from config. If that's false; we the most recent umap in the current
            results directory.

        return_verb : bool, optional
            If True, also return the underlying Visualize instance for post-hoc access
            to selection state. Defaults to False.

        make_lupton_rgb_opts : dict, optional
            Dictionary of options to pass to astropy's make_lupton_rgb function for RGB image creation.
            Default is {"stretch": 5, "Q": 8}. Common parameters include stretch (brightness/contrast)
            and Q (softening parameter for asinh transformation).

        kwargs :
            Keyword arguments are passed through as options for the plot object as
            ``plot_pane.opts(**plot_options)``. It is not recommended to override the "tools" plot option,
            because that will break the integration between the plot selection operations and the table.

        Returns
        -------
        Holoviews, if return_verb = True (defaul)
            A Collection of Haloviews Panes

        tuple of (pane, Visualize), if return_verb = True
           Returns a 2-tuple with the pane and the verb instance.
        """
        import numpy as np
        import panel as pn
        from holoviews import DynamicMap, extension
        from holoviews.operation.datashader import dynspread, rasterize
        from holoviews.streams import Lasso, Params, RangeXY, SelectionXY, Tap
        from scipy.spatial import KDTree

        from hyrax.data_sets.inference_dataset import InferenceDataSet

        fields = ["object_id"]
        fields += self.config["visualize"]["fields"]
        self.cmap = self.config["visualize"]["cmap"]

        if self.config["data_set"]["filename_column_name"]:
            self.filename_column_name = self.config["data_set"]["filename_column_name"]
        else:
            self.filename_column_name = "filename"

        if self.config["visualize"]["display_images"]:
            fields += [self.filename_column_name]

        # If no input directory is specified, read from config.
        if input_dir is None:
            logger.info("UMAP directory not specified at runtime. Reading from config values.")
            input_dir = (
                self.config["results"]["inference_dir"] if self.config["results"]["inference_dir"] else None
            )

        # Get the umap data and put it in a kdtree for indexing.
        self.umap_results = InferenceDataSet(self.config, results_dir=input_dir, verb="umap")
        logger.info(f"Rendering UMAP from the following directory: {self.umap_results.results_dir}")

        available_fields = self.umap_results.metadata_fields()
        for field in fields.copy():
            if field not in available_fields:
                logger.warning(f"Field {field} is unavailable for this dataset")
                fields.remove(field)

        if "object_id" not in fields:
            msg = "Umap dataset must support object_id field"
            raise RuntimeError(msg)

        self.data_fields = fields.copy()
        self.data_fields.remove("object_id")

        self.tree = KDTree(self.umap_results)

        # Store color column and extract color values if specified
        self.color_column = self.config["visualize"]["color_column"]
        self.color_values = None

        # Validate torch_tensor_bands configuration
        self.torch_tensor_bands = self.config["visualize"]["torch_tensor_bands"]  # Defaults to i-band
        if len(self.torch_tensor_bands) not in [1, 3]:
            raise ValueError(
                f"torch_tensor_bands must specify either 1 band (single-band) or 3 bands (RGB). "
                f"Got {len(self.torch_tensor_bands)} bands: {self.torch_tensor_bands}"
            )

        # Store make_lupton_rgb options with defaults
        self.make_lupton_rgb_opts = make_lupton_rgb_opts or {"stretch": 5, "Q": 8}

        if self.color_column:
            try:
                # Check if column exists
                available_fields = self.umap_results.metadata_fields()
                if self.color_column not in available_fields:
                    logger.warning(
                        f"Column '{self.color_column}' not found in dataset."
                        f" Available fields: {available_fields}"
                    )
                    self.color_column = False
                else:
                    # Get all indices for the dataset
                    all_indices = list(range(len(self.umap_results)))

                    # Extract metadata for the specified column
                    metadata = self.umap_results.metadata(all_indices, [self.color_column])
                    self.color_values = metadata[self.color_column]
                    logger.info(f"Successfully loaded color values from column '{self.color_column}'")
                    import numpy as np

                    logger.debug(
                        f"Color values range: {np.nanmin(self.color_values)} "
                        f"to {np.nanmax(self.color_values)}"
                    )
                    logger.debug(f"NaN count: {np.sum(np.isnan(self.color_values))}")
            except Exception as e:
                logger.warning(f"Could not load column '{self.color_column}': {e}")
                logger.warning("Proceeding without coloring")
                self.color_column = False
                self.color_values = None

        # Initialize holoviews with bokeh.
        extension("bokeh")

        # Set up the plot pane
        xmin, xmax, ymin, ymax = self._even_aspect_bounding_box()
        self.plot_options = {
            "tools": ["box_select", "lasso_select", "tap"],
            "width": 500,
            "height": 500,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax),
            "cnorm": "eq_hist",
        }
        self.plot_options.update(kwargs)

        if self.color_column:
            # For colored plots, show all points to preserve colorbar
            # This is a current Hack to overcome the fact that the
            # RangeXY stream breaks the colorbar. Needs to be investigated
            # further for permanent solution.
            plot_dm = DynamicMap(
                lambda: self.visible_points(
                    x_range=[float("-inf"), float("inf")], y_range=[float("-inf"), float("inf")]
                )
            )
        else:
            plot_dm = DynamicMap(self.visible_points, streams=[RangeXY()])

        if self.config["visualize"]["rasterize_plot"]:
            # Note that reasterization will break color-bar feature
            plot_pane = dynspread(rasterize(plot_dm).opts(**self.plot_options))
        else:
            plot_pane = plot_dm.opts(**self.plot_options)

        # Setup the table pane event handler
        self.prev_kwargs = {
            # For Lasso
            "geometry": None,
            # For Tap
            "x": None,
            "y": None,
            # For SelectionXY
            "bounds": None,
            "x_selection": None,
            "y_selection": None,
        }
        table_streams = [
            Lasso(source=plot_pane),
            Tap(source=plot_pane),
            SelectionXY(source=plot_pane),
        ]

        # Setup the table pane
        # self.table = Table(tuple([[0]]*(3+len(self.data_fields))), ["object_id"], self.data_fields)
        self.points = np.array([])
        self.points_id = np.array([])
        self.points_idx = np.array([])

        self.table = self._table_from_points()
        table_options = {"width": self.plot_options["width"]}
        table_pane = DynamicMap(self.selected_objects, streams=table_streams).opts(**table_options)

        # If display_images is set to True then display randomly chosen images from the selected
        # sample underneath the table pane
        if self.config["visualize"]["display_images"]:
            pn.extension()

            # Create a small loading spinner same height as button
            self.spinner = pn.indicators.LoadingSpinner(
                value=False,  # Start with spinner off
                height=30,  # Smaller height to match button
                width=30,  # Smaller width
                margin=(5, 10, 5, 0),  # Add some margin for spacing
            )

            refresh_btn = pn.widgets.Button(name="Resample Images", button_type="primary")

            # Create a button row with spinner next to button
            button_row = pn.Row(refresh_btn, self.spinner, align="start")

            image_pane = DynamicMap(
                self._load_images, streams=[Params(refresh_btn, ["clicks"]), *table_streams]
            )

            images_panel = pn.pane.HoloViews(image_pane)

            plot_panel = pn.panel(plot_pane)

            # Set the table pane to be max 30% of the height
            table_h = int(self.plot_options["height"] * 0.3)
            table_panel = pn.panel(table_pane, height=table_h)

            right = pn.Column(table_panel, images_panel, button_row)

            pane = pn.Row(plot_panel, right)

        else:
            # Plot pane and table pane side by side
            pane = plot_pane + table_pane

        # We attempt to display the pane (fails outside a notebook)
        try:
            from IPython.display import display

            display(pane)
        except ImportError:
            logger.warning("Couldn't find IPython display environment. Skipping display step.")

        if return_verb:
            return pane, self
        else:
            return pane

    def visible_points(self, x_range: Union[tuple, list], y_range: Union[tuple, list]):
        """Generate a hv.Points object with the points inside the bounding box passed.

        This is the event handler for moving or scaling the latent space plot, and is called by Holoviews.

        Parameters
        ----------
        x_range : tuple or list
            min and max x values
        y_range : tuple or list
            min and max y values

        Returns
        -------
        hv.Points
            Points lying inside the bounding box passed
        """
        import numpy as np
        from holoviews import Points

        if x_range is None or y_range is None:
            return Points([])

        if np.any(np.isinf([x_range, y_range])):
            # Show all points without filtering
            points = np.array([point.numpy() for point in self.umap_results])
            point_indices = list(range(len(self.umap_results)))
        else:
            # Use existing filtering logic
            points, _, point_indices = self.box_select_points(x_range, y_range)

        if self.color_values is not None and len(point_indices) > 0:
            visible_colors = self.color_values[point_indices]
            # Create Points object with color data (x, y, color)
            point_data = np.column_stack([points, visible_colors])
            pts = Points(point_data, vdims=[self.color_column])

            # Apply color options directly to the Points object
            pts = pts.opts(
                color=self.color_column,
                cmap=self.cmap,
                colorbar=True,
                colorbar_opts={
                    "width": 18,
                    "title": self.color_column,
                    "title_text_font_size": "14pt",
                    "title_text_font_style": "normal",
                },
            )
        else:
            pts = Points(points)

        return pts

    def update_points(self, **kwargs) -> None:
        """
        This is the main UI event handler for selection tools on the plot. If you are a dynamic map
        in the layout of the visualizer who updates based on plot selection you MUST call this function.

        This function accepts the data values from all streams and uses the differences between the current
        call and prior calls to differentiate between different UI events.

        The self.prev_kwargs dictionary is used to store previous calls to this function, and the
        ``_called_*`` helpers perform the differencing for each case.

        Calling this function GUARANTEES that self.points, self.points_id, and self.points_idx
        are up-to-date with the user's latest selection, regardless of the order that Holoviews evaluates
        the DynamicMaps in.
        """
        import numpy as np

        if self._called_lasso(kwargs):
            self.points, self.points_id, self.points_idx = self.poly_select_points(kwargs["geometry"])
        elif self._called_tap(kwargs):
            _, idx = self.tree.query([kwargs["x"], kwargs["y"]])
            self.points = np.array([self.umap_results[idx].numpy()])
            self.points_id = np.array([list(self.umap_results.ids())[idx]])
            self.points_idx = np.array([idx])
        elif self._called_box_select(kwargs):
            self.points, self.points_id, self.points_idx = self.box_select_points(
                kwargs["x_selection"], kwargs["y_selection"]
            )
        else:
            # We saw no change that indicated a user intent; therefore, this is either initialization
            # OR we are not the first DynamicMap to run.
            pass

        self.prev_kwargs = kwargs

    def _called_lasso(self, kwargs):
        return kwargs["geometry"] is not None and (
            self.prev_kwargs["geometry"] is None
            or len(self.prev_kwargs["geometry"]) != len(kwargs["geometry"])
            or any(self.prev_kwargs["geometry"].flatten() != kwargs["geometry"].flatten())
        )

    def _called_tap(self, kwargs):
        return (
            kwargs["x"] is not None
            and kwargs["y"] is not None
            and (self.prev_kwargs["x"] != kwargs["x"] or self.prev_kwargs["y"] != kwargs["y"])
        )

    def _called_box_select(self, kwargs):
        return (
            kwargs["x_selection"] is not None
            and kwargs["y_selection"] is not None
            and (
                (self.prev_kwargs["x_selection"] is None and self.prev_kwargs["x_selection"] is None)
                or (
                    self.prev_kwargs["x_selection"] != kwargs["x_selection"]
                    or self.prev_kwargs["y_selection"] != kwargs["y_selection"]
                )
            )
        )

    def poly_select_points(self, geometry) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Select points inside a polygon.

        Parameters
        ----------
        geometry : list
            List of x/y points describing the verticies of the polygon

        Returns
        -------
        Tuple
            First element is an ndarray of x/y points in latent space inside the polygon
            Second element is an ndarray of corresponding object ids
        """
        import numpy as np
        from scipy.spatial import Delaunay

        # Coarse grain the points within the axis-aligned bounding box of the geometry
        (xmin, xmax, ymin, ymax) = Visualize._bounding_box(geometry)
        point_indexes_coarse = self.box_select_indexes([xmin, xmax], [ymin, ymax])
        points_coarse = self.umap_results[point_indexes_coarse].numpy()

        tri = Delaunay(geometry)
        mask = tri.find_simplex(points_coarse) != -1

        mask = np.asarray(mask)

        if any(mask):
            points = points_coarse[mask]
            point_indexes = np.array(point_indexes_coarse)[mask]
            points_id = np.array(list(self.umap_results.ids()))[point_indexes]
            return points, points_id, point_indexes
        else:
            return np.array([[]]), np.array([]), np.array([])

    def box_select_points(
        self, x_range: Union[tuple, list], y_range: Union[tuple, list]
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Return the points and IDs for a box in the latent space

        Parameters
        ----------
        x_range : tuple or list
            min and max x values
        y_range : tuple or list
            min and max y values

        Returns
        -------
        Tuple
            First element is an ndarray of x/y points in latent space inside the box
            Second element is an ndarray of corresponding object ids
        """
        import numpy as np

        indexes = self.box_select_indexes(x_range, y_range)
        ids = np.array(list(self.umap_results.ids()))[indexes]
        points = self.umap_results[indexes].numpy()
        return points, ids, indexes

    def box_select_indexes(self, x_range: Union[tuple, list], y_range: Union[tuple, list]):
        """Return the indexes inside of a particular box in the latent space

        Parameters
        ----------
        x_range : tuple or list
            min and max x values
        y_range : tuple or list
            min and max y values


        Returns
        -------
        np.ndarray
            Array of data indexes where the latent space representation falls inside the given box.
        """
        import numpy as np

        # Find center
        xc = (x_range[0] + x_range[1]) / 2.0
        yc = (y_range[0] + y_range[1]) / 2.0
        query_pt = [xc, yc]

        # Find larger of  half-width and half-height to use as our search radius.
        radius = np.max([np.max(x_range) - xc, np.max(y_range) - yc])

        # This is slightly overzealous, grabbing points outside the box sometimes.
        indexes = self.tree.query_ball_point(query_pt, radius, p=np.inf)

        def _inside_box(pt):
            x, y = pt
            xmin, xmax = x_range
            ymin, ymax = y_range
            return x > xmin and x < xmax and y > ymin and y < ymax

        # Filter for points properly inside the box
        return [i for i in indexes if _inside_box(self.umap_results[i].numpy())]

    def selected_objects(self, **kwargs):
        """
        Generate the holoview table for a selected set of objects based on input from the
        Lasso, Tap, and SelectionXY streams.

        Returns
        -------
        hv.Table
            Table with Object ID, x, y locations of the selected objects
        """
        self.update_points(**kwargs)
        self.table = self._table_from_points()
        return self.table

    def _table_from_points(self):
        from holoviews import Table

        # Basic table with x/y pairs
        key_dims = ["object_id"]
        value_dims = ["x", "y"] + self.data_fields

        if not len(self.points_id):
            columns = [[1]] * (len(key_dims) + len(value_dims))
            return Table(tuple(columns), key_dims, value_dims)

        # these are the object_id, x, and y columns
        columns = [self.points_id, self.points.T[0], self.points.T[1]]  # type: ignore[list-item]

        # These are the rest of the columns, pulled from metadata
        try:
            metadata = self.umap_results.metadata(self.points_idx, self.data_fields)
        except Exception as e:
            # Leave in this try/catch beause some notebook implementations dont
            # allow us to return an exception to the console.
            return Table(([str(e)]), ["message"])

        columns += [metadata[field] for field in self.data_fields]  # type: ignore[call-overload,misc,index]
        return Table(tuple(columns), key_dims, value_dims)

    @staticmethod
    def _bounding_box(points):
        import numpy as np

        # Find bounding box for the current dataset.
        xmin, xmax, ymin, ymax = (np.inf, -np.inf, np.inf, -np.inf)
        for x, y in points:
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            ymin = y if y < ymin else ymin
            ymax = y if y > ymax else ymax

        return (xmin, xmax, ymin, ymax)

    def _even_aspect_bounding_box(self):
        # Bring aspect ratio to 1:1 by expanding the smaller axis range
        (xmin, xmax, ymin, ymax) = Visualize._bounding_box(point.numpy() for point in self.umap_results)

        x_dim = xmax - xmin
        x_center = (xmax + xmin) / 2.0
        y_dim = ymax - ymin
        y_center = (ymax + ymin) / 2.0

        if x_dim > y_dim:
            ymin = y_center - x_dim / 2.0
            ymax = y_center + x_dim / 2.0
        else:
            xmin = x_center - y_dim / 2.0
            xmax = x_center + x_dim / 2.0

        return (xmin, xmax, ymin, ymax)

    def get_selected_df(self):
        r"""
        Retrieve a pandas DataFrame containing the currently selected points and their associated metadata.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per selected point and columns:
            ["object_id", "x", "y", \*additional_fields].
        """
        import pandas as pd

        if len(self.points_id) == 0:
            logger.error("No points selected")

        df = pd.DataFrame(self.points, columns=["x", "y"])
        df["object_id"] = self.points_id
        meta = self.umap_results.metadata(self.points_idx, self.data_fields)
        meta_df = pd.DataFrame(meta, columns=self.data_fields)

        cols = ["object_id", "x", "y"] + self.data_fields
        result = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
        return result.reindex(columns=cols)

    def _load_images(self, **kwargs):
        # Turn on spinner manually before loading
        self.spinner.value = True
        self.update_points(**kwargs)
        # Load images
        result = self._make_image_pane(total_width=self.plot_options["width"])
        # Turn off spinner when done
        self.spinner.value = False
        return result

    def _make_image_pane(self, total_width: int = 500, *args, **kwargs):
        """
        Sample up to 6 of the selected object_ids,
        load their FITS cutouts from [general][data_dir], and
        render as small hv.Image thumbnails in a grid.
        """
        import numpy as np
        from astropy.io import fits
        from astropy.visualization import make_lupton_rgb
        from holoviews import RGB, Image, Layout

        def style_plot(plot, element):
            bokeh_plot = plot.state
            bokeh_plot.toolbar.autohide = True
            bokeh_plot.title.text_font_size = "8pt"

        def crop_center(arr: np.ndarray, crop_shape: tuple[int, int]) -> np.ndarray:
            crop_h, crop_w = crop_shape

            if arr.ndim == 3:  # RGB case
                h, w, c = arr.shape
            else:  # Single-band case
                h, w = arr.shape

            if crop_h > h or crop_w > w:
                logger.warning(f"Crop size {crop_shape} exceeds image size {(h, w)}. Skipping crop.")
                return arr

            top = (h - crop_h) // 2
            left = (w - crop_w) // 2

            if arr.ndim == 3:
                return arr[top : top + crop_h, left : left + crop_w, :]
            else:
                return arr[top : top + crop_h, left : left + crop_w]

        n_images = 6
        n_rows = 2
        n_cols = int(n_images / n_rows)
        imgs = []

        if len(self.points_id) > 0:
            id_map = dict(zip(self.points_idx, self.points_id))

            # If we have fewer than n_images points, use all of them but force a fresh load
            if len(self.points_idx) <= n_images:
                chosen_idx = list(self.points_idx)
            else:
                chosen_idx = random.sample(list(self.points_idx), n_images)

            # Get sampled ids correspoinding to the idxs
            sampled_ids = [id_map[idx] for idx in chosen_idx]

            # Get metadata - this is in the same order as chosen_idx
            meta = self.umap_results.metadata(
                chosen_idx, [self.object_id_column_name, self.filename_column_name]
            )

            # Extract metadata directly
            # DEBUG: object_ids = meta[self.object_id_column_name]
            raw_filenames = meta[self.filename_column_name]

            filenames = [f.decode("utf-8") for f in raw_filenames]

        else:
            sampled_ids = []
            filenames = []

        base_dir = Path(self.umap_results.original_config["general"]["data_dir"])
        crop_to = self.umap_results.original_config["data_set"]["crop_to"]

        # Defining a Fallback Image to Display in case of errors
        # Matching Shape is important because otherwise Haloviews'
        # DynamicMap fails silently
        if len(self.torch_tensor_bands) == 3:
            placeholder_arr = np.full((64, 64, 3), 1.0)
        else:
            placeholder_arr = np.full((64, 64), 1.0)

        for i in range(n_images):
            if i < len(sampled_ids):
                try:
                    cutout_path = Path(filenames[i])
                    if not cutout_path.is_absolute():
                        cutout_path = base_dir / cutout_path

                    if cutout_path.suffix.lower() == ".fits":
                        arr = fits.getdata(cutout_path)
                    elif cutout_path.suffix.lower() == ".pt":
                        tensor = torch.load(cutout_path, map_location="cpu", weights_only=True)

                        if len(self.torch_tensor_bands) == 1:
                            # Single-band extraction
                            band_idx = self.torch_tensor_bands[0]
                            arr = tensor[band_idx].numpy()
                        else:
                            # RGB extraction (3 bands)
                            rgb_arrays = []
                            for band_idx in self.torch_tensor_bands:
                                rgb_arrays.append(tensor[band_idx].numpy())
                            # Stack along new axis to create (H, W, 3) RGB array
                            arr = np.stack(rgb_arrays, axis=-1)
                    else:
                        raise ValueError(
                            f"Unsupported file format: {cutout_path.suffix}. Currently\
                                           the visualize module only supports FITS and PyTorch files"
                        )

                    if crop_to:
                        arr = crop_center(arr, crop_to)

                    # Handle normalization differently for single-band vs RGB
                    if arr.ndim == 3:
                        # Use astropy's Lupton RGB
                        # arr shape is (H, W, 3) but make_lupton_rgb expects (r, g, b) as separate arrays
                        r_band = arr[:, :, 0]
                        g_band = arr[:, :, 1]
                        b_band = arr[:, :, 2]

                        # make_lupton_rgb applies an asinh stretch and returns values in [0, 1] range
                        # Use configurable options for make_lupton_rgb
                        arr = make_lupton_rgb(r_band, g_band, b_band, **self.make_lupton_rgb_opts)
                    else:  # Single-band case
                        # Ensure data is positive for log scaling
                        min_positive = np.min(arr[arr > 0]) if np.any(arr > 0) else 1e-10
                        arr = np.maximum(
                            arr, min_positive
                        )  # Replace zeros/negatives with minimum positive value

                        # Apply LogNorm-like scaling
                        norm = LogNorm(vmin=min_positive, vmax=np.max(arr))
                        arr = norm(arr)

                    # DEBUG: title = f"{chosen_idx[i]}:{object_ids[i]}\n{sampled_ids[i]}"
                    title = f"{sampled_ids[i]}"

                except Exception as e:
                    logger.warning(f"Could not load FITS file: {e}")
                    with open("./hyrax_visualize.log", "a") as f:
                        f.write(f"Could not load FITS file: {e}\n")
                    arr = placeholder_arr
                    title = f"NL:{sampled_ids[i]}"
            else:
                arr = placeholder_arr
                title = "No Selection"

            # Configure image options based on array dimensions
            img_opts = {
                "width": int((0.9 * total_width) / n_cols),
                "height": int((0.9 * total_width) / n_cols),
                "title": title,
                "tools": [],
                "shared_axes": False,
                "hooks": [style_plot],
                "xaxis": None,
                "yaxis": None,
            }

            if arr.ndim == 3:  # RGB case
                img = RGB(arr).opts(**img_opts)
            else:  # Single-band case
                img_opts["cmap"] = "gray_r"
                img = Image(arr).opts(**img_opts)
            imgs.append(img)

        return Layout(imgs).cols(n_cols)
