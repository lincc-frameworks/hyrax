import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np
from holoviews import Points, Table

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

    def run(self, input_dir: Optional[Union[Path, str]] = None, **kwargs):
        """Generate an interactive notebook visualization of a latent space that has been umapped down to 2d.

        The plot contains two holoviews objects, a scatter plot of the latent space, and a table of objects
        which can be populated by selecting from the scatter plot.

        Parameters
        ----------
        input_dir : Optional[Union[Path, str]], optional
            Directory holding the output from the 'umap' verb, by default None. When not provided, we use
            the most recent umap int he current results directory.

        **kwargs :
            Keyword arguments are passed through as options for the plot object as
            `plot_pane.opts(**plot_options)`. It is not recommended to override the "tools" plot option,
            because that will break the integration between the plot selection operations and the table.

        Returns
        -------
        Holoview
            Combined holoview object
        """
        from holoviews import DynamicMap, extension
        from holoviews.operation.datashader import dynspread, rasterize
        from holoviews.streams import Lasso, RangeXY, SelectionXY, Tap
        from scipy.spatial import KDTree

        from hyrax.data_sets.inference_dataset import InferenceDataSet

        fields = ["object_id"]
        fields += self.config["visualize"]["fields"]

        # Get the umap data and put it in a kdtree for indexing.
        self.umap_results = InferenceDataSet(self.config, results_dir=input_dir, verb="umap")

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

        # Initialize holoviews with bokeh.
        extension("bokeh")

        # Set up the plot pane
        xmin, xmax, ymin, ymax = self._even_aspect_bounding_box()
        plot_options = {
            "tools": ["box_select", "lasso_select", "tap"],
            "width": 500,
            "height": 500,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax),
            "cnorm": "eq_hist",
        }
        plot_options.update(kwargs)

        plot_dm = DynamicMap(self.visible_points, streams=[RangeXY()])
        plot_pane = dynspread(rasterize(plot_dm).opts(**plot_options))

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
        self.table = self._table_from_points()
        table_options = {"width": plot_options["width"]}
        table_pane = DynamicMap(self.selected_objects, streams=table_streams).opts(**table_options)

        # Return the plot pane and table pane as a combined object
        return plot_pane + table_pane

    def visible_points(self, x_range: Union[tuple, list], y_range: Union[tuple, list]) -> Points:
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
        from holoviews import Points

        if x_range is None or y_range is None:
            return Points([])

        return Points(self.box_select_points(x_range, y_range)[0])

    def selected_objects(self, **kwargs) -> Table:
        """Generate the holoview table for a selected set of objects based on input from the
        Lasso, Tap, and SelectionXY streams.

        This is the main UI event handler for selection tools on the plot, and is called by Holoviews.

        This function accepts the data values from all streams and uses the differences between the current
        call and prior calls to differentiate between different UI events.

        The self.prev_kwargs dictionary is used to store previous calls to this function, and the _called_*
        helpers perform the differencing for each case.

        Returns
        -------
        hv.Table
            Table with Object ID, x, y locations of the selected objects
        """

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
            # We return whatever cached table state we have if we were not called by any event
            # This normally happens during initialization.
            self.prev_kwargs = kwargs
            return self.table

        self.table = self._table_from_points()

        self.prev_kwargs = kwargs
        return self.table

    def _table_from_points(self) -> Table:
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

    def poly_select_points(self, geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        indexes = self.box_select_indexes(x_range, y_range)
        ids = np.array(list(self.umap_results.ids()))[indexes]
        return self.umap_results[indexes].numpy(), ids, indexes

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
        # Find center
        xc = (x_range[0] + x_range[1]) / 2.0
        yc = (y_range[0] + y_range[1]) / 2.0
        query_pt = [xc, yc]

        # Find larger of  half-width and half-height to use as our search radius.
        radius = np.max([np.max(x_range) - xc, np.max(y_range) - yc])

        return self.tree.query_ball_point(query_pt, radius, p=np.inf)

    @staticmethod
    def _bounding_box(points):
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
