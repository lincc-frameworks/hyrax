"""Streaming dataset that reads rows from a HATS catalog via LSDB.

:class:`LSDBStreamDataset` is a :class:`torch.utils.data.IterableDataset` that wraps
``lsdb.streams.CatalogStream`` (a single finite pass over every partition) or
``lsdb.streams.InfiniteStream`` (endless random resampling of partitions), so a HATS
catalog can be trained on or inferred over without ever materializing it in memory. This
is the streaming counterpart to :class:`~hyrax.datasets.hats_dataset.HyraxHATSDataset`,
which calls ``catalog.compute()`` and holds the whole table.

Rule of thumb: use ``stream_type = "catalog"`` for **inference** (every object is visited
exactly once and the run ends on its own when the catalog is exhausted) and
``stream_type = "infinite"`` for **training** (batches keep arriving until you stop the
session, so training is not bounded by a single pass).

**Fixed-size batching.** LSDB yields one ``pandas.DataFrame`` per *chunk of partitions*, so
chunk sizes are ragged - a catalog may hand back 9 rows, then 10, then 171. This dataset
buffers rows across chunks and splits large chunks, so every yielded batch holds exactly
``[data_loader] batch_size`` rows. Only the final batch, emitted when a finite stream is
exhausted or ``stop()`` is called, may be short.

**Specifying the catalog.** ``data_location`` is either a HATS path/URL, which is opened
with ``lsdb.open_catalog``, or an ``lsdb://<name>`` handle referring to an in-memory
catalog registered with ``register_catalog()``. The registry exists because derived
catalogs (``gaia.crossmatch(tess).query(...)``) only live in memory and cannot be placed in
the config: Hyrax serializes the whole runtime config with ``tomlkit`` when a verb starts,
which a live ``lsdb.Catalog`` object would break.

.. code-block:: python

    import hyrax
    import lsdb
    from hyrax.datasets import LSDBStreamDataset

    gaia = lsdb.open_catalog("https://data.lsdb.io/hats/gaia_dr3")
    data_location = LSDBStreamDataset.register_catalog(
        "gaia_bright", gaia.query("phot_g_mean_mag < 20")
    )

    hy = hyrax.Hyrax()
    hy.config["data_request"] = {
        "infer_stream": {
            "data": {
                "dataset_class": "LSDBStreamDataset",
                "data_location": data_location,
                "primary_id_field": "_healpix_29",
                "fields": ["ra", "dec", "phot_g_mean_mag"],
            }
        }
    }

    with hy.infer_stream() as session:  # builds the provider + loader internally
        for batch, results in session:
            ...

.. warning::
    Nested columns (light curves, spectra) are not supported in this version. A nested
    column arrives as a per-row sub-DataFrame of varying length, and stacking those into a
    batch fails in ``numpy`` with "setting an array element with a sequence ...
    inhomogeneous shape". Request only scalar or fixed-width columns, or define a
    ``collate_<field>`` method on a subclass to handle the ragged field yourself.
    Fixed-length nested columns do collate correctly, to shape ``(batch, k, n_columns)``.

.. warning::
    The stream owns a single in-process iterator, so the loader must run with
    ``num_workers = 0`` (the default applied by ``dist_data_loader``). A plain
    ``IterableDataset`` does no worker sharding, so with multiple workers each would build
    its own stream and every row would be emitted once per worker.
"""

import logging
import threading
from pathlib import Path
from types import MethodType

import numpy as np
from torch.utils.data import IterableDataset

from .dataset_registry import HyraxDataset

logger = logging.getLogger(__name__)

LSDB_URI_PREFIX = "lsdb://"

# Catalogs registered by register_catalog(), keyed by the name used in an "lsdb://<name>"
# data_location. This is process-local and is not inherited by DataLoader worker processes,
# which is only safe because dist_data_loader forces num_workers = 0 for iterable datasets.
_CATALOG_REGISTRY: dict[str, object] = {}


class LSDBStreamDataset(HyraxDataset, IterableDataset):
    """Streams rows from a HATS catalog and yields fixed-size batches.

    The stream is configured in ``[data_set.LSDBStreamDataset]``; the catalog itself comes
    from the ``data_location`` of the data request, either as a HATS path/URL or as an
    ``lsdb://<name>`` handle into the in-memory catalog registry.

    Each row becomes a flat ``dict`` of column name to value (e.g.
    ``{"_healpix_29": 2787..., "ra": 95.4, "dec": -36.3}``); the wrapping
    :class:`~hyrax.datasets.streaming_data_provider.StreamingDataProvider` turns each flat
    sample into the structured form the collation + model machinery expect.
    """

    def __init__(self, config: dict, data_location=None):
        # The config block is looked up by literal name rather than type(self).__name__ so
        # that user subclasses read the same defaults instead of needing their own block.
        ds_config = config["data_set"]["LSDBStreamDataset"]

        if data_location is None or data_location is False:
            raise ValueError(
                "LSDBStreamDataset requires a `data_location`: either a path/URL to a HATS "
                "catalog, or 'lsdb://<name>' naming a catalog passed to "
                "LSDBStreamDataset.register_catalog()."
            )

        # Kept as the raw string (including any "lsdb://" prefix) because
        # _requested_columns_from_config matches data request entries against it.
        self.data_location = str(data_location)

        self.stream_type = str(ds_config["stream_type"]).lower()
        if self.stream_type not in ("catalog", "infinite"):
            raise ValueError(
                "config['data_set']['LSDBStreamDataset']['stream_type'] must be "
                f"'catalog' or 'infinite', got {ds_config['stream_type']!r}."
            )

        # `false` is the TOML "not set" sentinel, and bool is a subclass of int, so
        # int(False) would silently become 0 here. lsdb clips partitions_per_chunk with
        # min(value, npartitions), and a 0 makes it stream the entire catalog as one chunk.
        partitions_per_chunk = ds_config["partitions_per_chunk"]
        if partitions_per_chunk is False or int(partitions_per_chunk) < 1:
            raise ValueError(
                "config['data_set']['LSDBStreamDataset']['partitions_per_chunk'] must be a "
                f"positive integer, got {partitions_per_chunk!r}."
            )
        self.partitions_per_chunk = int(partitions_per_chunk)

        self.shuffle = bool(ds_config["shuffle"])

        # `seed = 0` is a legitimate seed, so this tests the `false` sentinel identity
        # rather than falsiness.
        seed = ds_config["seed"]
        self.seed = None if seed is False else int(seed)

        # The batch size every yielded batch is padded up to; see __iter__.
        batch_size = config["data_loader"]["batch_size"]
        if not batch_size or int(batch_size) < 1:
            raise ValueError(
                f"config['data_loader']['batch_size'] must be a positive integer, got {batch_size!r}."
            )
        self.batch_size = int(batch_size)

        # Populated as a side effect: the primary id field and whether fields were listed.
        self._primary_id_field = None
        self._explicit_fields = False
        requested_columns = self._requested_columns_from_config(config)

        self._catalog = self._resolve_catalog(ds_config, requested_columns)

        # Set from another thread (or session teardown) to end iteration; see stop().
        self._stop = threading.Event()

        # Single shared iterator, created lazily. Shared between peek_sample() and
        # __iter__ so peeked rows can be replayed into the first batch.
        self._iterator = None
        self._exhausted = False
        self._buffered: list[dict] = []
        self._peek_index = 0
        self._row_checked = False

        super().__init__(config, metadata_table=None)

        # Dynamic getter creation - assumes only one level of nesting.
        all_columns = set(self._catalog.columns)
        nested_columns = set(self._catalog.nested_columns)
        self._register_getters(list(all_columns - nested_columns))

        # For each of the nested columns, get all the subcolumns
        for nested_column in nested_columns:
            nested_subcols = list(self._catalog[nested_column].columns)
            self._register_nested_getters(nested_column, nested_subcols)

    def _register_getters(self, columns) -> None:
        def _make_getter(field_name: str):
            def getter(self, sample, _field_name=field_name):
                return sample[_field_name]

            return getter

        for field_name in columns:
            method_name = f"get_{field_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(field_name), self))

    def _register_nested_getters(self, nested_column, nested_subcolumns) -> None:
        def _make_getter(nested_column: str, subnested_column: str):
            def getter(self, sample, _nested_name=nested_column, _subnested_name=subnested_column):
                return np.asarray(sample[_nested_name][_subnested_name])

            return getter

        for subnested_column in nested_subcolumns:
            method_name = f"get_{nested_column}_{subnested_column}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(nested_column, subnested_column), self))

    #
    # In-memory catalog registry
    #

    @classmethod
    def register_catalog(cls, name: str, catalog) -> str:
        """Register an in-memory LSDB catalog and return its ``lsdb://`` data_location.

        Use this for catalogs that cannot be expressed as a path, such as the result of a
        ``crossmatch`` or ``query``. The returned handle is an ordinary string, so it can
        be stored in the config and serialized with the rest of the runtime config.

        Parameters
        ----------
        name : str
            Name to register the catalog under. Must not contain whitespace or ``/``.
        catalog : lsdb.Catalog
            The catalog to stream from.

        Returns
        -------
        str
            The ``lsdb://<name>`` string to use as the request's ``data_location``.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Catalog name must be a non-empty string, got {name!r}.")
        if "/" in name or any(char.isspace() for char in name):
            raise ValueError(
                f"Catalog name {name!r} must not contain whitespace or '/' so that it "
                "round-trips through an 'lsdb://<name>' data_location."
            )

        import lsdb

        if not isinstance(catalog, lsdb.Catalog):
            raise TypeError(
                f"register_catalog() expects an lsdb.Catalog, got {type(catalog).__name__}. "
                "Open one with lsdb.open_catalog(), or derive one with .query()/.crossmatch()."
            )

        if name in _CATALOG_REGISTRY:
            logger.warning(f"Replacing the catalog already registered under '{name}'.")

        _CATALOG_REGISTRY[name] = catalog
        return f"{LSDB_URI_PREFIX}{name}"

    @classmethod
    def unregister_catalog(cls, name: str) -> None:
        """Remove a registered catalog. Does nothing if the name is not registered."""
        _CATALOG_REGISTRY.pop(name, None)

    @classmethod
    def clear_catalogs(cls) -> None:
        """Remove every registered catalog."""
        _CATALOG_REGISTRY.clear()

    @classmethod
    def registered_catalogs(cls) -> list[str]:
        """Return the sorted names of all registered catalogs."""
        return sorted(_CATALOG_REGISTRY)

    @classmethod
    def _lookup_catalog(cls, name: str):
        """Return the catalog registered under ``name``, or raise a helpful KeyError."""
        try:
            return _CATALOG_REGISTRY[name]
        except KeyError as err:
            raise KeyError(
                f"No lsdb catalog is registered under '{name}'. "
                f"Registered catalogs: {cls.registered_catalogs()}. Register one first with "
                f"LSDBStreamDataset.register_catalog('{name}', catalog)."
            ) from err

    #
    # Catalog and column resolution
    #

    def _resolve_catalog(self, ds_config: dict, requested_columns):
        """Look up a registered catalog, or open the HATS catalog at ``data_location``."""
        if self.data_location.startswith(LSDB_URI_PREFIX):
            # A plain prefix strip rather than urlparse().netloc, which would apply
            # host/port semantics and mangle any name containing a colon.
            return self._lookup_catalog(self.data_location[len(LSDB_URI_PREFIX) :])

        import lsdb

        open_catalog_kwargs = dict(ds_config["open_catalog_kwargs"] or {})
        if requested_columns and "columns" not in open_catalog_kwargs:
            open_catalog_kwargs["columns"] = requested_columns

        return lsdb.open_catalog(self.data_location, **open_catalog_kwargs)

    def _requested_columns_from_config(self, config: dict):
        """Derive the columns to read from the data request, or None to read them all.

        Mirrors ``HyraxHATSDataset._requested_columns_from_config``, but matches
        requests on the raw ``data_location`` string so that ``lsdb://`` handles work, and
        ignores ``join_field`` because ``StreamingDataProvider`` rejects joined datasets.
        """
        from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

        data_request = config.get("data_request") or config.get("model_inputs") or {}
        requested_columns = set()

        for request_group in data_request.values():
            for dataset_definition in request_group.values():
                if dataset_definition.get("dataset_class") != type(self).__name__:
                    continue
                # .get(), not [...]: a streaming request may omit data_location entirely.
                if not self._same_location(dataset_definition.get("data_location"), self.data_location):
                    continue

                primary_id_field = dataset_definition.get("primary_id_field")
                if primary_id_field:
                    self._primary_id_field = primary_id_field

                # If any dataset request has no fields specified, that means we need all the
                # columns no matter what any other request group says.
                fields = dataset_definition.get("fields") or []
                if not fields:
                    self._explicit_fields = False
                    return None

                self._explicit_fields = True
                requested_columns.update(fields)
                if primary_id_field:
                    requested_columns.add(primary_id_field)

        # The spatial index is the DataFrame index rather than a column, so requesting it
        # is meaningless; _flatten_index restores it as a column instead.
        requested_columns.discard(SPATIAL_INDEX_COLUMN)

        return sorted(requested_columns)

    @staticmethod
    def _same_location(candidate, target) -> bool:
        """Compare two data_location values, tolerating URIs and unresolved paths."""
        if candidate is None or target is None:
            return False

        candidate, target = str(candidate), str(target)
        if candidate == target:
            return True

        # URIs ("lsdb://", "s3://", "https://") have no filesystem path semantics.
        if "://" in candidate or "://" in target:
            return False

        try:
            return Path(candidate).expanduser().resolve() == Path(target).expanduser().resolve()
        except (OSError, ValueError):
            return False

    #
    # Stream lifecycle
    #

    def _make_stream(self):
        """Build the LSDB stream over the catalog.

        Built lazily (not in ``__init__``) so that constructing the dataset stays cheap,
        and so tests can substitute a stream of plain DataFrames.
        """
        from lsdb.streams import CatalogStream, InfiniteStream

        if self._catalog.npartitions < 1:
            raise ValueError(f"The catalog at '{self.data_location}' has no partitions to stream.")

        if self.stream_type == "infinite":
            if not self.shuffle:
                logger.warning(
                    "config['data_set']['LSDBStreamDataset']['shuffle'] is ignored when "
                    "stream_type = 'infinite'; lsdb's InfiniteStream always shuffles."
                )
            return InfiniteStream(
                self._catalog,
                partitions_per_chunk=self.partitions_per_chunk,
                seed=self.seed,
            )

        return CatalogStream(
            self._catalog,
            partitions_per_chunk=self.partitions_per_chunk,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    def _ensure_iterator(self):
        """Return the shared stream iterator, creating it on first use.

        ``CatalogStream.__iter__`` hands back a *new* iterator with a freshly spawned RNG
        on every call, so peek_sample() and __iter__() must share a single one or peeked
        rows would be drawn from a different traversal than the batches.
        """
        if self._iterator is None:
            self._exhausted = False
            self._iterator = iter(self._make_stream())
        return self._iterator

    def stop(self):
        """Signal :meth:`__iter__` to flush any pending rows and stop fetching chunks.

        LSDB offers no cancellation or timeout, so a :meth:`__iter__` already blocked
        inside a chunk computation cannot be interrupted; the stop takes effect at the next
        chunk boundary. In the usual flow this is invisible, because the session calls
        ``stop()`` from the same thread after iteration has ended. To end an ``infinite``
        stream promptly, ``break`` out of the session loop rather than calling ``stop()``
        from another thread. A smaller ``partitions_per_chunk`` bounds the delay.
        """
        self._stop.set()

    def __len__(self):
        """A stream has no length.

        Defined only so ``HyraxDataset.__init_subclass__`` (which requires a ``__len__``
        attribute) accepts the class. The iterable branch of ``dist_data_loader`` never
        calls it.
        """
        raise TypeError("LSDBStreamDataset is a stream and has no length.")

    #
    # Row production
    #

    @staticmethod
    def _flatten_index(frame):
        """Promote a named index to a real column so it can be used as a field.

        LSDB chunks carry the HATS spatial index (``_healpix_29``) as the DataFrame index,
        which makes it the only usable ``primary_id_field`` for a crossmatched catalog with
        no natural id of its own.
        """
        name = frame.index.name
        if name is not None and name not in frame.columns:
            frame = frame.reset_index()
        return frame

    def _check_first_row(self, row: dict):
        """Validate the first row produced, once per dataset."""
        self._row_checked = True

        if self._primary_id_field is not None and self._primary_id_field not in row:
            raise RuntimeError(
                f"The primary_id_field '{self._primary_id_field}' is not a column of the "
                f"catalog at '{self.data_location}'. Available columns: {sorted(row)}."
            )

        if not self._explicit_fields:
            logger.warning(
                "No 'fields' were listed in the data request for LSDBStreamDataset, so every "
                f"column becomes a model input: {sorted(row)}. Set 'fields' to select only "
                "the columns the model should see."
            )

    def peek_sample(self) -> dict:
        """Return one row without removing it from the batch stream.

        Pulls chunks until a row is available, buffers them so :meth:`__iter__` replays
        them as part of the first batch, and advances a cursor so repeated calls return
        distinct rows. Used to pre-flight the model architecture without losing data.

        Returns
        -------
        dict
            The flat row, mapping column name to value.

        Raises
        ------
        RuntimeError
            If the stream is stopped, or produces no rows, before one is available.
        """
        iterator = self._ensure_iterator()

        # A `while`, not an `if`: a partition can legitimately compute to zero rows.
        while self._peek_index >= len(self._buffered):
            if self._stop.is_set():
                raise RuntimeError("LSDBStreamDataset.peek_sample() was stopped before a row arrived.")
            try:
                chunk = next(iterator)
            except StopIteration as err:
                raise RuntimeError(
                    f"LSDBStreamDataset.peek_sample(): the catalog at '{self.data_location}' "
                    "produced no rows."
                ) from err
            self._buffered.extend(self._flatten_index(chunk).to_dict(orient="records"))

        sample = self._buffered[self._peek_index]
        self._peek_index += 1
        if not self._row_checked:
            self._check_first_row(sample)
        return sample

    def __iter__(self):
        """Yield ``list[dict]`` batches of exactly ``batch_size`` rows.

        Rows are buffered across chunks and large chunks are split, so only the final
        batch - emitted when a finite stream is exhausted or :meth:`stop` is set - may be
        shorter than ``batch_size``.
        """
        iterator = self._ensure_iterator()

        # Replay any peeked-but-not-yet-delivered rows ahead of the first batch.
        batch: list[dict] = list(self._buffered)
        self._buffered = []
        self._peek_index = 0

        # The chunk currently being drained, and how far into it we have read. Tracked out
        # here so the `finally` can push back the rows of a half-drained chunk.
        frame = None
        taken = 0

        try:
            # Establishes len(batch) < batch_size before the chunk loop. Without this, a
            # `need` of zero below would make every window empty and the loop spin forever.
            while len(batch) >= self.batch_size:
                full, batch = batch[: self.batch_size], batch[self.batch_size :]
                yield full

            while not self._stop.is_set():
                try:
                    chunk = next(iterator)
                except StopIteration:
                    # Required, not decorative: under PEP 479 a StopIteration escaping a
                    # generator becomes "RuntimeError: generator raised StopIteration",
                    # which would turn every clean end-of-catalog into a crash.
                    self._exhausted = True
                    frame, taken = None, 0
                    break

                frame = self._flatten_index(chunk)
                taken, n_rows = 0, len(frame)
                while taken < n_rows:
                    need = self.batch_size - len(batch)
                    window = frame.iloc[taken : taken + need]
                    rows = window.to_dict(orient="records")
                    if rows and not self._row_checked:
                        self._check_first_row(rows[0])
                    batch.extend(rows)
                    taken += len(window)

                    if len(batch) >= self.batch_size:
                        # Clear before yielding: if the consumer stops here, the `finally`
                        # below must not re-buffer a batch that was actually delivered.
                        full, batch = batch, []
                        yield full

                # Note there is no early exit on self._stop here: a chunk already fetched is
                # always drained, so stop() never silently discards rows. It takes effect on
                # the outer loop, which stops fetching *new* chunks.

            # Only the final batch is allowed to be short. This must stay inside the `try`
            # body: yielding while unwinding a GeneratorExit raises RuntimeError.
            if batch:
                final, batch = batch, []
                yield final
        finally:
            # Rows pulled from the catalog but never yielded - because the consumer broke
            # out of its loop, leaving both a partial batch and the undrained tail of the
            # current chunk - are pushed back so a later pass resumes exactly where this
            # one stopped, in order and without duplicates.
            leftover = frame.iloc[taken:].to_dict(orient="records") if frame is not None else []
            if batch or leftover:
                self._buffered = batch + leftover + self._buffered
            if self._exhausted or self._stop.is_set():
                self._iterator = None

    def collate_lightcurve(self, samples):
        """
        TODO this is experimental and probably doesn't belong here.
        It was created for testing, and should either be solidified or removed
        before this code is merged to main.
        """
        import numpy as np

        result = {}

        vals = [s["lightcurve"]["sap_flux"] for s in samples]
        lengths = np.array([len(s) for s in vals], dtype=np.int64)
        max_len = int(lengths.max())

        padded = np.zeros((len(vals), max_len), dtype=np.float32)
        mask = np.zeros((len(vals), max_len), dtype=np.int64)
        for i, s in enumerate(vals):
            padded[i, : lengths[i]] = s
            mask[i, : lengths[i]] = 1

        result["lightcurve"] = padded
        result["lightcurve_lengths"] = lengths
        result["lightcurve_mask"] = mask

        return result
