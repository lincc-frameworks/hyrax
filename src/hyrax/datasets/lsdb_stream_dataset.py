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

**Specifying the catalog.** ``data_location`` is an ``lsdb://<name>`` handle referring
to an in-memory catalog registered with ``register_catalog()``.

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
                "primary_id_field": "gaia_id",
                "fields": ["ra", "dec", "phot_g_mean_mag"],
            }
        }
    }

    with hy.infer_stream() as session:  # builds the provider + loader internally
        for batch, results in session:
            ...

.. note::
    Nested columns are collated by default. Variable-length nested arrays are padded to the
    longest row in each batch, in the column's own dtype, and accompanied by a boolean
    ``<field>_mask``; fixed-length nested columns are stacked directly. Padding is whatever
    ``numpy`` zero-initializes that dtype to (``0``, or ``""`` for a string column), so the
    mask is the only reliable record of what is real. Define a ``collate_<field>`` method on
    a subclass when a different representation is needed.

.. warning::
    The stream owns a single in-process iterator, so the loader must run with
    ``num_workers = 0`` (the default applied by ``dist_data_loader``). A plain
    ``IterableDataset`` does no worker sharding, so with multiple workers each would build
    its own stream and every row would be emitted once per worker.

**Keeping the consumer fed.** Because the loader runs with ``num_workers = 0``, PyTorch's own
``prefetch_factor`` does nothing here and this dataset has to hide fetch latency itself. Three
settings do that, and they stack:

``use_dask_client``
    lsdb's ``CatalogIterator`` submits the *next* chunk's future before returning the current
    one, but only a real ``dask.distributed`` client makes that submission asynchronous -
    without one it computes inline, so every fetch blocks. Hyrax attaches to whatever client
    is already active, so create one before the run starts. On a single machine prefer
    ``Client(processes=False)``: it still gives real futures, but keeps results in-process
    rather than serializing every chunk back from a worker, which is a real cost for the
    nested frames light-curve catalogs produce.

``partitions_per_chunk``
    How much work one fetch is worth. Raising it amortizes per-chunk overhead over more rows
    and, with a client, spreads a single chunk across more workers; lsdb recommends at least
    twice the worker count. The cost is memory.

Enable DEBUG logging on this module to see where the time actually goes - fetch, decode, and
how long the consumer blocked. When prefetching is working, the consumer wait falls toward
zero while the fetch time does not.
"""

import logging
import threading
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
    from the ``data_location`` of the data request, as an ``lsdb://<name>`` handle
    of the in-memory catalog registry.

    Each row becomes a flat ``dict`` of column name to value (e.g.
    ``{"object_id": 2787..., "ra": 95.4, "dec": -36.3}``); the wrapping
    :class:`~hyrax.datasets.streaming_data_provider.StreamingDataProvider` turns
    each flat sample into the structured form the collation + model machinery expect.
    """

    def __init__(self, config: dict, data_location=None):
        # The config block is looked up by literal name rather than type(self).__name__ so
        # that user subclasses read the same defaults instead of needing their own block.
        ds_config = config["data_set"]["LSDBStreamDataset"]

        if data_location is None or data_location is False:
            raise ValueError(
                "LSDBStreamDataset requires a `data_location` as a 'lsdb://<name>' "
                "naming a catalog passed to LSDBStreamDataset.register_catalog()."
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

        self.use_dask_client = bool(ds_config["use_dask_client"])

        dask_client_address = ds_config["dask_client_address"]
        self.dask_client_address = None if dask_client_address is False else str(dask_client_address)

        # The batch size every yielded batch is padded up to; see __iter__.
        batch_size = config["data_loader"]["batch_size"]
        if not batch_size or int(batch_size) < 1:
            raise ValueError(
                f"config['data_loader']['batch_size'] must be a positive integer, got {batch_size!r}."
            )
        self.batch_size = int(batch_size)

        self._catalog = self._lookup_catalog(self.data_location[len(LSDB_URI_PREFIX) :])

        # Set from another thread (or session teardown) to end iteration; see stop().
        self._stop = threading.Event()

        # Single shared iterator, created lazily. Shared between peek_sample() and
        # __iter__ so peeked rows can be replayed into the first batch.
        self._iterator = None
        self._exhausted = False
        self._buffered: list[dict] = []
        self._peek_index = 0

        # A dask Client this dataset created (and therefore owns) via
        # A client attached via get_client() is owned by the caller and is never
        # stored here; see _resolve_dask_client() and close().
        self._owned_client = None

        super().__init__(config, metadata_table=None)

        # Dynamic getter creation - assumes only one level of nesting.
        all_columns = set(self._catalog.columns)
        nested_columns = set(self._catalog.nested_columns)
        self._register_getters(list(all_columns - nested_columns))

        # For each of the nested columns, get all the subcolumns
        for nested_column in nested_columns:
            nested_subcols = list(self._catalog[nested_column].columns)
            self._register_nested_getters(nested_column, nested_subcols)
            self._register_nested_collators(nested_column, nested_subcols)

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

    def _register_nested_collators(self, nested_column, nested_subcolumns) -> None:
        def _make_collator(nested_column: str, subnested_column: str):
            def collator(self, batch, _nested_name=nested_column, _subnested_name=subnested_column):
                # Get the length of each subnested array in the batch
                col_name = f"{_nested_name}_{_subnested_name}"
                arrays = [np.asarray(sample[col_name]) for sample in batch]
                lengths = [len(array) for array in arrays]
                max_length = max(lengths)

                # Pad in the column's own dtype rather than float64. A string column (a
                # photometric band, say) cannot be assigned into a float array at all, and
                # an integer column would otherwise be silently widened to float.
                dtype = np.result_type(*(array.dtype for array in arrays))

                # Create a padded array and mask with shape (# arrays, max length)
                padded_batch = np.zeros((len(batch), max_length), dtype=dtype)
                mask = np.zeros((len(batch), max_length), dtype=bool)
                for i, array in enumerate(arrays):
                    padded_batch[i, : lengths[i]] = array
                    mask[i, : lengths[i]] = True
                return {
                    col_name: padded_batch,
                    f"{col_name}_mask": mask,
                }

            return collator

        for subnested_column in nested_subcolumns:
            method_name = f"collate_{nested_column}_{subnested_column}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_collator(nested_column, subnested_column), self))

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
    # Stream lifecycle
    #

    def _resolve_dask_client(self):
        """Return the dask client to hand lsdb, or ``None`` to compute synchronously.

        This is the difference between a stream that overlaps with training and one that
        does not. ``CatalogIterator.__next__`` submits the *next* chunk before returning
        the current one, but with no client lsdb's ``submit_next_partitions`` computes
        inline and hands back an already-resolved stand-in, so every fetch blocks.
        """
        if not self.use_dask_client:
            return None

        from dask.distributed import Client, get_client

        if self.dask_client_address:
            if self._owned_client is None:
                logger.info(
                    f"Connecting the lsdb stream to the dask scheduler at {self.dask_client_address}."
                )
                self._owned_client = Client(self.dask_client_address)
                logger.info(f"Dask client dashboard: {self._owned_client.dashboard_link}")
            return self._owned_client

        try:
            client = get_client()
        except ValueError:
            logger.info(
                "No active dask.distributed Client was found, so lsdb will compute each chunk "
                "synchronously and every fetch will block the consumer. Create a client before "
                "starting the run - Client(processes=False) keeps results in-process and avoids "
                "serializing every chunk back from a worker."
            )
            return None

        logger.info(f"The lsdb stream is using the active dask client: {client}")
        return client

    def _make_stream(self):
        """Build the LSDB stream over the catalog.

        Built lazily (not in ``__init__``) so that constructing the dataset stays cheap, so
        a dask client created around the run is picked up, and so tests can substitute a
        stream of plain DataFrames.
        """
        from lsdb.streams import CatalogStream, InfiniteStream

        if self._catalog.npartitions < 1:
            raise ValueError(f"The catalog at '{self.data_location}' has no partitions to stream.")

        client = self._resolve_dask_client()

        if self.stream_type == "infinite":
            if self.shuffle:
                logger.warning(
                    "config['data_set']['LSDBStreamDataset']['shuffle'] is ignored when "
                    "stream_type = 'infinite'; lsdb's InfiniteStream always shuffles."
                )
            return InfiniteStream(
                self._catalog,
                client=client,
                partitions_per_chunk=self.partitions_per_chunk,
                seed=self.seed,
            )

        return CatalogStream(
            self._catalog,
            client=client,
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

    def close(self):
        """Stop iteration and close the dask client this dataset created, if any. Idempotent.

        Only a client created here from ``dask_client_address`` is closed. A client
        attached via the active-client fallback (``get_client()``) is owned by whoever
        created it - closing it here would pull it out from under the rest of the
        session. Skipping this for an owned client leaves its scheduler registration
        and background threads running after the stream is done with it.

        ``_owned_client is None`` is the guard, so a second call is a no-op rather than
        a double close.
        """
        self._stop.set()

        try:
            if self._owned_client is None:
                return
            else:
                self._owned_client.close()
                self._owned_client = None
        except Exception as err:
            # Never let teardown replace the exception that triggered it.
            logger.warning(f"Error closing dask client: {err}")

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
            self._buffered.extend(chunk.to_dict(orient="records"))

        sample = self._buffered[self._peek_index]
        self._peek_index += 1
        return sample

    def __iter__(self):
        """Yield ``list[dict]`` batches of exactly ``batch_size`` rows.

        Rows are buffered across chunks and large chunks are split, so only the final
        batch - emitted when a finite stream is exhausted or :meth:`stop` is set - may be
        shorter than ``batch_size``.

        Prefetching changes *when* chunks are computed, never which rows come out or in
        what order: the loop below still refuses to take a new chunk once :meth:`stop` is
        set, and every prefetched row the consumer does not reach is pushed back.
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
                # Inside the loop, so a dataset stopped before iteration starts no thread.
                try:
                    chunk = next(iterator)
                except StopIteration:
                    # Required, not decorative: under PEP 479 a StopIteration escaping a
                    # generator becomes "RuntimeError: generator raised StopIteration",
                    # which would turn every clean end-of-catalog into a crash.
                    self._exhausted = True
                    frame, taken = None, 0
                    break

                frame = chunk
                logger.debug(f"Most recent chunk size: {len(chunk)}")
                logger.debug(
                    f"Will produce {np.floor(len(chunk) / self.batch_size)} batches of size {self.batch_size}"
                )
                taken, n_rows = 0, len(frame)
                while taken < n_rows:
                    need = self.batch_size - len(batch)
                    window = frame.iloc[taken : taken + need]
                    rows = window.to_dict(orient="records")
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
            # Rows pulled from the catalog but never yielded - the undrained tail of the
            # current chunk, a partial batch, and anything the prefetcher ran ahead and
            # fetched - are pushed back so a later pass resumes exactly where this one
            # stopped, in order and without duplicates.
            leftover = frame.iloc[taken:].to_dict(orient="records") if frame is not None else []
            if batch or leftover:
                self._buffered = batch + leftover + self._buffered
            if self._exhausted or self._stop.is_set():
                self._iterator = None
