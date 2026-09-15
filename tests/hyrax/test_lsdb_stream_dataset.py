"""Tests for LSDBStreamDataset's fixed-size batching, catalog registry, and streaming.

Most batching tests substitute a list of plain DataFrames for the lsdb stream by patching
``_make_stream``, so the buffering logic is exercised without touching dask. The tests that
do build real ``CatalogStream`` / ``InfiniteStream`` objects run against a tiny in-memory
catalog created with ``lsdb.from_dataframe``.
"""

import lsdb
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

import hyrax
from hyrax.config_schemas.data_request import DataRequestConfig
from hyrax.datasets.lsdb_stream_dataset import LSDBStreamDataset
from hyrax.datasets.streaming_data_provider import StreamingDataProvider
from hyrax.pytorch_ignite import dist_data_loader

# lsdb.from_dataframe on these 12 rows produces 5 partitions of sizes [2, 3, 2, 3, 2],
# which is deliberately ragged so batching across chunk boundaries is exercised.
CATALOG_ROWS = 12


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep registered catalogs from leaking between tests."""
    yield
    LSDBStreamDataset.clear_catalogs()


@pytest.fixture
def source_frame():
    """A small flat table with a non-identifier column name and a fixed-width vector."""
    return pd.DataFrame(
        {
            "object_id": [f"id{i:02d}" for i in range(CATALOG_ROWS)],
            "coord_ra": np.linspace(0.0, 350.0, CATALOG_ROWS),
            "coord_dec": np.linspace(-80.0, 80.0, CATALOG_ROWS),
            "magr": np.arange(CATALOG_ROWS, dtype=float),
            "mag-r": np.arange(CATALOG_ROWS, dtype=float),
        }
    )


@pytest.fixture
def tiny_catalog(source_frame):
    """An in-memory lsdb catalog over ``source_frame``."""
    return lsdb.from_dataframe(source_frame, ra_column="coord_ra", dec_column="coord_dec")


def _chunk(start, count, value_offset=0.0):
    """Build a stand-in stream chunk of ``count`` rows starting at ``start``."""
    return pd.DataFrame(
        {
            "object_id": [f"id{i:02d}" for i in range(start, start + count)],
            "magr": [float(i) + value_offset for i in range(start, start + count)],
        }
    )


def _build_dataset(
    catalog,
    *,
    batch_size=5,
    stream_type="catalog",
    fields=("magr",),
    primary_id="object_id",
    shuffle=False,
    seed=0,
    partitions_per_chunk=1,
    name="tiny",
    data_location=None,
    open_catalog_kwargs=None,
    use_dask_client=False,
):
    """Register ``catalog`` (when given) and construct a dataset against it.

    Prefetching and the dask client are off unless a test asks for them, so a test that says
    nothing about either exercises the plain synchronous path.
    """
    location = data_location
    if location is None:
        location = LSDBStreamDataset.register_catalog(name, catalog)

    h = hyrax.Hyrax()
    h.config["data_loader"]["batch_size"] = batch_size

    ds_config = h.config["data_set"]["LSDBStreamDataset"]
    ds_config["stream_type"] = stream_type
    ds_config["shuffle"] = shuffle
    ds_config["seed"] = seed
    ds_config["partitions_per_chunk"] = partitions_per_chunk
    ds_config["use_dask_client"] = use_dask_client
    if open_catalog_kwargs is not None:
        ds_config["open_catalog_kwargs"] = open_catalog_kwargs

    request = {"dataset_class": "LSDBStreamDataset", "data_location": location}
    if primary_id is not None:
        request["primary_id_field"] = primary_id
    if fields is not None:
        request["fields"] = list(fields)
    h.config["data_request"] = {"train_stream": {"data": request}}

    return LSDBStreamDataset(h.config, data_location=location)


def _with_chunks(monkeypatch, dataset, chunks):
    """Replace the dataset's lsdb stream with a fixed list of DataFrame chunks."""
    monkeypatch.setattr(dataset, "_make_stream", lambda: list(chunks))
    return dataset


def _ids(batches):
    """Flatten yielded batches into a list of object ids."""
    return [row["object_id"] for batch in batches for row in batch]


#
# Construction and configuration
#


def test_requires_data_location(tiny_catalog):
    """A missing data_location names both accepted forms."""
    LSDBStreamDataset.register_catalog("tiny", tiny_catalog)
    h = hyrax.Hyrax()
    for missing in (None, False):
        with pytest.raises(ValueError, match="lsdb://"):
            LSDBStreamDataset(h.config, data_location=missing)


def test_invalid_stream_type_raises(tiny_catalog):
    """stream_type is restricted to the two lsdb stream classes."""
    with pytest.raises(ValueError, match="stream_type"):
        _build_dataset(tiny_catalog, stream_type="firehose")


@pytest.mark.parametrize("bad_value", [False, 0, -1])
def test_partitions_per_chunk_must_be_positive(tiny_catalog, bad_value):
    """A zero/false partitions_per_chunk would make lsdb stream the whole catalog at once."""
    with pytest.raises(ValueError, match="partitions_per_chunk"):
        _build_dataset(tiny_catalog, partitions_per_chunk=bad_value)


def test_seed_false_is_none_but_zero_survives(tiny_catalog):
    """`false` is the not-set sentinel; 0 is a legitimate seed."""
    assert _build_dataset(tiny_catalog, seed=False).seed is None
    assert _build_dataset(tiny_catalog, seed=0, name="tiny0").seed == 0


#
# Catalog registry
#


def test_register_catalog_returns_uri(tiny_catalog):
    """register_catalog hands back the data_location string to put in the config."""
    assert LSDBStreamDataset.register_catalog("gaia_mmu", tiny_catalog) == "lsdb://gaia_mmu"
    assert LSDBStreamDataset.registered_catalogs() == ["gaia_mmu"]


def test_lsdb_uri_resolves_to_registered_catalog(tiny_catalog):
    """An lsdb:// data_location streams the exact object that was registered."""
    dataset = _build_dataset(tiny_catalog, name="gaia_mmu")
    assert dataset._catalog is tiny_catalog


def test_unknown_lsdb_uri_raises_listing_known_names(tiny_catalog):
    """The lookup error names what is registered and how to fix it."""
    LSDBStreamDataset.register_catalog("known", tiny_catalog)
    h = hyrax.Hyrax()
    with pytest.raises(KeyError) as excinfo:
        LSDBStreamDataset(h.config, data_location="lsdb://missing")
    message = str(excinfo.value)
    assert "known" in message
    assert "register_catalog" in message


def test_register_catalog_rejects_non_catalog(source_frame):
    """A bare DataFrame fails at registration rather than deep inside lsdb."""
    with pytest.raises(TypeError, match="lsdb.Catalog"):
        LSDBStreamDataset.register_catalog("frame", source_frame)


@pytest.mark.parametrize("bad_name", ["", "  ", "has space", "has/slash"])
def test_register_catalog_rejects_unusable_names(tiny_catalog, bad_name):
    """Names must round-trip through an lsdb://<name> data_location."""
    with pytest.raises(ValueError):
        LSDBStreamDataset.register_catalog(bad_name, tiny_catalog)


def test_unregister_and_clear(tiny_catalog):
    """Catalogs can be removed individually or all at once."""
    LSDBStreamDataset.register_catalog("a", tiny_catalog)
    LSDBStreamDataset.register_catalog("b", tiny_catalog)

    LSDBStreamDataset.unregister_catalog("a")
    LSDBStreamDataset.unregister_catalog("not-registered")  # no error
    assert LSDBStreamDataset.registered_catalogs() == ["b"]

    LSDBStreamDataset.clear_catalogs()
    assert LSDBStreamDataset.registered_catalogs() == []


def test_lsdb_uri_survives_data_request_validation():
    """The data_request schema must not rewrite an lsdb:// handle into a path."""
    validated = DataRequestConfig.model_validate(
        {
            "dataset_class": "LSDBStreamDataset",
            "data_location": "lsdb://gaia_mmu",
            "primary_id_field": "object_id",
        }
    )
    assert validated.data_location == "lsdb://gaia_mmu"


#
# Batching and buffering
#


def test_exact_batch_size_across_ragged_chunks(monkeypatch, tiny_catalog):
    """Ragged chunks are buffered into full batches, with only the last one short."""
    dataset = _build_dataset(tiny_catalog, batch_size=5)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 3), _chunk(3, 7), _chunk(10, 2)])

    batches = list(dataset)

    assert [len(batch) for batch in batches] == [5, 5, 2]
    assert _ids(batches) == [f"id{i:02d}" for i in range(12)]


def test_single_large_chunk_split_across_batches(
    monkeypatch,
    tiny_catalog,
):
    """One chunk bigger than batch_size is split into several batches."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 11)])

    batches = list(dataset)

    assert [len(batch) for batch in batches] == [4, 4, 3]


def test_no_short_batch_when_total_is_a_multiple(monkeypatch, tiny_catalog):
    """An exact multiple of batch_size yields no trailing short batch."""
    dataset = _build_dataset(tiny_catalog, batch_size=5)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 4), _chunk(4, 6)])

    batches = list(dataset)

    assert [len(batch) for batch in batches] == [5, 5]


def test_only_final_batch_is_short(monkeypatch, tiny_catalog):
    """Whatever the chunk sizes, every batch but the last is full."""
    dataset = _build_dataset(tiny_catalog, batch_size=3)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 1), _chunk(1, 8), _chunk(9, 2), _chunk(11, 5)])

    batches = list(dataset)

    assert all(len(batch) == 3 for batch in batches[:-1])
    assert 0 < len(batches[-1]) <= 3
    assert sum(len(batch) for batch in batches) == 16


def test_empty_chunk_is_tolerated(monkeypatch, tiny_catalog):
    """A partition that computes to zero rows does not break batching."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 3), _chunk(0, 0), _chunk(3, 3)])

    batches = list(dataset)

    assert [len(batch) for batch in batches] == [4, 2]


def test_peek_sample_does_not_lose_a_row(monkeypatch, tiny_catalog):
    """The peeked row is replayed at the head of the first batch."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 3), _chunk(3, 3)])

    sample = dataset.peek_sample()
    assert sample["object_id"] == "id00"

    batches = list(dataset)
    assert _ids(batches) == [f"id{i:02d}" for i in range(6)]


def test_peek_sample_twice_returns_distinct_rows_and_both_replay(monkeypatch, tiny_catalog):
    """Repeated peeks advance a cursor rather than re-reading the same row."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 3), _chunk(3, 3)])

    assert dataset.peek_sample()["object_id"] == "id00"
    assert dataset.peek_sample()["object_id"] == "id01"

    assert _ids(list(dataset)) == [f"id{i:02d}" for i in range(6)]


def test_peek_sample_on_empty_stream_raises(monkeypatch, tiny_catalog):
    """A catalog with no rows fails with an actionable error, not StopIteration."""
    dataset = _build_dataset(tiny_catalog)
    _with_chunks(monkeypatch, dataset, [])

    with pytest.raises(RuntimeError, match="produced no rows"):
        dataset.peek_sample()


def test_stop_ends_iteration_and_flushes_partial(monkeypatch, tiny_catalog):
    """stop() lets the current chunk finish, then flushes what is buffered."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 6), _chunk(6, 6), _chunk(12, 6)])

    batches = []
    for batch in dataset:
        batches.append(batch)
        dataset.stop()

    # First chunk yields one full batch, then stop() prevents fetching a second chunk
    # and the two remaining buffered rows flush as the short final batch.
    assert [len(batch) for batch in batches] == [4, 2]


def test_stop_before_iteration_yields_nothing(monkeypatch, tiny_catalog):
    """A dataset stopped before iteration produces no batches."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 6)])
    dataset.stop()

    assert list(dataset) == []


def test_break_then_resume_loses_no_rows(monkeypatch, tiny_catalog):
    """Breaking out mid-stream re-buffers undelivered rows for the next pass."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 6), _chunk(6, 6)])

    first = []
    for batch in dataset:
        first.append(batch)
        break

    rest = list(dataset)

    seen = _ids(first) + _ids(rest)
    assert seen == [f"id{i:02d}" for i in range(12)]
    assert len(seen) == len(set(seen))


def test_second_pass_after_exhaustion_restarts(monkeypatch, tiny_catalog):
    """Once a finite stream is exhausted, re-iterating starts a fresh pass."""
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(0, 4)])

    assert _ids(list(dataset)) == [f"id{i:02d}" for i in range(4)]
    assert _ids(list(dataset)) == [f"id{i:02d}" for i in range(4)]


#
# Prefetching
#


def test_prefetched_but_unconsumed_rows_are_pushed_back(monkeypatch, tiny_catalog):
    """Chunks the producer ran ahead and fetched survive the consumer walking away.

    A finite ``catalog`` stream is used for inference, where every object must be visited
    exactly once - dropping a speculatively fetched chunk would silently lose objects.
    """
    dataset = _build_dataset(tiny_catalog, batch_size=4)
    _with_chunks(monkeypatch, dataset, [_chunk(i, 4) for i in range(0, 24, 4)])

    first = []
    for batch in dataset:
        first.append(batch)
        break

    rest = list(dataset)

    seen = _ids(first) + _ids(rest)
    assert seen == [f"id{i:02d}" for i in range(24)]
    assert len(seen) == len(set(seen))


def test_producer_exception_reaches_the_consumer(monkeypatch, tiny_catalog):
    """A failure in the background thread is re-raised where the caller can see it."""

    def exploding_stream():
        yield _chunk(0, 4)
        raise RuntimeError("partition read failed")

    dataset = _build_dataset(tiny_catalog, batch_size=4)
    monkeypatch.setattr(dataset, "_make_stream", exploding_stream)

    with pytest.raises(RuntimeError, match="partition read failed"):
        list(dataset)


#
# Dask client wiring
#


def _captured_stream_kwargs(monkeypatch):
    """Patch both lsdb stream classes to record their kwargs instead of building a stream."""
    import lsdb.streams

    captured = {}

    def _record(cls_name):
        def factory(catalog, **kwargs):
            captured.update(kwargs)
            captured["class"] = cls_name
            return []

        return factory

    monkeypatch.setattr(lsdb.streams, "CatalogStream", _record("CatalogStream"))
    monkeypatch.setattr(lsdb.streams, "InfiniteStream", _record("InfiniteStream"))
    return captured


def test_active_dask_client_is_passed_to_lsdb(monkeypatch, tiny_catalog):
    """Without this, lsdb computes each chunk inline and its own prefetch never engages."""
    captured = _captured_stream_kwargs(monkeypatch)
    sentinel_client = object()
    monkeypatch.setattr("dask.distributed.get_client", lambda: sentinel_client)

    dataset = _build_dataset(tiny_catalog, batch_size=4, use_dask_client=True)
    list(dataset)

    assert captured["client"] is sentinel_client
    assert captured["class"] == "CatalogStream"


def test_no_active_dask_client_streams_synchronously(monkeypatch, tiny_catalog):
    """No client is not an error - lsdb just computes inline."""
    captured = _captured_stream_kwargs(monkeypatch)

    def no_client():
        raise ValueError("No clients found")

    monkeypatch.setattr("dask.distributed.get_client", no_client)

    dataset = _build_dataset(tiny_catalog, batch_size=4, use_dask_client=True)
    list(dataset)

    assert captured["client"] is None


def test_use_dask_client_false_never_looks_for_one(monkeypatch, tiny_catalog):
    """Opting out must not touch dask at all."""
    captured = _captured_stream_kwargs(monkeypatch)

    def fail():
        raise AssertionError("get_client() must not be called when use_dask_client is false")

    monkeypatch.setattr("dask.distributed.get_client", fail)

    dataset = _build_dataset(tiny_catalog, batch_size=4, stream_type="infinite", use_dask_client=False)
    list(dataset)

    assert captured["client"] is None
    assert captured["class"] == "InfiniteStream"


#
# Real lsdb streams
#


def test_catalog_stream_over_registered_catalog(tiny_catalog):
    """A finite pass over the real catalog visits every row exactly once."""
    dataset = _build_dataset(tiny_catalog, batch_size=5)

    batches = list(dataset)

    assert [len(batch) for batch in batches] == [5, 5, 2]
    assert sorted(_ids(batches)) == [f"id{i:02d}" for i in range(CATALOG_ROWS)]


def test_infinite_stream_does_not_exhaust(tiny_catalog):
    """An InfiniteStream keeps producing full batches until it is stopped."""
    import itertools

    dataset = _build_dataset(tiny_catalog, batch_size=5, stream_type="infinite")

    batches = list(itertools.islice(dataset, 6))
    dataset.stop()

    # A single finite pass over 12 rows could only produce 3 batches.
    assert len(batches) == 6
    assert all(len(batch) == 5 for batch in batches)


def test_infinite_stream_ignores_shuffle_false(tiny_catalog, caplog):
    """InfiniteStream has no `shuffle` argument, so the setting is dropped with a warning."""
    import itertools

    dataset = _build_dataset(tiny_catalog, batch_size=5, stream_type="infinite", shuffle=False)

    with caplog.at_level("WARNING"):
        batches = list(itertools.islice(dataset, 2))
    dataset.stop()

    assert len(batches) == 2
    assert "shuffle" in caplog.text


def test_healpix_index_promoted_to_column(tiny_catalog):
    """The HATS spatial index is usable as the primary_id_field."""
    from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

    dataset = _build_dataset(tiny_catalog, batch_size=5, primary_id=SPATIAL_INDEX_COLUMN, fields=("magr",))

    batches = list(dataset)

    assert all(SPATIAL_INDEX_COLUMN in row for batch in batches for row in batch)
    assert len({row[SPATIAL_INDEX_COLUMN] for batch in batches for row in batch}) == CATALOG_ROWS


def test_non_identifier_column_name_survives(tiny_catalog):
    """`mag-r` must not be mangled into a positional name during row conversion."""
    dataset = _build_dataset(tiny_catalog, batch_size=5, fields=("magr", "mag-r"))

    first_row = next(iter(dataset))[0]

    assert "mag-r" in first_row
    assert "magr" in first_row


#
# End to end through the provider, loader, and verbs
#


def test_streaming_data_provider_end_to_end(tiny_catalog):
    """The dataset flows through StreamingDataProvider and dist_data_loader."""
    location = LSDBStreamDataset.register_catalog("tiny", tiny_catalog)

    h = hyrax.Hyrax()
    h.config["data_loader"]["batch_size"] = 5
    h.config["data_set"]["LSDBStreamDataset"]["shuffle"] = False
    h.config["data_set"]["LSDBStreamDataset"]["seed"] = 0
    request = {
        "data": {
            "dataset_class": "LSDBStreamDataset",
            "data_location": location,
            "primary_id_field": "object_id",
            "fields": ["magr"],
        }
    }
    h.config["data_request"] = {"infer_stream": request}

    provider = StreamingDataProvider(h.config, request)
    loader = dist_data_loader(provider, h.config)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size is None
    assert loader.collate_fn == provider.collate

    batches = list(loader)
    assert [len(batch["object_id"]) for batch in batches] == [5, 5, 2]
    assert batches[0]["data"]["magr"].shape == (5,)
