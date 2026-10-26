# ruff: noqa: D102
"""Tests for DataLoader worker processes, in particular persistent workers.

These describe the intended behavior of persistent DataLoader workers (see the
"persistent_workers" design sketch):

1. ``dist_data_loader`` takes a ``persistent_workers`` argument; only the training
   verb's validation loader opts in.
2. ``DataProvider.on_epoch_start`` (augmentation RNG reset + dataset hooks) reaches
   persistent workers, so they behave as if freshly forked each epoch.
3. Workers disarm DataLoader iterators they inherit by fork (``_hyrax_worker_init``),
   so they can never shut down another loader's workers.
4. Per-worker data caches survive across epochs.
5. No worker processes are left behind once a verb returns.
"""

import gc
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import psutil
import pytest

import hyrax
from hyrax.datasets.data_provider import DataProvider
from hyrax.datasets.dataset_registry import HyraxDataset
from hyrax.pytorch_ignite import dist_data_loader

NUM_WORKERS = 2
NUM_EPOCHS = 3

# ---------------------------------------------------------------------------
# Test dataset (auto-registered in DATASET_REGISTRY on import)
# ---------------------------------------------------------------------------


class WorkerProbeDataset(HyraxDataset):
    """Reports, for every item, which process produced it and what that process
    believes about the current epoch, augmentation seed, and getter call count."""

    def __init__(self, config, data_location=None):
        self.epoch = 0
        self.fetch_counts: dict[int, int] = {}
        super().__init__(config)

    def __len__(self):
        return 16

    def on_epoch_start(self, verb):
        self.epoch += 1

    def get_object_id(self, idx):
        return f"{idx:02d}"

    def get_epoch(self, idx):
        return np.int64(self.epoch)

    def get_pid(self, idx):
        return np.int64(os.getpid())

    def get_fetch_count(self, idx):
        # Counts calls in *this* process. A cached item returns its first value (1).
        self.fetch_counts[idx] = self.fetch_counts.get(idx, 0) + 1
        return np.int64(self.fetch_counts[idx])

    def get_seed(self, idx):
        return np.int64(0)

    def augment_seed(self, value, idx, rng_seed):
        # Surface the augmentation seed this process used for this item.
        return np.int64(rng_seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(fields, *, use_cache=False, augment=False, seed=False):
    h = hyrax.Hyrax()
    h.config["data_set"]["use_cache"] = use_cache
    h.config["data_set"]["seed"] = seed
    h.config["data_loader"]["batch_size"] = 4
    h.config["data_loader"]["num_workers"] = NUM_WORKERS

    definition = {
        "dataset_class": "WorkerProbeDataset",
        "primary_id_field": "object_id",
        "fields": list(fields),
    }
    if augment:
        definition["augment"] = ["seed"]

    return h.config, DataProvider(h.config, {"data": definition})


def _run_epochs(provider, loader, n_epochs=NUM_EPOCHS, verb="train"):
    """Drive the loader the way Train does: on_epoch_start, then a full pass.

    Returns one dict per epoch mapping field name -> array ordered by object_id.
    """
    epochs = []
    for _ in range(n_epochs):
        provider.on_epoch_start(verb)
        batches = list(loader)
        ids = np.concatenate([b["object_id"] for b in batches])
        order = np.argsort(ids)
        fields = {
            name: np.concatenate([b["data"][name] for b in batches])[order] for name in batches[0]["data"]
        }
        fields["object_id"] = ids[order]
        epochs.append(fields)
    return epochs


def _shutdown(loader):
    """Stop a persistent loader's workers so they don't outlive the test."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def _is_running_worker(process, parent_cmdline):
    """True for a live process forked from this one, e.g. a DataLoader worker.

    Forked children keep the parent's command line. This excludes helpers that are
    started with exec, such as multiprocessing's long-lived resource tracker.
    """
    try:
        return process.status() != psutil.STATUS_ZOMBIE and process.cmdline() == parent_cmdline
    except psutil.NoSuchProcess:
        return False


def _new_workers(before, timeout=1.0):
    """Worker processes forked since ``before`` that are still running after ``timeout``."""
    me = psutil.Process()
    parent_cmdline = me.cmdline()
    deadline = time.monotonic() + timeout
    while True:
        children = [p for p in me.children(recursive=True) if p.pid not in before]
        alive = [p for p in children if _is_running_worker(p, parent_cmdline)]
        if not alive or time.monotonic() > deadline:
            return alive
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# 2. Epoch propagation into persistent workers
# ---------------------------------------------------------------------------


def test_on_epoch_start_reaches_persistent_workers():
    """Every item in epoch N should come from a worker whose dataset saw N
    on_epoch_start calls, and the same worker processes should serve every epoch."""
    config, provider = _make_provider(["epoch", "pid"])
    loader = dist_data_loader(provider, config, persistent_workers=True)
    try:
        epochs = _run_epochs(provider, loader)
    finally:
        _shutdown(loader)

    for n, epoch in enumerate(epochs, start=1):
        assert set(epoch["epoch"].tolist()) == {n}, f"epoch {n}: workers reported {set(epoch['epoch'])}"
        assert os.getpid() not in set(epoch["pid"].tolist()), "items should be produced by worker processes"

    worker_pids = [frozenset(epoch["pid"].tolist()) for epoch in epochs]
    assert len(set(worker_pids)) == 1, f"workers were re-forked between epochs: {worker_pids}"


def test_worker_augmentation_seeds_match_non_persistent_workers():
    """With a fixed seed, persistent workers should draw exactly the augmentation
    seeds that freshly forked (non-persistent) workers draw, epoch for epoch."""
    seeds = {}
    for persistent in (False, True):
        config, provider = _make_provider(["seed"], augment=True, seed=1234)
        loader = dist_data_loader(provider, config, persistent_workers=persistent)
        try:
            seeds[persistent] = [epoch["seed"] for epoch in _run_epochs(provider, loader)]
        finally:
            _shutdown(loader)

    for n, (fresh, persisted) in enumerate(zip(seeds[False], seeds[True]), start=1):
        np.testing.assert_array_equal(persisted, fresh, err_msg=f"augmentation seeds differ in epoch {n}")


# ---------------------------------------------------------------------------
# 3. Inherited DataLoader iterators
# ---------------------------------------------------------------------------


def test_worker_init_disarms_inherited_dataloader_iterators():
    """A forked child holds a copy of every live DataLoader iterator in the parent.
    If one of those copies shuts down in the child (e.g. when garbage collected), torch
    signals the *parent's* workers to exit before failing its own assertion.
    ``_hyrax_worker_init`` must make inherited iterators inert."""
    from hyrax.pytorch_ignite import _hyrax_worker_init

    config, provider = _make_provider(["epoch"])
    loader = dist_data_loader(provider, config, persistent_workers=True)
    try:
        _run_epochs(provider, loader, n_epochs=1)  # starts the persistent workers

        def child(queue):
            try:
                _hyrax_worker_init(0)
                loader._iterator._shutdown_workers()  # what __del__ does in the child
                queue.put(None)
            except BaseException as err:  # noqa: BLE001
                queue.put(repr(err))

        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        process = ctx.Process(target=child, args=(queue,))
        process.start()
        error = queue.get(timeout=60)
        process.join(timeout=60)
        assert error is None, f"inherited iterator shutdown raised in child: {error}"

        # The parent's workers must still be alive and serving data.
        epoch = _run_epochs(provider, loader, n_epochs=1)[0]
        assert len(epoch["object_id"]) == len(provider)
    finally:
        _shutdown(loader)


def test_worker_loaders_disarm_inherited_iterators():
    """Every multi-worker loader must install _hyrax_worker_init, persistent or not:
    re-forked non-persistent workers inherit the parent's live persistent iterators too."""
    from hyrax.pytorch_ignite import _hyrax_worker_init

    config, provider = _make_provider(["epoch"])
    for persistent in (False, True):
        loader = dist_data_loader(provider, config, persistent_workers=persistent)
        assert loader.worker_init_fn is _hyrax_worker_init, f"persistent_workers={persistent}"


def test_live_iterators_are_registered_for_disarming():
    """_hyrax_worker_init only disarms iterators in _live_worker_iterators, so every
    multi-worker iterator must be registered, persistent or not: validation workers are
    forked while the non-persistent train iterator is still alive."""
    from hyrax.pytorch_ignite import _live_worker_iterators

    config, provider = _make_provider(["epoch"])
    for persistent in (False, True):
        loader = dist_data_loader(provider, config, persistent_workers=persistent)
        iterator = iter(loader)
        try:
            assert iterator in _live_worker_iterators, f"persistent_workers={persistent}"
        finally:
            iterator._shutdown_workers()
            loader._iterator = None


@pytest.mark.parametrize("num_workers", [NUM_WORKERS], indirect=True)
def test_train_with_workers_raises_no_unraisable_exceptions(loopback_hyrax, tmp_path, monkeypatch):
    """End to end: training with a validation split and workers should not produce
    'Exception ignored in ... _MultiProcessingDataLoaderIter.__del__' errors in any
    process. The hook is inherited by forked workers and logs to a shared file.

    Note: this failure depends on garbage-collection timing in the workers, so it may
    not fail on every run before the fix. The deterministic version is the test above.
    """
    log_path = tmp_path / "unraisable.log"

    def record_unraisable(unraisable):
        with open(log_path, "a") as f:
            f.write(f"pid={os.getpid()} {unraisable.exc_type.__name__}: {unraisable.exc_value}\n")

    monkeypatch.setattr(sys, "unraisablehook", record_unraisable)

    h, _ = loopback_hyrax
    h.config["train"]["epochs"] = 5
    h.train()
    gc.collect()

    logged = log_path.read_text() if log_path.exists() else ""
    assert logged == "", f"unraisable exceptions during training:\n{logged}"


# ---------------------------------------------------------------------------
# 4. Per-worker caches survive across epochs
# ---------------------------------------------------------------------------


def test_validation_cache_is_reused_across_epochs_with_workers():
    """With caching on, a sequential loader should only call dataset getters in the
    first epoch; later epochs are served from each persistent worker's cache."""
    config, provider = _make_provider(["fetch_count"], use_cache=True)
    loader = dist_data_loader(provider, config, persistent_workers=True)
    try:
        epochs = _run_epochs(provider, loader)
    finally:
        _shutdown(loader)

    for n, epoch in enumerate(epochs, start=1):
        counts = set(epoch["fetch_count"].tolist())
        assert counts == {1}, f"epoch {n}: getters were re-called (fetch counts {counts}), cache was lost"


# ---------------------------------------------------------------------------
# 5. Worker cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [NUM_WORKERS], indirect=True)
def test_no_worker_processes_left_after_train(loopback_hyrax):
    """train() must shut down its persistent DataLoader workers before it returns.

    Leaving it to garbage collection stalls: the loaders sit in reference cycles with the
    Ignite engine, and when cyclic GC frees them, mp.Queue's feeder threads are stopped
    before the iterator's __del__ sends the workers their stop signal. The parent then
    waits out torch's 5 s join timeout for every worker. Automatic GC is disabled here so
    that only an explicit shutdown inside the verb can make this pass.
    """
    h, _ = loopback_hyrax
    h.config["train"]["epochs"] = 2
    before = {p.pid for p in psutil.Process().children(recursive=True)}

    gc_was_enabled = gc.isenabled()
    gc.disable()
    leftover = []
    try:
        h.train()
        leftover = _new_workers(before)
    finally:
        # Don't let a failure here stall later tests: dead workers make the eventual
        # iterator cleanup return immediately instead of waiting out join timeouts.
        for process in leftover:
            process.kill()
        if gc_was_enabled:
            gc.enable()

    assert leftover == [], (
        f"worker processes still running 1 s after train() returned: {[p.pid for p in leftover]}"
    )


# ---------------------------------------------------------------------------
# 1. Which loaders use persistent workers
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded_loaders(monkeypatch):
    """Record every DataLoader created through dist_data_loader, in creation order."""
    import hyrax.pytorch_ignite
    import hyrax.verbs.train

    loaders = []
    original = hyrax.pytorch_ignite.dist_data_loader

    def recording_dist_data_loader(*args, **kwargs):
        loader = original(*args, **kwargs)
        loaders.append(loader)
        return loader

    # Train imports dist_data_loader at module level; infer and test import it at call time.
    monkeypatch.setattr(hyrax.pytorch_ignite, "dist_data_loader", recording_dist_data_loader)
    monkeypatch.setattr(hyrax.verbs.train, "dist_data_loader", recording_dist_data_loader)
    return loaders


@pytest.mark.parametrize("num_workers", [NUM_WORKERS], indirect=True)
def test_train_persists_only_the_validation_loader(loopback_hyrax, recorded_loaders):
    """The validation loader is sequential, so persistent workers give every worker a
    stable shard to cache. The shuffled train loader stays non-persistent so each worker
    does not grow a cache of the whole train set."""
    h, _ = loopback_hyrax
    h.train()

    # Train creates its loaders in split order: train, then validate.
    train_loader, validate_loader = recorded_loaders
    assert not train_loader.persistent_workers
    assert validate_loader.persistent_workers


@pytest.mark.parametrize("num_workers", [NUM_WORKERS], indirect=True)
def test_infer_does_not_persist_workers(loopback_hyrax, recorded_loaders):
    """Inference is a single pass, so persistent workers would only outlive the verb."""
    h, _ = loopback_hyrax
    h.infer()

    assert recorded_loaders, "infer did not create a data loader"
    assert not any(loader.persistent_workers for loader in recorded_loaders)


@pytest.mark.parametrize("num_workers", [NUM_WORKERS], indirect=True)
def test_test_verb_does_not_persist_workers(loopback_hyrax, recorded_loaders):
    """Testing is a single pass, so persistent workers would only outlive the verb."""
    h, _ = loopback_hyrax
    h.train()
    recorded_loaders.clear()

    h.test()

    assert recorded_loaders, "test did not create a data loader"
    assert not any(loader.persistent_workers for loader in recorded_loaders)
