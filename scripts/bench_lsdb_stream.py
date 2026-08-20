"""Measure where ``train_stream`` spends its time on an LSDB-backed catalog.

Splits each batch into the two halves that matter and reports the totals:

    data wait   - blocked in the DataLoader, i.e. fetching + decoding + structuring + collating
    train step  - the model's forward/backward/optimizer step

and then breaks the data wait down using the dataset's own counters (chunk fetch vs. row
decode). If data wait is small, the stream is not the bottleneck and prefetching will not
help. If data wait is large but *chunk fetch* is not, the cost is in the Python decode path,
not in lsdb.

Usage
-----
    python bench_stream.py                                  # baseline: no client, no prefetch
    python bench_stream.py --client threads --prefetch 2    # both mechanisms on
    python bench_stream.py --sweep                          # every combination, one table

Run it from anywhere; it only needs `hyrax` importable and network access to
data.lsdb.io.
"""

import argparse
import itertools
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

CATALOG_URL = "https://data.lsdb.io/hats/tess/tess_lightcurve"


def build_catalog(min_observations: int):
    """Open the TESS light-curve catalog and filter it the way the demo notebook does."""
    import lsdb
    from nested_pandas.utils import count_nested

    catalog = lsdb.open_catalog(CATALOG_URL)

    def drop_nans(frame):
        frame["lightcurve.sap_flux"] = frame["lightcurve.sap_flux"].astype(float)
        return frame.dropna(subset=["lightcurve.sap_flux"]).dropna(subset=["lightcurve"])

    catalog = catalog.map_partitions(drop_nans)
    catalog = catalog.map_partitions(lambda pts: count_nested(pts, "lightcurve"))
    return catalog.query(f"n_lightcurve >= {min_observations}")


def configure(hyrax_instance, data_location, args, partitions_per_chunk, prefetch_chunks, use_client):
    """Apply the demo notebook's TESS configuration plus the knobs under test."""
    hyrax_instance.set_config(
        "data_request",
        {
            "train_stream": {
                "data": {
                    "dataset_class": "LightCurveLSDBStreamDataset",
                    "data_location": data_location,
                    "primary_id_field": "ticid",
                    "fields": [
                        "lightcurve_time",
                        "lightcurve_sap_flux",
                        "lightcurve_sap_flux_err",
                    ],
                }
            }
        },
    )
    hyrax_instance.set_config("model.name", "HyraxTs2Vec")

    lightcurve = "data_set.LightCurveLSDBStreamDataset"
    hyrax_instance.set_config(f"{lightcurve}.band_field", False)
    hyrax_instance.set_config(f"{lightcurve}.time_field", "lightcurve_time")
    hyrax_instance.set_config(f"{lightcurve}.flux_field", "lightcurve_sap_flux")
    hyrax_instance.set_config(f"{lightcurve}.flux_err_field", "lightcurve_sap_flux_err")
    hyrax_instance.set_config(f"{lightcurve}.max_sequence_length", args.max_sequence_length)

    stream = "data_set.LSDBStreamDataset"
    hyrax_instance.set_config(f"{stream}.stream_type", "infinite")
    # Pinned, so every configuration draws the same partitions in the same order. Partition
    # sizes vary by more than the effect being measured, so without this the comparison is
    # noise.
    hyrax_instance.set_config(f"{stream}.seed", args.seed)
    hyrax_instance.set_config(f"{stream}.partitions_per_chunk", partitions_per_chunk)
    hyrax_instance.set_config(f"{stream}.prefetch_chunks", prefetch_chunks)
    hyrax_instance.set_config(f"{stream}.use_dask_client", use_client)

    hyrax_instance.set_config("data_loader.batch_size", args.batch_size)


@dataclass
class Result:
    """One measured configuration."""

    label: str
    batches: int
    wait_seconds: float
    train_seconds: float
    fetch_seconds: float
    convert_seconds: float
    chunks: int
    rows: int

    @property
    def wall(self) -> float:
        """Total measured wall-clock time."""
        return self.wait_seconds + self.train_seconds

    def report(self) -> str:
        """Render this result as an indented block."""
        wall = self.wall or float("inf")
        return "\n".join(
            [
                f"{self.label}",
                f"    batches        {self.batches}",
                f"    wall           {self.wall:8.2f} s",
                f"    data wait      {self.wait_seconds:8.2f} s  ({self.wait_seconds / wall:6.1%})",
                f"    train step     {self.train_seconds:8.2f} s  ({self.train_seconds / wall:6.1%})",
                f"    chunk fetch    {self.fetch_seconds:8.2f} s  over {self.chunks} chunk(s)",
                f"    row decode     {self.convert_seconds:8.2f} s  for {self.rows} row(s)",
            ]
        )


def measure(catalog, args, partitions_per_chunk, prefetch_chunks, client_mode) -> Result:
    """Run one configuration and return its timings."""
    import ignite.distributed as idist

    import hyrax
    from hyrax.datasets import LSDBStreamDataset

    LSDBStreamDataset.clear_catalogs()
    data_location = LSDBStreamDataset.register_catalog("bench_catalog", catalog)

    hyrax_instance = hyrax.Hyrax()
    configure(
        hyrax_instance,
        data_location,
        args,
        partitions_per_chunk,
        prefetch_chunks,
        use_client=client_mode != "none",
    )

    if client_mode == "none":
        client_context = nullcontext()
    else:
        from dask.distributed import Client

        client_context = Client(
            n_workers=args.workers,
            threads_per_worker=1,
            processes=client_mode == "processes",
        )

    wait_seconds = 0.0
    train_seconds = 0.0
    batches = 0

    with client_context:
        with hyrax_instance.train_stream() as session:
            stream = session._provider._stream

            loader = iter(session.data_loader)

            # Drain the chunk that setup_model's peek already pulled, and let the prefetch
            # thread reach steady state. Without this the first batches come free out of the
            # peek buffer and the run looks as though it never fetched anything.
            for _ in range(args.warmup):
                try:
                    session.process(next(loader))
                except StopIteration:
                    break

            stream._timings.reset()

            for _ in range(args.batches):
                start = perf_counter()
                try:
                    batch = next(loader)
                except StopIteration:
                    break
                waited = perf_counter()
                session.process(batch)

                wait_seconds += waited - start
                train_seconds += perf_counter() - waited
                batches += 1

            timings = stream._timings
            result = Result(
                label=(
                    f"client={client_mode}"
                    + (f"({args.workers})" if client_mode != "none" else "")
                    + f"  partitions_per_chunk={partitions_per_chunk}"
                    f"  prefetch_chunks={prefetch_chunks}"
                    f"  batch_size={args.batch_size}"
                    f"  device={idist.device()}"
                ),
                batches=batches,
                wait_seconds=wait_seconds,
                train_seconds=train_seconds,
                fetch_seconds=timings.fetch_seconds,
                convert_seconds=timings.convert_seconds,
                chunks=timings.chunks,
                rows=timings.rows,
            )

    return result


def main(argv=None) -> int:
    """Parse arguments, run the requested configurations, and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=50, help="batches to time per configuration")
    parser.add_argument(
        "--warmup", type=int, default=10, help="untimed batches, to drain the model pre-flight peek"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--partitions-per-chunk", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=0, help="prefetch_chunks")
    parser.add_argument("--client", choices=["none", "threads", "processes"], default="none")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17, help="pins the partition draw across configs")
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="run every combination below")
    parser.add_argument("--debug", action="store_true", help="turn on the stream's DEBUG logging")
    args = parser.parse_args(argv)

    if args.debug:
        logging.getLogger("hyrax.datasets.lsdb_stream_dataset").setLevel(logging.DEBUG)
        logging.getLogger("hyrax.datasets.streaming_data_provider").setLevel(logging.DEBUG)

    print(f"opening {CATALOG_URL} ...", flush=True)
    catalog = build_catalog(args.min_observations)

    if args.sweep:
        combinations = list(
            itertools.product(["none", "threads", "processes"], [args.partitions_per_chunk], [0, 2])
        )
    else:
        combinations = [(args.client, args.partitions_per_chunk, args.prefetch)]

    results = []
    for client_mode, partitions_per_chunk, prefetch_chunks in combinations:
        print(f"\nrunning: client={client_mode} chunks={partitions_per_chunk} prefetch={prefetch_chunks}")
        results.append(measure(catalog, args, partitions_per_chunk, prefetch_chunks, client_mode))

    print("\n" + "=" * 78)
    for result in results:
        print(result.report())
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
