# Self-hosted runtime benchmarks

End-to-end wall-clock comparisons of Hyrax against an equivalent plain-PyTorch script. Each
benchmark trains a model and then runs inference, repeats that several times for both frameworks,
and writes a JSON summary (means, standard deviations, and the Hyrax/PyTorch slowdown ratios).

These are meant to run on a dedicated machine (see `.github/workflows/pipeline_time_bench.yml`),
but they can be run by hand from the repository root.

## Benchmarks

| Name    | Hyrax side                                   | PyTorch side                                              | Quality metric                  |
| ------- | -------------------------------------------- | --------------------------------------------------------- | ------------------------------- |
| `cifar` | `HyraxCNN` + `HyraxCifarDataset`             | small CNN + `torchvision.datasets.CIFAR10`                | test accuracy                   |
| `hsc`   | `HyraxAutoencoderV2` + `HSCDataset` on HSC1k | identical autoencoder + a plain `Dataset` reading FITS    | final epoch-averaged train MSE  |

`timebench_utils.py` holds the CIFAR benchmark, `timebench_hsc_utils.py` the HSC one.

The HSC benchmark uses the ~1000 HSC cutouts (g/r/i, 8 arcsec) published on Zenodo
(`doi:10.5281/zenodo.14498536`). The PyTorch baseline is idiomatic PyTorch: it reads the FITS
files on every `__getitem__` with no caching, while Hyrax keeps its default
`data_set.use_cache = true`. The `use_cache` setting of each side is recorded in the result JSON.
Training uses the full dataset and inference encodes every object.

## Running

```bash
python benchmarks_self_hosted/run_timebench.py --benchmark cifar # CIFAR only (default)
python benchmarks_self_hosted/run_timebench.py --benchmark hsc   # HSC only
python benchmarks_self_hosted/run_timebench.py --benchmark all   # both
```

Run configurations (`epochs`, `batch_size`, `repeats`, ...) are the `CIFAR_CONFIG` and
`HSC_CONFIG` dicts at the top of `run_timebench.py`.

## Environment variables

| Variable      | Default                          | Meaning                                                            |
| ------------- | -------------------------------- | ------------------------------------------------------------------ |
| `CIFAR_DIR`   | `./data`                         | Where CIFAR-10 is (or will be downloaded to).                      |
| `HSC_DIR`     | `./data`                         | Where the HSC1k archive is extracted to (`<HSC_DIR>/hsc_8asec_1000`). |
| `RESULT_PATH` | `benchmarks_self_hosted/results` | Directory for the result JSON files.                               |

Data is downloaded automatically the first time and is not part of the timed region.

## Output

One file per benchmark, named `<short_commit>_<benchmark>_<YYYYMMDD_HHMMSS>.json`. The script
prints a `RESULT_FILE=<absolute path>` line for each file it writes; the workflow uses those lines
to collect the results.
