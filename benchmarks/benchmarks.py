"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import subprocess

import hyrax


def time_import():
    """
    time how long it takes to import our package. This should stay relatively fast.

    Note, the actual import time will be slightly lower than this on a comparable system
    However, high import times do affect this metric proportionally.
    """
    result = subprocess.run(["python", "-c", "import hyrax"])
    assert result.returncode == 0


def time_help():
    """
    time how long it takes to run --help from the CLI
    """
    result = subprocess.run(["hyrax", "--help"])
    assert result.returncode == 0


def time_infer_help():
    """
    time how long it takes to do verb-specific help for infer
    """
    result = subprocess.run(["hyrax", "infer", "--help"])
    assert result.returncode == 0


def time_train_help():
    """
    time how long it takes to do verb-specific help for train
    """
    result = subprocess.run(["hyrax", "train", "--help"])
    assert result.returncode == 0


def time_lookup_help():
    """
    time how long it takes to do verb-specific help for lookup
    """
    result = subprocess.run(["hyrax", "lookup", "--help"])
    assert result.returncode == 0


def time_umap_help():
    """
    time how long it takes to do verb-specific help for umap
    """
    result = subprocess.run(["hyrax", "umap", "--help"])
    assert result.returncode == 0


def time_save_to_database_help():
    """
    time how long it takes to do verb-specific help for save_to_database
    """
    result = subprocess.run(["hyrax", "save_to_database", "--help"])
    assert result.returncode == 0


def time_database_connection_help():
    """
    time how long it takes to do verb-specific help for database_connection
    """
    result = subprocess.run(["hyrax", "database_connection", "--help"])
    assert result.returncode == 0


def time_download_help():
    """
    time how long it takes to do verb-specific help for download
    """
    result = subprocess.run(["hyrax", "download", "--help"])
    assert result.returncode == 0


def time_prepare_help():
    """
    time how long it takes to do verb-specific help for prepare
    """
    result = subprocess.run(["hyrax", "prepare", "--help"])
    assert result.returncode == 0


def time_rebuild_manifest_help():
    """
    time how long it takes to do verb-specific help for rebuild_manifest
    """
    result = subprocess.run(["hyrax", "rebuild_manifest", "--help"])
    assert result.returncode == 0


def time_visualize_help():
    """
    time how long it takes to do verb-specific help for visualize
    """
    result = subprocess.run(["hyrax", "visualize", "--help"])
    assert result.returncode == 0


def time_nb_obj_construct():
    """
    time how long notebook users must wait for our interface object to construct
    """
    hyrax.Hyrax()


def time_nb_obj_dir():
    """
    Time how long it takes to construct the interface object and load the
    dynamcally generated list of verbs using `dir()`
    """
    h = hyrax.Hyrax()
    dir(h)
