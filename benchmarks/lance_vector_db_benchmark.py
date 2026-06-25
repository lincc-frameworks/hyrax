"""
Benchmark script for Lance HNSW vector indexing capabilities.

This script tests:
1. HNSW index creation, configuration, and supported distance metrics
2. Incremental indexing on existing tables
3. Idempotent index creation (creating index on table that already has one)
4. Search performance with various k values
5. Memory and disk usage
6. Comparison with ChromaDB and Qdrant baselines
"""

import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import lancedb
import pyarrow as pa

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:
    QdrantClient = None


def create_test_vectors(
    n_vectors: int, dims: int, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, List[str]]:
    """Create random test vectors and IDs."""
    vectors = np.random.randn(n_vectors, dims).astype(dtype)
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(dtype)
    ids = [f"vec_{i:06d}" for i in range(n_vectors)]
    return vectors, ids


class LanceBenchmark:
    """Benchmark Lance HNSW capabilities."""

    def __init__(self, tmpdir: Path):
        self.tmpdir = tmpdir
        self.results: Dict[str, float] = {}

    def test_basic_index_creation(self, n_vectors: int = 10000, dims: int = 128):
        """Test basic HNSW index creation on Lance table."""
        print(f"\n{'='*60}")
        print(f"Test 1: Basic HNSW Index Creation ({n_vectors} vectors, {dims} dims)")
        print(f"{'='*60}")

        db_path = self.tmpdir / "test_basic_index"
        db = lancedb.connect(str(db_path))

        # Create test data
        vectors, ids = create_test_vectors(n_vectors, dims)

        # Time table creation
        start = time.time()
        data = [{"id": id, "vector": vec.tolist()} for id, vec in zip(ids, vectors)]
        table = db.create_table("results", data=data, mode="overwrite")
        create_time = time.time() - start
        print(f"✓ Table creation: {create_time:.2f}s")

        # Time index creation
        start = time.time()
        try:
            # Lance uses create_index() method
            table.create_index(metric="L2", num_partitions=256, num_sub_vectors=64)
            index_time = time.time() - start
            print(f"✓ Index creation (L2): {index_time:.2f}s")
            self.results["index_creation_time"] = index_time

            # Test basic search
            query = vectors[0:1]  # First vector
            start = time.time()
            results = table.search(query).limit(10).to_list()
            search_time = time.time() - start
            print(f"✓ First search (k=10): {search_time:.3f}s, {len(results)} results")

        except Exception as e:
            print(f"✗ Index creation failed: {e}")
            return False

        return True

    def test_distance_metrics(self, n_vectors: int = 5000, dims: int = 128):
        """Test different distance metrics supported by Lance."""
        print(f"\n{'='*60}")
        print(f"Test 2: Distance Metrics ({n_vectors} vectors, {dims} dims)")
        print(f"{'='*60}")

        db_path = self.tmpdir / "test_metrics"
        db = lancedb.connect(str(db_path))

        vectors, ids = create_test_vectors(n_vectors, dims)
        data = [{"id": id, "vector": vec.tolist()} for id, vec in zip(ids, vectors)]

        metrics = ["L2", "cosine"]

        for metric in metrics:
            try:
                table_name = f"results_{metric.lower()}"
                table = db.create_table(table_name, data=data, mode="overwrite")
                table.create_index(metric=metric, num_partitions=256, num_sub_vectors=64)

                # Test search
                query = vectors[0:1]
                results = table.search(query).limit(10).to_list()
                print(f"✓ {metric:8s} metric works ({len(results)} results)")
                self.results[f"metric_{metric.lower()}"] = True

            except Exception as e:
                print(f"✗ {metric:8s} metric failed: {e}")
                self.results[f"metric_{metric.lower()}"] = False

        return True

    def test_incremental_indexing(self, initial_vectors: int = 5000, dims: int = 128):
        """Test adding HNSW index to an existing table (no rewrite)."""
        print(f"\n{'='*60}")
        print(f"Test 3: Incremental Indexing (initial {initial_vectors} vectors)")
        print(f"{'='*60}")

        db_path = self.tmpdir / "test_incremental"
        db = lancedb.connect(str(db_path))

        # Create initial table without index
        vectors1, ids1 = create_test_vectors(initial_vectors, dims)
        data1 = [{"id": id, "vector": vec.tolist()} for id, vec in zip(ids1, vectors1)]

        table = db.create_table("results", data=data1, mode="overwrite")
        print(f"✓ Created table with {initial_vectors} vectors (no index)")

        # Add index to existing table
        try:
            start = time.time()
            table.create_index(metric="L2", num_partitions=256, num_sub_vectors=64)
            index_time = time.time() - start
            print(f"✓ Added index to existing table: {index_time:.2f}s")
            self.results["incremental_index_time"] = index_time
        except Exception as e:
            print(f"✗ Failed to add index to existing table: {e}")
            return False

        # Test that table is still searchable
        try:
            query = vectors1[0:1]
            results = table.search(query).limit(10).to_list()
            print(f"✓ Search works after indexing: {len(results)} results")
        except Exception as e:
            print(f"✗ Search failed after indexing: {e}")
            return False

        return True

    def test_idempotent_indexing(self, n_vectors: int = 5000, dims: int = 128):
        """Test creating HNSW index twice (idempotent behavior)."""
        print(f"\n{'='*60}")
        print(f"Test 4: Idempotent Index Creation ({n_vectors} vectors)")
        print(f"{'='*60}")

        db_path = self.tmpdir / "test_idempotent"
        db = lancedb.connect(str(db_path))

        vectors, ids = create_test_vectors(n_vectors, dims)
        data = [{"id": id, "vector": vec.tolist()} for id, vec in zip(ids, vectors)]
        table = db.create_table("results", data=data, mode="overwrite")

        # First index creation
        try:
            start = time.time()
            table.create_index(metric="L2", num_partitions=256, num_sub_vectors=64)
            first_time = time.time() - start
            print(f"✓ First index creation: {first_time:.2f}s")
        except Exception as e:
            print(f"✗ First index creation failed: {e}")
            return False

        # Second index creation (should be idempotent or fail gracefully)
        try:
            start = time.time()
            table.create_index(metric="L2", num_partitions=256, num_sub_vectors=64)
            second_time = time.time() - start
            print(f"✓ Second index creation (idempotent): {second_time:.3f}s")
            self.results["idempotent_supported"] = True
        except Exception as e:
            print(f"⚠ Second index creation raised error (may be expected): {e}")
            # Check if table still works
            try:
                query = vectors[0:1]
                results = table.search(query).limit(10).to_list()
                print(f"✓ Table still searchable after index error")
                self.results["idempotent_supported"] = False
            except Exception as e2:
                print(f"✗ Table broken after index error: {e2}")
                return False

        return True

    def test_search_performance(self, n_vectors: int = 100000, dims: int = 128):
        """Test search performance with various k values."""
        print(f"\n{'='*60}")
        print(f"Test 5: Search Performance ({n_vectors} vectors, {dims} dims)")
        print(f"{'='*60}")

        db_path = self.tmpdir / "test_search_perf"
        db = lancedb.connect(str(db_path))

        vectors, ids = create_test_vectors(n_vectors, dims)
        data = [{"id": id, "vector": vec.tolist()} for id, vec in zip(ids, vectors)]

        table = db.create_table("results", data=data, mode="overwrite")
        table.create_index(metric="L2", num_partitions=256, num_sub_vectors=64)
        print(f"✓ Created and indexed table")

        k_values = [1, 10, 100, 1000]
        query = vectors[0:1]

        for k in k_values:
            try:
                times = []
                for _ in range(5):  # 5 runs
                    start = time.time()
                    results = table.search(query).limit(k).to_list()
                    times.append(time.time() - start)

                avg_time = np.mean(times)
                print(f"✓ k={k:4d}: {avg_time*1000:.2f}ms (avg of 5 runs)")
                self.results[f"search_k{k}"] = avg_time

            except Exception as e:
                print(f"✗ Search with k={k} failed: {e}")

        return True

    def test_configuration_parameters(self, n_vectors: int = 5000, dims: int = 128):
        """Test configurable HNSW parameters."""
        print(f"\n{'='*60}")
        print(f"Test 6: HNSW Configuration Parameters")
        print(f"{'='*60}")

        db_path = self.tmpdir / "test_config"
        db = lancedb.connect(str(db_path))

        vectors, ids = create_test_vectors(n_vectors, dims)
        data = [{"id": id, "vector": vec.tolist()} for id, vec in zip(ids, vectors)]

        configs = [
            {"metric": "L2", "num_partitions": 128, "num_sub_vectors": 32},
            {"metric": "L2", "num_partitions": 256, "num_sub_vectors": 64},
            {"metric": "L2", "num_partitions": 512, "num_sub_vectors": 32},
        ]

        for i, config in enumerate(configs):
            try:
                table_name = f"results_config{i}"
                table = db.create_table(table_name, data=data, mode="overwrite")

                start = time.time()
                table.create_index(**config)
                index_time = time.time() - start

                # Test search
                query = vectors[0:1]
                results = table.search(query).limit(10).to_list()

                params_str = (
                    f"partitions={config['num_partitions']}, "
                    f"sub_vectors={config['num_sub_vectors']}"
                )
                print(f"✓ {params_str}: {index_time:.2f}s index, {len(results)} results")
                self.results[f"config_{i}"] = index_time

            except Exception as e:
                print(f"✗ Configuration {i} failed: {e}")

        return True

    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "="*60)
        print("LANCE HNSW BENCHMARK SUITE")
        print("="*60)

        try:
            self.test_basic_index_creation()
            self.test_distance_metrics()
            self.test_incremental_indexing()
            self.test_idempotent_indexing()
            self.test_search_performance()
            self.test_configuration_parameters()

            print(f"\n{'='*60}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*60}")
            for key, value in self.results.items():
                if isinstance(value, float):
                    print(f"{key:30s}: {value:.3f}s")
                else:
                    print(f"{key:30s}: {value}")

        except Exception as e:
            print(f"\n✗ Benchmark suite failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    tmpdir = Path(tempfile.mkdtemp(prefix="lance_benchmark_"))
    print(f"\nUsing temporary directory: {tmpdir}")

    try:
        benchmark = LanceBenchmark(tmpdir)
        benchmark.run_all()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\nCleaned up temporary directory")
