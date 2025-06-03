import os
import shutil
import time

import numpy as np

from hyrax.vector_dbs.qdrant_impl import QdrantDB


def random_vector_generator(batch_size=1):
    """Create random vectors"""
    while True:
        batch = [np.random.rand(64) for _ in range(batch_size)]
        yield batch


def main():
    """Run the code"""
    if os.path.exists("./vdb_benchmark_output"):
        shutil.rmtree("./vdb_benchmark_output")

    # vdb = ChromaDB({}, {'results_dir': './vdb_benchmark_output'})
    # vdb = MilvusDB({}, {"results_dir": "./vdb_benchmark_output"})
    vdb = QdrantDB({}, {"results_dir": "./vdb_benchmark_output"})
    vdb.create()

    num_batches = 120
    batch_size = 2048
    vector_generator = random_vector_generator(batch_size * num_batches)

    time.sleep(3)

    ids = [str(i) for i in range(batch_size * num_batches)]
    vectors = [t.flatten() for t in next(vector_generator)]

    for i in range(num_batches):
        print(f"Inserting batch {i} of {num_batches}")

        # time.sleep(1)

        # insert vector into vector database
        vdb.insert(
            ids=ids[batch_size * i : batch_size * (i + 1)],
            vectors=vectors[batch_size * i : batch_size * (i + 1)],
        )

        # vdb.insert(
        #     ids=[str(i) for i in range(batch_size*i, batch_size*(i+1))],
        #     vectors=[t.flatten() for t in next(vector_generator)]
        # )


if __name__ == "__main__":
    main()
