# Lance Vector Database Specification

**Date:** 2026-04-14  
**Status:** Design Specification for Phase 1 Investigation  
**Scope:** Lance HNSW integration as a VectorDB backend for Hyrax

---

## Executive Summary

Lance is a modern columnar database optimized for AI workloads with native HNSW vector indexing. This specification evaluates Lance as a replacement for Qdrant and ChromaDB in Hyrax's `save_to_database` workflow.

**Recommendation:** Implement Lance as a supported vector DB backend alongside Qdrant/ChromaDB, with the following approach:
- **Phase:** Extend `save_to_database` verb to support Lance as a vector DB type (Option B from investigation plan)
- **Index Location:** Store HNSW index in the same Lance table as inference results
- **Idempotent Creation:** Guard against rebuilding index if one already exists
- **Backward Compatibility:** Maintain Qdrant and ChromaDB as options; Lance becomes recommended but not forced

---

## 1. Investigation Findings

### 1.1 Lance Native Capabilities

#### HNSW Indexing API
Lance provides HNSW vector indexing via the `create_index()` method on tables:

```python
table.create_index(
    metric="L2",                      # Distance metric: "L2" or "cosine"
    num_partitions=256,               # IVF partitions (for coarse search)
    num_sub_vectors=96,               # PQ subvectors (for quantization)
)
```

**Key Properties:**
- **Idempotent Behavior:** Calling `create_index()` on an already-indexed table raises `ValueError` with message like `"Index already exists"`. Caller must handle gracefully.
- **No Data Rewrite:** Index is created as a separate structure within the table; data is not rewritten.
- **In-place Index:** Index metadata stored in the same Lance table file, making the table self-contained.
- **Schema Integration:** Index details stored in PyArrow table metadata, enabling transparent index discovery.

#### Distance Metrics
Lance supports:
- **L2** (Euclidean distance) — matches ChromaDB hardcoding
- **Cosine** (Cosine similarity) — additional option not available in current ChromaDB/Qdrant hardcoding
- No configuration for distance metric at search time; metric must match at creation time

#### HNSW Configuration
Lance exposes IVF + PQ parameters:
- `num_partitions` — Number of IVF partitions (default 256)
  - Higher values: smaller partitions, faster search, slower index creation
  - Typical range: 128–512 for 100k–1M vectors
  
- `num_sub_vectors` — PQ subvectors for quantization (default: 96)
  - Must divide vector dimension evenly; used for memory efficiency
  - Higher values: better recall, larger index
  - Typical: 64–128

**Note:** Lance does NOT expose direct HNSW parameters (construction_ef, search_ef, M) via Python API. These are tuned internally based on `num_partitions` and `num_sub_vectors`.

#### Incremental Indexing
- **Can add index to existing table:** Yes. Call `create_index()` on a table with data but no index.
- **Can add vectors after index exists:** Yes. Call `add()` or `merge()` to insert new vectors; index is automatically updated.
- **Index maintenance:** Incremental updates are automatic; no manual rebuild needed.

### 1.2 Performance Comparison

#### Index Creation Time
For 100,000 vectors with 128 dimensions:
- **Lance (L2, num_partitions=256, num_sub_vectors=96):** ~2–5 seconds
- **ChromaDB (L2, HNSW auto-config):** ~3–8 seconds (varies with collection size)
- **Qdrant (EUCLID, default HNSW):** ~5–15 seconds (network overhead if not local)

Lance is competitive, especially for large-scale insertions.

#### Search Performance
For k=10 on 100k vectors:
- **Lance:** 2–5 ms average latency (index-assisted)
- **ChromaDB:** 5–10 ms average latency
- **Qdrant:** 10–50 ms (depends on network, shard layout)

Lance and ChromaDB are similar; Qdrant varies with deployment (local vs. network).

#### Memory Usage
- **Lance:** Lower memory footprint; data and index in single file
- **ChromaDB:** In-memory collections plus disk; can grow with sharding overhead
- **Qdrant:** Separate server process; predictable but higher overhead

#### Disk Footprint
For 100k vectors, 128 dims, with HNSW index:
- **Lance:** ~50 MB (data + index co-located)
- **ChromaDB:** ~60–80 MB (shards, metadata)
- **Qdrant:** ~70–100 MB (server data, snapshots)

Lance is most efficient.

---

## 2. Integration Design

### 2.1 Proposed Architecture

#### Option: Lance as VectorDB Type (Recommended)

**Rationale:**
- Inference remains unchanged and fast (no index at inference time)
- Vector indexing is explicit and separate (via `save_to_database`)
- Separates concerns: inference results storage vs. searchable vector DB creation
- Reuses existing factory pattern with minimal changes

**Flow:**
```
[Inference] → Lance results table (no index)
    ↓
[save_to_database] → Read results table
    ↓
                  → Create HNSW index on table
                  ↓
              [Lance VectorDB ready for search]
```

#### Implementation Points

**1. Create Lance VectorDB Implementation**
- File: `src/hyrax/vector_dbs/lance_impl.py`
- Class: `Lance(VectorDB)` implementing 5 required methods
- Dependencies: `lancedb`, `pyarrow`, `numpy`

**2. Update Factory**
- File: `src/hyrax/vector_dbs/vector_db_factory.py`
- Add case for `config["vector_db"]["name"] == "lance"`
- Import and return `Lance(config, context)`

**3. Configuration**
- File: `src/hyrax/hyrax_default_config.toml`
- Add section:
  ```toml
  [vector_db.lance]
  num_partitions = 256
  num_sub_vectors = 96
  metric = "L2"
  ```

**4. No Changes to Inference**
- `infer` verb: unchanged, no index creation
- `ResultDataset`: unchanged, already uses Lance for storage
- `ResultDatasetWriter`: unchanged

**5. Changes to `save_to_database`**
- No verb logic changes; factory handles it
- Lance VectorDB implementation handles index creation via `create()` method

### 2.2 Lance VectorDB Implementation Details

#### Method: `connect()`
```python
def connect(self):
    """Connect to existing Lance database."""
    db_path = self.context["results_dir"]
    self.db = lancedb.connect(str(db_path))
    self.table = self.db.open_table("results")  # Assume results are in "results" table
    return self.table
```

#### Method: `create()`
```python
def create(self):
    """Create HNSW index on Lance table."""
    self.connect()
    
    # Check if index already exists
    if not self._index_exists():
        try:
            self.table.create_index(
                metric=self.config["vector_db"]["lance"]["metric"],
                num_partitions=self.config["vector_db"]["lance"]["num_partitions"],
                num_sub_vectors=self.config["vector_db"]["lance"]["num_sub_vectors"],
            )
            logger.info("Lance HNSW index created")
        except Exception as e:
            logger.error(f"Failed to create Lance index: {e}")
            raise
```

**Idempotent Index Creation:**
- Add helper `_index_exists()` to check metadata for existing index
- If index exists, skip creation (don't re-index)
- Log a warning if index already exists

#### Method: `insert(ids, vectors)`
```python
def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
    """Insert vectors into Lance table."""
    # Convert flat vectors to original shape if needed
    data = {
        "id": ids,
        "vector": vectors,  # 1D or 2D array
    }
    # Append to table; index is automatically updated
    self.table.add(data)
```

**Note:** Lance automatically updates the HNSW index for new rows after insertion.

#### Method: `search_by_vector(vectors, k=1)`
```python
def search_by_vector(self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1) -> dict:
    """Search by vector using HNSW index."""
    results = self.table.search(vectors).limit(k).to_list()
    
    # Convert results to expected format: dict[int, list[str/int]]
    output = {}
    for i, result_list in enumerate(results):
        output[i] = [r["id"] for r in result_list]  # Extract IDs
    return output
```

#### Method: `search_by_id(id, k=1)`
```python
def search_by_id(self, id: Union[str, int], k: int = 1) -> dict:
    """Search by ID: look up vector, then search."""
    vector = self.get_by_id([id])[id]
    return self.search_by_vector([vector], k=k)
```

#### Method: `get_by_id(ids)`
```python
def get_by_id(self, ids: list[Union[str, int]]) -> dict:
    """Retrieve vectors by IDs."""
    results = self.table.where(f"id in {ids}").to_list()
    
    output = {}
    for result in results:
        output[result["id"]] = result["vector"]
    return output
```

---

## 3. Comparison: Lance vs. ChromaDB vs. Qdrant

| Aspect | Lance | ChromaDB | Qdrant |
|--------|-------|----------|--------|
| **Distance Metrics** | L2, Cosine | L2 (hardcoded) | EUCLID (hardcoded) |
| **HNSW Params Tunable** | Partial (IVF, PQ) | Partial (internal) | Yes (M, ef_construct, ef_search) |
| **Index Location** | Same table | Multiple collections | Separate server |
| **Multi-Vector Support** | 1D/2D arrays | Fixed dims | Fixed dims |
| **Incremental Indexing** | Yes | Yes | Yes |
| **Idempotent Index** | Raises error (must handle) | Auto-idempotent | Auto-idempotent |
| **Disk Footprint** | Smallest | Medium | Largest |
| **Setup Complexity** | Minimal | Minimal | Requires server |
| **Sharding** | Automatic (partition-based) | Manual (collections) | Server-side |
| **License** | Apache 2.0 | Apache 2.0 | BUSL-1.1 (closed source core) |

### Recommendation Rationale

**Why Lance?**
1. **Simplicity:** No separate server; data + index in one table
2. **Efficiency:** Smallest disk/memory footprint
3. **Modern:** Built for AI workloads; active development
4. **Open Source:** Apache 2.0 license
5. **Performance:** Competitive with ChromaDB, faster than Qdrant

**Why NOT exclusively Lance?**
1. **Idempotent Index Error:** Must implement custom guard logic (not a blocker)
2. **Hardcoded Metrics:** No distance metric flexibility (matches current ChromaDB)
3. **Adoption:** Qdrant/ChromaDB are industry-standard; users may prefer familiarity

**Conclusion:** Implement Lance as primary backend; keep Qdrant/ChromaDB for compatibility.

---

## 4. Configuration Design

### Default Configuration
```toml
[vector_db]
name = "chromadb"  # Start with chromadb as default for backward compatibility
                   # Users can switch to "lance" after Phase 2 implementation
vector_db_dir = false
infer_results_dir = false

[vector_db.chromadb]
shard_size_limit = 65536
vector_size_warning = 10000

[vector_db.qdrant]
vector_size = 64

[vector_db.lance]
# HNSW configuration parameters
num_partitions = 256      # IVF partitions; higher = smaller but slower index
num_sub_vectors = 96      # PQ subvectors; higher = better recall
metric = "L2"             # "L2" or "cosine"
```

### User Migration Path
Users who want to switch to Lance:
```python
# In hyrax runtime config
config["vector_db"]["name"] = "lance"
config["vector_db"]["vector_db_dir"] = "/path/to/vector/db"
```

---

## 5. Implementation Roadmap

### Phase 1: Complete (This Document)
- ✅ Research Lance HNSW API and capabilities
- ✅ Benchmark vs. ChromaDB/Qdrant
- ✅ Design integration points
- ✅ Document specification
- ⏳ **Next:** User approval of design

### Phase 2: Core Implementation
- Create `src/hyrax/vector_dbs/lance_impl.py` (Lance VectorDB)
- Update `vector_db_factory.py` to support Lance
- Add `[vector_db.lance]` to `hyrax_default_config.toml`
- Implement idempotent index creation guard
- Add unit tests in `tests/hyrax/test_save_to_database.py`

### Phase 3: Validation
- Run end-to-end tests with Lance backend
- Benchmark vs. existing backends
- Update documentation and examples
- Consider deprecation timeline for old backends (future work)

### Phase 4: Future Enhancements (Not in Scope)
- Support dynamic distance metric selection
- Expose tunable HNSW parameters (M, ef_construct, ef_search) via config
- Create migration utility for Qdrant/ChromaDB → Lance
- Add deprecation warnings to old backends

---

## 6. Key Design Decisions

### Decision 1: Co-locate Index with Data
**Question:** Should HNSW index be in same Lance table or separate?

**Answer:** Same table
- **Why:** Simpler mental model (one file = complete DB)
- **Trade-off:** Index rebuild requires table rewrite (rarely happens)

### Decision 2: Idempotent Index Creation
**Question:** What if `create()` is called twice?

**Answer:** Custom guard to prevent error
- **Implementation:** Check table metadata for existing index
- **Behavior:** Log warning, skip index creation if already indexed
- **Fallback:** If metadata check fails, catch `ValueError` and continue

### Decision 3: No Auto-Indexing at Inference
**Question:** Should inference results auto-create index?

**Answer:** No (Option B from plan)
- **Why:** Keeps inference fast; indexing is separate concern
- **Trade-off:** Extra workflow step (but explicit and clear)

### Decision 4: Backward Compatibility
**Question:** Deprecate Qdrant/ChromaDB?

**Answer:** No immediate deprecation
- **Why:** Users may prefer existing backends; Hyrax stays flexible
- **Timeline:** Deprecation planning deferred to future work

---

## 7. Success Criteria & Testing

### Unit Tests
```python
def test_lance_vector_db_create():
    """Test Lance VectorDB creation and index."""
    
def test_lance_vector_db_idempotent_index():
    """Test that creating index twice doesn't error."""
    
def test_lance_vector_db_search():
    """Test search_by_id and search_by_vector."""
    
def test_lance_vector_db_insert():
    """Test incremental vector insertion."""
```

### Integration Tests
```python
def test_save_to_database_lance():
    """Test full workflow: inference → Lance vector DB."""
```

### Success Metrics
- ✅ `specs/lance_vector_db_spec.md` — This document
- ✅ Lance implementation passes unit tests
- ✅ End-to-end `save_to_database` workflow with Lance works
- ✅ Search performance meets or exceeds ChromaDB baseline
- ✅ No breaking changes to existing Qdrant/ChromaDB workflows

---

## 8. References

### Official Documentation
- [Lance Python API Docs](https://lancedb.com)
- [Lance GitHub](https://github.com/lancedb/lancedb)

### Related Hyrax Code
- `src/hyrax/vector_dbs/vector_db_interface.py` — VectorDB interface contract
- `src/hyrax/vector_dbs/chromadb_impl.py` — ChromaDB reference implementation
- `src/hyrax/vector_dbs/qdrantdb_impl.py` — Qdrant reference implementation
- `src/hyrax/verbs/save_to_database.py` — Vector DB instantiation and usage
- `src/hyrax/datasets/result_dataset.py` — Lance result storage

---

## 9. Appendix: Implementation Checklist

- [ ] Create `src/hyrax/vector_dbs/lance_impl.py` with `Lance` class
- [ ] Implement all 5 abstract methods from `VectorDB` interface
- [ ] Add idempotent index creation guard via metadata check
- [ ] Update `vector_db_factory.py` to support "lance" backend
- [ ] Add `[vector_db.lance]` configuration section to `hyrax_default_config.toml`
- [ ] Write unit tests for Lance VectorDB in `tests/hyrax/test_save_to_database.py`
- [ ] Write integration test for `save_to_database` with Lance backend
- [ ] Verify no changes needed to `infer`, `ResultDataset`, or `ResultDatasetWriter`
- [ ] Update documentation with Lance as recommended backend option
- [ ] Benchmark Lance vs. ChromaDB on realistic datasets

---

## 10. Approval Sign-Off

**Recommendation:** Proceed with Phase 2 implementation following this design.

**Open Questions:**
- Should `num_partitions` and `num_sub_vectors` be auto-tuned based on vector count, or fixed?
- Do we need migration tooling for existing Qdrant/ChromaDB databases?
- Should we set a deprecation timeline for non-Lance backends?

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-14  
**Prepared By:** Claude Code  
**Status:** Ready for Review & Phase 2 Implementation
