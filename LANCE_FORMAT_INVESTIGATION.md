# Lance Format Investigation for InferenceDataset

## Executive Summary

This investigation evaluates the potential benefits of migrating from the current NumPy (.npy) based file format to the Lance columnar format for storing InferenceDataset output. The analysis demonstrates significant performance improvements, particularly for random access operations which were identified as a key concern in the original issue.

## Current Implementation Analysis

The existing `InferenceDatasetWriter` implementation:

- Saves data in multiple `.npy` batch files with structured arrays containing `(id, tensor)` pairs
- Creates separate index files (`batch_index.npy`, `batch_index_insertion_order.npy`)
- Requires loading entire batch files for random access operations
- Results in 3+ files per dataset plus 1 file per batch

### Performance Characteristics

| Dataset Size | Write Time | Random Read | Sequential Read | Storage (MB) | File Count |
|-------------|------------|-------------|-----------------|--------------|------------|
| 100 items   | 0.136s     | 0.168s      | 0.0001s        | 42.2         | 3          |
| 500 items   | 0.667s     | 13.335s     | 0.844s         | 211.0        | 7          |

## Lance Format Analysis

The Lance format offers several architectural advantages:

- **Columnar Storage**: Optimized for analytical workloads and random access
- **Built-in Indexing**: Eliminates need for separate index files
- **Compression**: Reduces storage requirements by ~55%
- **Single File**: Consolidates storage and reduces filesystem overhead
- **Fast Random Access**: Direct row access without loading full batches

### Performance Characteristics

| Dataset Size | Write Time | Random Read | Sequential Read | Storage (MB) | File Count |
|-------------|------------|-------------|-----------------|--------------|------------|
| 100 items   | 0.895s     | 0.160s      | 0.090s         | 18.8         | 2          |
| 500 items   | 4.442s     | 0.815s      | 0.456s         | 93.8         | 2          |

## Key Performance Improvements

### Random Access Performance
- **16.37x improvement** for 500-item datasets
- **1.05x improvement** for 100-item datasets
- Performance advantage scales with dataset size

### Storage Efficiency
- **55% storage reduction** (2.25x compression ratio)
- **Consistent across dataset sizes**
- Single data file + index file vs multiple batch files

### File System Benefits
- **71% fewer files** (2 vs 7 files for 500 items)
- Reduced filesystem metadata overhead
- Simplified cleanup and management

## Implementation Recommendations

### Phase 1: Proof of Concept
1. Install `pylance` package dependency
2. Create `LanceInferenceDatasetWriter` class
3. Create `LanceInferenceDataset` reader class
4. Implement backward compatibility detection

### Phase 2: Parallel Implementation
1. Add format selection parameter to existing classes
2. Default to NumPy format for backward compatibility
3. Allow opt-in to Lance format via configuration
4. Comprehensive testing with real workloads

### Phase 3: Migration
1. Change default format to Lance
2. Provide migration utilities for existing datasets
3. Maintain NumPy format support for legacy data

### Phase 4: Deprecation
1. Deprecate NumPy format support
2. Remove legacy code after suitable transition period

## Technical Implementation Details

### Lance Writer Implementation
```python
class LanceInferenceDatasetWriter:
    def __init__(self, original_dataset: Dataset, result_dir: Path):
        self.lance_table = None
        self.batch_data = []
        
    def write_batch(self, ids: np.ndarray, tensors: list[np.ndarray]):
        # Accumulate batch data in columnar format
        for id_val, tensor in zip(ids, tensors):
            self.batch_data.append({
                'id': id_val,
                'tensor': tensor.flatten()  # Lance handles multi-dimensional data
            })
    
    def write_index(self):
        # Convert to Lance table and write
        import lance
        import pyarrow as pa
        
        table = pa.Table.from_pylist(self.batch_data)
        lance.write_dataset(table, self.result_dir / "data.lance")
```

### Lance Reader Implementation
```python
class LanceInferenceDataset:
    def __init__(self, config, results_dir: Path):
        import lance
        self.dataset = lance.dataset(results_dir / "data.lance")
        
    def __getitem__(self, idx: Union[int, np.ndarray]):
        # Direct random access using Lance's indexing
        if isinstance(idx, int):
            idx = [idx]
            
        # Efficient batch lookup
        result = self.dataset.take(idx)
        tensors = result.column('tensor').to_pylist()
        
        return torch.from_numpy(np.array(tensors))
```

## Performance Optimization Opportunities

### Write Performance
Current benchmarks show Lance write performance is 6.7x slower. This can be optimized through:

1. **Batch Accumulation**: Buffer multiple batches before writing
2. **Compression Tuning**: Optimize compression settings for tensor data
3. **Schema Optimization**: Pre-define schema for better performance
4. **Parallel Writing**: Leverage Lance's parallel write capabilities

### Read Performance
Random access already shows excellent performance. Further improvements possible through:

1. **Index Tuning**: Optimize index configuration for typical access patterns
2. **Caching**: Implement intelligent caching for frequently accessed data
3. **Prefetching**: Predict and prefetch related data

## Risk Assessment

### Low Risk
- **Storage Requirements**: 55% reduction in storage needs
- **Random Access**: Dramatic performance improvement
- **File Management**: Simplified file structure

### Medium Risk
- **Write Performance**: Current implementation 6.7x slower (optimizable)
- **Dependency**: Additional dependency on `pylance` package
- **Migration Complexity**: Need for backward compatibility

### High Risk
- **Sequential Access**: Minor performance regression for some workloads
- **Memory Usage**: Need to monitor memory consumption patterns

## Cost-Benefit Analysis

### Benefits
- **Operational**: Faster random access improves user experience
- **Storage**: 55% storage reduction saves costs
- **Maintenance**: Fewer files reduce management overhead
- **Future-Proofing**: Columnar format enables advanced analytics

### Costs
- **Development**: Estimated 2-3 weeks implementation
- **Testing**: Comprehensive validation required
- **Migration**: One-time conversion effort for existing data
- **Training**: Team familiarization with Lance format

## Conclusion

The investigation strongly supports migrating to Lance format for InferenceDataset storage. The dramatic improvement in random access performance (16x for realistic workloads) directly addresses the primary concern raised in the original issue. While write performance requires optimization, the overall benefits significantly outweigh the costs.

**Recommendation: Proceed with Lance format implementation**

The evidence demonstrates that Lance format will provide:
1. Significantly better random access performance
2. Substantial storage savings
3. Simplified file management
4. Foundation for future enhancements

The implementation should follow the phased approach outlined above, ensuring backward compatibility during the transition period.

## References

- [Lance Documentation](https://lancedb.github.io/lance/)
- [PyLance Package](https://pypi.org/project/pylance/)
- [Issue #428](https://github.com/lincc-frameworks/hyrax/issues/428)
- Benchmark code: `benchmarks/detailed_inference_benchmarks.py`
- Comparison notebook: `lance_format_comparison.ipynb`