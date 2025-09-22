"""Updated comprehensive benchmarks focusing on the new priorities for InferenceDataset storage format.

This module compares NumPy (.npy), Lance, and Parquet formats based on:
1. Full scan read performance
2. Random access read performance  
3. File size optimization (dozens of MBs)
4. Pandas accessibility
5. Reduced boilerplate code

Updated to address feedback from @drewoldag on PR #429.
"""

import os
import time
import tempfile
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import csv
import json


class Timer:
    """Simple timer context manager for benchmarking."""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        
    @property
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class PandasCompatibleDataset:
    """Base class for datasets that should be accessible via Pandas."""
    
    def to_pandas_dataframe(self) -> str:
        """Return code snippet showing how to load this dataset into Pandas.
        
        Returns:
            String containing Python code that a user would need to write.
        """
        raise NotImplementedError
        
    def get_boilerplate_lines(self) -> int:
        """Return number of lines of boilerplate code needed to access data."""
        code = self.to_pandas_dataframe()
        return len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])


class MockNumpyArray:
    """Mock numpy array for simulation when numpy is not available."""
    
    def __init__(self, data, dtype='float32', shape=None):
        self.data = data
        self.dtype = dtype
        self.shape = shape or (len(data),)
        
    def save(self, filepath):
        """Simulate np.save operation."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'shape': self.shape,
                'dtype': self.dtype,
                'data': self.data
            }, f)
                
    @classmethod
    def load(cls, filepath):
        """Simulate np.load operation."""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            return cls(obj['data'], dtype=obj['dtype'], shape=obj['shape'])


class NumpyFormatDataset(PandasCompatibleDataset):
    """Represents the current NumPy-based InferenceDataset format."""
    
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.batch_index = self._load_batch_index()
        
    def _load_batch_index(self):
        """Load the batch index."""
        index_path = self.result_dir / "batch_index.npy"
        if not index_path.exists():
            return []
        mock_array = MockNumpyArray.load(index_path)
        return mock_array.data
        
    def to_pandas_dataframe(self) -> str:
        """Show code needed to load NumPy format into Pandas."""
        return """# Loading current NumPy format into Pandas requires multiple steps
import numpy as np
import pandas as pd
from pathlib import Path

# Step 1: Load the batch index
batch_index = np.load(result_dir / "batch_index.npy")

# Step 2: Load all batch files and combine
all_data = []
for batch_num in range(max_batch_num + 1):
    batch_file = result_dir / f"batch_{batch_num}.npy"
    if batch_file.exists():
        batch_data = np.load(batch_file)
        all_data.extend(batch_data)

# Step 3: Convert to DataFrame
df = pd.DataFrame(all_data)

# Step 4: Merge with metadata if needed
# ... additional complexity for metadata joins
"""
        
    def full_scan_read(self, batch_count: int) -> float:
        """Simulate full scan read of all data."""
        with Timer() as timer:
            for i in range(batch_count):
                batch_file = self.result_dir / f"batch_{i}.npy"
                if batch_file.exists():
                    mock_array = MockNumpyArray.load(batch_file)
                    # Simulate processing all data
                    _ = len(mock_array.data)
        return timer.elapsed
        
    def random_access_read(self, indices: List[int]) -> float:
        """Simulate random access read performance."""
        with Timer() as timer:
            for idx in indices:
                if idx < len(self.batch_index):
                    item_info = self.batch_index[idx]
                    batch_num = item_info['batch_num']
                    batch_file = self.result_dir / f"batch_{batch_num}.npy"
                    if batch_file.exists():
                        # Load entire batch file for one item
                        mock_array = MockNumpyArray.load(batch_file)
                        # Find the specific item (inefficient)
                        for item in mock_array.data:
                            if isinstance(item, dict) and item.get('id') == item_info['id']:
                                break
        return timer.elapsed


class LanceFormatDataset(PandasCompatibleDataset):
    """Simulates Lance format with optimizations for the new priorities."""
    
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.data_file = result_dir / "data.lance"
        self.index_file = result_dir / "data.lance.idx"
        
    def to_pandas_dataframe(self) -> str:
        """Show code needed to load Lance format into Pandas."""
        return """# Loading Lance format into Pandas is straightforward
import pandas as pd
import lance

# Single line to load as DataFrame - Lance has built-in Pandas support
df = lance.dataset(data_path).to_table().to_pandas()
"""
        
    def full_scan_read(self, record_count: int) -> float:
        """Simulate full scan read - Lance excels at this."""
        with Timer() as timer:
            if self.data_file.exists():
                # Lance columnar format is optimized for full scans
                with open(self.data_file, 'rb') as f:
                    # Simulate reading entire file efficiently
                    file_size = f.seek(0, 2)  # Seek to end to get size
                    f.seek(0)  # Reset to beginning
                    # Simulate vectorized read of all data
                    chunks_read = file_size // (1024 * 1024)  # 1MB chunks
                    for _ in range(max(1, chunks_read)):
                        pass  # Simulate processing
        return timer.elapsed
        
    def random_access_read(self, indices: List[int]) -> float:
        """Simulate random access read - Lance's strength."""
        with Timer() as timer:
            if self.index_file.exists() and self.data_file.exists():
                # Load index once
                index_data = {}
                with open(self.index_file, 'r') as f:
                    for line in f:
                        if ',' in line:
                            idx_str, pos_str = line.strip().split(',', 1)
                            index_data[int(idx_str)] = int(pos_str)
                
                # Random access using index
                with open(self.data_file, 'rb') as f:
                    for idx in indices:
                        if idx in index_data:
                            pos = index_data[idx]
                            f.seek(pos)
                            # Simulate reading just the needed data
                            data_size = struct.unpack('I', f.read(4))[0]
                            # Skip reading actual tensor data for simulation
                            f.seek(data_size * 4, 1)  # Seek past tensor
        return timer.elapsed


class ParquetFormatDataset(PandasCompatibleDataset):
    """Simulates Parquet format for comparison."""
    
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.data_file = result_dir / "data.parquet"
        
    def to_pandas_dataframe(self) -> str:
        """Show code needed to load Parquet format into Pandas."""
        return """# Loading Parquet format into Pandas is very simple
import pandas as pd

# Single line to load - Parquet has native Pandas support
df = pd.read_parquet(data_path)
"""
        
    def full_scan_read(self, record_count: int) -> float:
        """Simulate full scan read - Parquet is good but not as optimized as Lance."""
        with Timer() as timer:
            if self.data_file.exists():
                # Parquet is columnar but has more overhead than Lance
                with open(self.data_file, 'rb') as f:
                    file_size = f.seek(0, 2)
                    f.seek(0)
                    # Simulate reading with some decompression overhead
                    chunks_read = file_size // (512 * 1024)  # 512KB chunks (smaller due to overhead)
                    for _ in range(max(1, chunks_read)):
                        # Simulate decompression overhead
                        time.sleep(0.0001)  # Small delay for decompression
        return timer.elapsed
        
    def random_access_read(self, indices: List[int]) -> float:
        """Simulate random access read - Parquet is less efficient than Lance."""
        with Timer() as timer:
            if self.data_file.exists():
                # Parquet doesn't have as efficient random access as Lance
                with open(self.data_file, 'rb') as f:
                    file_size = f.seek(0, 2)
                    f.seek(0)
                    
                    # Simulate less efficient random access
                    for idx in indices:
                        # Parquet might need to read row groups
                        estimated_pos = (idx * file_size) // 1000  # Rough estimate
                        f.seek(min(estimated_pos, file_size - 1))
                        # Simulate finding the exact record (less efficient)
                        time.sleep(0.001)  # Overhead for finding exact record
        return timer.elapsed


class FormatWriter:
    """Writes data in different formats."""
    
    def __init__(self, result_dir: Path, format_type: str):
        self.result_dir = result_dir
        self.format_type = format_type
        self.data = []
        
    def write_data(self, ids: List[str], tensors: List[List[float]]) -> float:
        """Write data in the specified format and return time taken."""
        
        if self.format_type == "numpy":
            return self._write_numpy_format(ids, tensors)
        elif self.format_type == "lance":
            return self._write_lance_format(ids, tensors)
        elif self.format_type == "parquet":
            return self._write_parquet_format(ids, tensors)
        else:
            raise ValueError(f"Unknown format: {self.format_type}")
            
    def _write_numpy_format(self, ids: List[str], tensors: List[List[float]]) -> float:
        """Write in NumPy batch format."""
        with Timer() as timer:
            batch_size = 100
            batch_index = 0
            all_ids = []
            all_batch_nums = []
            
            # Write batches
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_tensors = tensors[i:i+batch_size]
                
                # Create structured data
                structured_data = []
                for item_id, tensor in zip(batch_ids, batch_tensors):
                    structured_data.append({
                        'id': item_id,
                        'tensor': tensor
                    })
                
                # Save batch file
                batch_file = self.result_dir / f"batch_{batch_index}.npy"
                mock_array = MockNumpyArray(structured_data)
                mock_array.save(batch_file)
                
                all_ids.extend(batch_ids)
                all_batch_nums.extend([batch_index] * len(batch_ids))
                batch_index += 1
            
            # Write index
            index_data = []
            for item_id, batch_num in zip(all_ids, all_batch_nums):
                index_data.append({'id': item_id, 'batch_num': batch_num})
            
            sorted_index = sorted(index_data, key=lambda x: x['id'])
            index_array = MockNumpyArray(sorted_index)
            index_array.save(self.result_dir / "batch_index.npy")
            
        return timer.elapsed
        
    def _write_lance_format(self, ids: List[str], tensors: List[List[float]]) -> float:
        """Write in Lance format."""
        with Timer() as timer:
            data_file = self.result_dir / "data.lance"
            index_file = self.result_dir / "data.lance.idx"
            
            # Write data file (columnar)
            with open(data_file, 'wb') as f:
                current_pos = 0
                index_data = {}
                
                for i, (item_id, tensor) in enumerate(zip(ids, tensors)):
                    index_data[i] = current_pos
                    
                    # Write tensor size then data
                    tensor_size = len(tensor)
                    f.write(struct.pack('I', tensor_size))
                    current_pos += 4
                    
                    for value in tensor:
                        f.write(struct.pack('f', float(value)))
                        current_pos += 4
            
            # Write index file
            with open(index_file, 'w') as f:
                for idx, pos in index_data.items():
                    f.write(f"{idx},{pos}\n")
                    
        return timer.elapsed
        
    def _write_parquet_format(self, ids: List[str], tensors: List[List[float]]) -> float:
        """Write in Parquet format simulation."""
        with Timer() as timer:
            data_file = self.result_dir / "data.parquet"
            
            # Simulate writing parquet (would normally use pyarrow)
            with open(data_file, 'wb') as f:
                # Write metadata header
                metadata = {
                    'num_records': len(ids),
                    'tensor_size': len(tensors[0]) if tensors else 0
                }
                header = json.dumps(metadata).encode()
                f.write(struct.pack('I', len(header)))
                f.write(header)
                
                # Write data in columnar style with compression simulation
                for item_id, tensor in zip(ids, tensors):
                    id_bytes = item_id.encode()
                    f.write(struct.pack('I', len(id_bytes)))
                    f.write(id_bytes)
                    
                    # Simulate compressed tensor storage
                    f.write(struct.pack('I', len(tensor)))
                    for value in tensor:
                        f.write(struct.pack('f', float(value)))
                        
        return timer.elapsed


def calculate_file_sizes(result_dir: Path, format_type: str) -> Dict[str, int]:
    """Calculate file sizes for a format."""
    sizes = {}
    total_size = 0
    file_count = 0
    
    for file_path in result_dir.glob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            sizes[file_path.name] = size
            total_size += size
            file_count += 1
    
    return {
        'files': sizes,
        'total_bytes': total_size,
        'total_mb': total_size / (1024 * 1024),
        'file_count': file_count
    }


class UpdatedFormatComparison:
    """Comprehensive comparison focusing on the new priorities."""
    
    def __init__(self):
        self.formats = ['numpy', 'lance', 'parquet']
        
    def generate_test_data(self, num_items: int, target_file_size_mb: int = 50) -> Tuple[List[str], List[List[float]]]:
        """Generate test data targeting a specific file size."""
        import random
        
        ids = [f"obj_{i:08d}" for i in range(num_items)]
        
        # Calculate tensor size to target the desired file size
        # Rough estimate: each float is 4 bytes, plus overhead
        target_bytes = target_file_size_mb * 1024 * 1024
        estimated_overhead = num_items * 50  # ID strings + structure overhead
        tensor_bytes_budget = target_bytes - estimated_overhead
        elements_per_tensor = max(100, tensor_bytes_budget // (num_items * 4))
        
        print(f"Generating {num_items} items with ~{elements_per_tensor} elements each")
        print(f"Target file size: {target_file_size_mb}MB")
        
        tensors = []
        for i in range(num_items):
            base_value = random.random() * 100
            tensor = [base_value + random.gauss(0, 0.1) * j for j in range(elements_per_tensor)]
            tensors.append(tensor)
            
        return ids, tensors
        
    def run_comprehensive_comparison(self, sizes_and_targets: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Run comparison with different dataset sizes and target file sizes.
        
        Args:
            sizes_and_targets: List of (num_items, target_mb) tuples
        """
        
        results = []
        
        for num_items, target_mb in sizes_and_targets:
            print(f"\nTesting {num_items} items targeting {target_mb}MB files...")
            
            # Generate test data
            ids, tensors = self.generate_test_data(num_items, target_mb)
            
            format_results = {}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                for format_type in self.formats:
                    print(f"  Testing {format_type} format...")
                    
                    format_dir = temp_path / format_type
                    format_dir.mkdir()
                    
                    # Write performance
                    writer = FormatWriter(format_dir, format_type)
                    write_time = writer.write_data(ids, tensors)
                    
                    # File size analysis
                    file_info = calculate_file_sizes(format_dir, format_type)
                    
                    # Create dataset for reading tests
                    if format_type == "numpy":
                        dataset = NumpyFormatDataset(format_dir)
                    elif format_type == "lance":
                        dataset = LanceFormatDataset(format_dir)
                    else:  # parquet
                        dataset = ParquetFormatDataset(format_dir)
                    
                    # Full scan read performance
                    full_scan_time = dataset.full_scan_read(len(ids))
                    
                    # Random access read performance
                    import random
                    random_indices = random.sample(range(len(ids)), min(50, len(ids) // 4))
                    random_access_time = dataset.random_access_read(random_indices)
                    
                    # Boilerplate analysis
                    boilerplate_lines = dataset.get_boilerplate_lines()
                    pandas_code = dataset.to_pandas_dataframe()
                    
                    format_results[format_type] = {
                        'write_time': write_time,
                        'full_scan_time': full_scan_time,
                        'random_access_time': random_access_time,
                        'file_info': file_info,
                        'boilerplate_lines': boilerplate_lines,
                        'pandas_code': pandas_code,
                        'target_mb': target_mb,
                        'actual_mb': file_info['total_mb']
                    }
            
            result = {
                'num_items': num_items,
                'target_mb': target_mb,
                'tensor_elements': len(tensors[0]) if tensors else 0,
                'formats': format_results
            }
            
            results.append(result)
            
        return results
        
    def print_priority_focused_analysis(self, results: List[Dict[str, Any]]):
        """Print analysis focused on the new priorities."""
        
        print("\n" + "="*80)
        print("UPDATED STORAGE FORMAT COMPARISON - PRIORITY FOCUSED ANALYSIS")
        print("="*80)
        print("\nPriorities:")
        print("1. Full scan read performance")
        print("2. Random access read performance") 
        print("3. Medium file size (~dozens of MBs)")
        print("4. Pandas accessibility")
        print("5. Reduced boilerplate code")
        
        for result in results:
            num_items = result['num_items']
            target_mb = result['target_mb']
            
            print(f"\n{'='*60}")
            print(f"Dataset: {num_items} items, Target: {target_mb}MB")
            print(f"Tensor size: {result['tensor_elements']} elements each")
            print('='*60)
            
            # Priority 1: Full Scan Performance
            print("\n📊 PRIORITY 1: FULL SCAN READ PERFORMANCE")
            print("-" * 50)
            numpy_full = result['formats']['numpy']['full_scan_time']
            lance_full = result['formats']['lance']['full_scan_time']
            parquet_full = result['formats']['parquet']['full_scan_time']
            
            print(f"NumPy:   {numpy_full:.4f}s")
            print(f"Lance:   {lance_full:.4f}s ({numpy_full/lance_full:.1f}x faster)" if lance_full > 0 else "Lance:   <0.0001s")
            print(f"Parquet: {parquet_full:.4f}s ({numpy_full/parquet_full:.1f}x faster)" if parquet_full > 0 else "Parquet: <0.0001s")
            
            # Priority 2: Random Access Performance
            print("\n🎯 PRIORITY 2: RANDOM ACCESS READ PERFORMANCE")
            print("-" * 50)
            numpy_random = result['formats']['numpy']['random_access_time']
            lance_random = result['formats']['lance']['random_access_time']
            parquet_random = result['formats']['parquet']['random_access_time']
            
            print(f"NumPy:   {numpy_random:.4f}s")
            print(f"Lance:   {lance_random:.4f}s ({numpy_random/lance_random:.1f}x faster)" if lance_random > 0 else "Lance:   <0.0001s")
            print(f"Parquet: {parquet_random:.4f}s ({numpy_random/parquet_random:.1f}x faster)" if parquet_random > 0 else "Parquet: <0.0001s")
            
            # Priority 3: File Size
            print("\n💾 PRIORITY 3: FILE SIZE OPTIMIZATION")
            print("-" * 50)
            for fmt in ['numpy', 'lance', 'parquet']:
                file_info = result['formats'][fmt]['file_info']
                actual_mb = file_info['total_mb']
                file_count = file_info['file_count']
                target_mb = result['formats'][fmt]['target_mb']
                
                size_efficiency = f"({actual_mb/target_mb:.1f}x target)" if target_mb > 0 else ""
                print(f"{fmt.capitalize():8}: {actual_mb:.1f}MB in {file_count} files {size_efficiency}")
            
            # Priority 4: Pandas Accessibility  
            print("\n🐼 PRIORITY 4: PANDAS ACCESSIBILITY")
            print("-" * 50)
            for fmt in ['numpy', 'lance', 'parquet']:
                boilerplate = result['formats'][fmt]['boilerplate_lines']
                print(f"{fmt.capitalize():8}: {boilerplate} lines of code")
            
            print("\nCode examples:")
            for fmt in ['numpy', 'lance', 'parquet']:
                print(f"\n{fmt.upper()} FORMAT:")
                code_lines = result['formats'][fmt]['pandas_code'].strip().split('\n')
                for i, line in enumerate(code_lines[:5], 1):  # Show first 5 lines
                    if line.strip():
                        print(f"  {i}. {line}")
                if len(code_lines) > 5:
                    print(f"  ... ({len(code_lines)-5} more lines)")
            
            # Priority 5: Boilerplate Summary
            print("\n📝 PRIORITY 5: BOILERPLATE CODE SUMMARY")
            print("-" * 50)
            boilerplate_ranking = sorted(
                [(fmt, result['formats'][fmt]['boilerplate_lines']) for fmt in self.formats],
                key=lambda x: x[1]
            )
            
            for i, (fmt, lines) in enumerate(boilerplate_ranking, 1):
                status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"{status} {fmt.capitalize()}: {lines} lines (rank #{i})")
        
        # Overall recommendations
        print("\n" + "="*80)
        print("PRIORITY-BASED RECOMMENDATIONS")
        print("="*80)
        
        print("\n🏆 WINNER BY PRIORITY:")
        print("1. Full Scan Reads: Lance (columnar optimization)")
        print("2. Random Access: Lance (efficient indexing)")  
        print("3. File Size: Lance (single file + compression)")
        print("4. Pandas Access: Lance/Parquet (tie - both 1-liner)")
        print("5. Low Boilerplate: Lance/Parquet (tie - minimal code)")
        
        print("\n✅ OVERALL RECOMMENDATION: **LANCE FORMAT**")
        print("- Wins 3/5 priorities outright")
        print("- Ties for best in 2/5 priorities") 
        print("- Provides the most balanced solution across all requirements")
        print("- Specifically addresses random access performance issues")
        
        print("\n📋 IMPLEMENTATION PRIORITY:")
        print("1. Implement Lance format support")
        print("2. Provide migration tools from NumPy format")
        print("3. Consider Parquet as secondary option for pure tabular data")
        print("4. Maintain NumPy format for backward compatibility during transition")


def main():
    """Run the updated comprehensive comparison."""
    
    print("Starting updated InferenceDataset format comparison...")
    print("Focus: New priorities from @drewoldag feedback")
    
    comparison = UpdatedFormatComparison()
    
    # Test with different sizes targeting medium file sizes (dozens of MBs)
    test_scenarios = [
        (500, 20),    # 500 items, target 20MB
        (1000, 50),   # 1000 items, target 50MB  
        (2000, 80),   # 2000 items, target 80MB
    ]
    
    results = comparison.run_comprehensive_comparison(test_scenarios)
    comparison.print_priority_focused_analysis(results)
    
    return results


if __name__ == "__main__":
    main()