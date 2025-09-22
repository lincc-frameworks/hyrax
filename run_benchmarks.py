#!/usr/bin/env python3
"""
Quick script to run Lance format benchmarks and generate summary.

Usage:
    python run_benchmarks.py [--sizes 100,500,1000] [--output results.txt]
"""

import argparse
import sys
from pathlib import Path

# Add benchmarks directory to path
sys.path.append(str(Path(__file__).parent / "benchmarks"))

from detailed_inference_benchmarks import DetailedBenchmarkSuite


def parse_sizes(sizes_str):
    """Parse comma-separated sizes string."""
    return [int(x.strip()) for x in sizes_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Run Lance format benchmarks')
    parser.add_argument('--sizes', default='100,500,1000', 
                       help='Comma-separated dataset sizes to test (default: 100,500,1000)')
    parser.add_argument('--output', help='Output file for results (default: stdout)')
    
    args = parser.parse_args()
    
    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as e:
        print(f"Error parsing sizes: {e}")
        return 1
    
    print(f"Running benchmarks with sizes: {sizes}")
    
    # Run benchmarks
    benchmark = DetailedBenchmarkSuite()
    results = benchmark.run_comprehensive_benchmark(sizes)
    
    # Prepare output
    if args.output:
        with open(args.output, 'w') as f:
            # Redirect stdout to file
            original_stdout = sys.stdout
            sys.stdout = f
            benchmark.print_detailed_analysis(results)
            sys.stdout = original_stdout
        print(f"Results saved to {args.output}")
    else:
        benchmark.print_detailed_analysis(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())