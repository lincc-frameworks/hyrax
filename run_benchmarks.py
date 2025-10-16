#!/usr/bin/env python3
"""
Updated script to run priority-focused storage format benchmarks.

Usage:
    python run_benchmarks.py [--scenarios 500,20 1000,50] [--output results.txt]
"""

import argparse
import sys
from pathlib import Path

# Add benchmarks directory to path
sys.path.append(str(Path(__file__).parent / "benchmarks"))

from updated_format_comparison import UpdatedFormatComparison


def parse_scenarios(scenarios_str):
    """Parse comma-separated scenarios string like '500,20 1000,50'."""
    scenarios = []
    for scenario in scenarios_str.split():
        parts = scenario.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid scenario format: {scenario}. Expected 'items,mb'")
        scenarios.append((int(parts[0]), int(parts[1])))
    return scenarios


def main():
    parser = argparse.ArgumentParser(description='Run updated format comparison benchmarks')
    parser.add_argument('--scenarios', default='500,20 1000,50 2000,80', 
                       help='Space-separated scenarios as items,target_mb (default: 500,20 1000,50 2000,80)')
    parser.add_argument('--output', help='Output file for results (default: stdout)')
    
    args = parser.parse_args()
    
    try:
        scenarios = parse_scenarios(args.scenarios)
    except ValueError as e:
        print(f"Error parsing scenarios: {e}")
        return 1
    
    print(f"Running priority-focused benchmarks with scenarios: {scenarios}")
    print("Priorities: Full scan, Random access, File size, Pandas access, Low boilerplate")
    
    # Run benchmarks
    comparison = UpdatedFormatComparison()
    results = comparison.run_comprehensive_comparison(scenarios)
    
    # Prepare output
    if args.output:
        with open(args.output, 'w') as f:
            # Redirect stdout to file
            original_stdout = sys.stdout
            sys.stdout = f
            comparison.print_priority_focused_analysis(results)
            sys.stdout = original_stdout
        print(f"Results saved to {args.output}")
    else:
        comparison.print_priority_focused_analysis(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())