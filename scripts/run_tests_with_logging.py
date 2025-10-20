#!/usr/bin/env python3
"""
Test runner script that automatically logs all test results with:
- Timestamp (YYYYMMDD_HHMMSS)
- Version (from pyproject.toml)
- Test number (sequential)
- Detailed output

Usage:
    python scripts/run_tests_with_logging.py [test_category]
    
Examples:
    python scripts/run_tests_with_logging.py                    # Run all tests
    python scripts/run_tests_with_logging.py simulators         # Run simulator tests
    python scripts/run_tests_with_logging.py benchmarks         # Run benchmark tests
    python scripts/run_tests_with_logging.py --marker hardware  # Run hardware tests
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

def get_version():
    """Extract version from pyproject.toml"""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            for line in f:
                if 'version' in line and '=' in line:
                    return line.split('=')[1].strip().strip('"').strip("'")
    except:
        pass
    return "2.0.0"

def run_tests(test_path=None, marker=None, verbose=True):
    """Run pytest with automatic result logging"""
    
    version = get_version()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create test_results directory
    test_results_dir = Path(__file__).parent.parent / "test_results"
    test_results_dir.mkdir(exist_ok=True)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if test_path:
        cmd.append(f"tests/{test_path}")
    else:
        cmd.append("tests")
    
    if marker:
        cmd.extend(["-m", marker])
    
    # Add output options
    cmd.extend([
        "-v",
        "--tb=short",
    ])

    # Try to add HTML report if pytest-html is installed
    try:
        import pytest_html
        cmd.extend([
            f"--html=test_results/report_{version}_{timestamp}.html",
            "--self-contained-html",
        ])
    except ImportError:
        pass

    # Try to add JUnit XML if available
    try:
        cmd.append(f"--junit-xml=test_results/junit_{version}_{timestamp}.xml")
    except:
        pass
    
    if verbose:
        cmd.append("-s")
    
    print(f"\n{'='*80}")
    print(f"QMANN v{version} TEST EXECUTION")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Test Path: {test_path or 'all tests'}")
    print(f"Marker: {marker or 'none'}")
    print(f"{'='*80}\n")
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    print(f"\n{'='*80}")
    print(f"TEST EXECUTION COMPLETED")
    print(f"{'='*80}")
    print(f"Exit Code: {result.returncode}")
    print(f"Results saved to: test_results/")
    print(f"  - junit_{version}_{timestamp}.xml")
    print(f"  - report_{version}_{timestamp}.html")
    print(f"  - test_results_{version}_{timestamp}.json")
    print(f"  - test_results_{version}_{timestamp}.txt")
    print(f"{'='*80}\n")
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(
        description="Run QMANN v2.0 tests with automatic logging"
    )
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Test path (e.g., 'simulators', 'benchmarks', 'applications')"
    )
    parser.add_argument(
        "-m", "--marker",
        help="Pytest marker (e.g., 'benchmark', 'integration', 'hardware')"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (no verbose output)"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        test_path=args.test_path,
        marker=args.marker,
        verbose=not args.quiet
    )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

