#!/usr/bin/env python3
"""
Generate comprehensive test reports from test results.

Usage:
    python scripts/generate_test_report.py [--latest] [--version VERSION] [--format FORMAT]
    
Examples:
    python scripts/generate_test_report.py --latest              # Latest results
    python scripts/generate_test_report.py --version 2.0.0       # All results for version
    python scripts/generate_test_report.py --format json         # JSON format only
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import statistics

def load_test_results(version: str = None, latest: bool = False) -> List[Dict[str, Any]]:
    """Load test results from JSON files"""
    test_results_dir = Path(__file__).parent.parent / "test_results"
    
    json_files = sorted(test_results_dir.glob("test_results_*.json"))
    
    if not json_files:
        print("❌ No test results found!")
        return []
    
    if latest:
        json_files = [json_files[-1]]
    elif version:
        json_files = [f for f in json_files if f"_{version}_" in f.name]
    
    results = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    return results

def generate_summary_report(results: List[Dict[str, Any]]) -> str:
    """Generate summary report"""
    if not results:
        return "No results to report"
    
    report = []
    report.append("╔" + "="*78 + "╗")
    report.append("║" + " "*20 + "QMANN TEST RESULTS SUMMARY" + " "*32 + "║")
    report.append("╚" + "="*78 + "╝")
    report.append("")
    
    # Aggregate statistics
    total_tests = sum(r['total_tests'] for r in results)
    total_passed = sum(r['summary']['passed'] for r in results)
    total_failed = sum(r['summary']['failed'] for r in results)
    total_skipped = sum(r['summary']['skipped'] for r in results)
    
    report.append(f"📊 AGGREGATE STATISTICS")
    report.append(f"{'─'*80}")
    report.append(f"Total Test Runs:    {len(results)}")
    report.append(f"Total Tests:        {total_tests}")
    report.append(f"✅ Passed:          {total_passed} ({total_passed/total_tests*100:.1f}%)")
    report.append(f"❌ Failed:          {total_failed} ({total_failed/total_tests*100:.1f}%)")
    report.append(f"⏭️  Skipped:         {total_skipped} ({total_skipped/total_tests*100:.1f}%)")
    report.append("")
    
    # Per-run details
    report.append(f"📋 DETAILED RESULTS BY RUN")
    report.append(f"{'─'*80}")
    
    for i, result in enumerate(results, 1):
        report.append(f"\nRun #{i}: {result['version']} ({result['timestamp']})")
        report.append(f"  Session ID: {result['session_id']}")
        report.append(f"  Tests: {result['total_tests']}")
        report.append(f"  Passed: {result['summary']['passed']} | Failed: {result['summary']['failed']} | Skipped: {result['summary']['skipped']}")
        report.append(f"  Pass Rate: {result['summary']['pass_rate']}")
    
    report.append("")
    report.append(f"{'─'*80}")
    
    return "\n".join(report)

def generate_detailed_report(results: List[Dict[str, Any]]) -> str:
    """Generate detailed test-by-test report"""
    if not results:
        return "No results to report"
    
    report = []
    report.append("╔" + "="*78 + "╗")
    report.append("║" + " "*15 + "QMANN DETAILED TEST RESULTS" + " "*35 + "║")
    report.append("╚" + "="*78 + "╝")
    report.append("")
    
    for result in results:
        report.append(f"📌 Session: {result['session_id']} | Version: {result['version']}")
        report.append(f"{'─'*80}")
        
        # Group tests by status
        passed_tests = [t for t in result['tests'] if t['status'] == 'PASSED']
        failed_tests = [t for t in result['tests'] if t['status'] == 'FAILED']
        skipped_tests = [t for t in result['tests'] if t['status'] == 'SKIPPED']
        
        # Passed tests
        if passed_tests:
            report.append(f"\n✅ PASSED ({len(passed_tests)} tests)")
            for test in passed_tests[:10]:  # Show first 10
                report.append(f"  #{test['test_number']}: {test['test_name']}")
                report.append(f"     Duration: {test['duration_seconds']:.4f}s")
            if len(passed_tests) > 10:
                report.append(f"  ... and {len(passed_tests) - 10} more")
        
        # Failed tests
        if failed_tests:
            report.append(f"\n❌ FAILED ({len(failed_tests)} tests)")
            for test in failed_tests:
                report.append(f"  #{test['test_number']}: {test['test_name']}")
                if test['error']:
                    error_lines = test['error'].split('\n')[:3]
                    for line in error_lines:
                        report.append(f"     {line}")
        
        # Skipped tests
        if skipped_tests:
            report.append(f"\n⏭️  SKIPPED ({len(skipped_tests)} tests)")
            for test in skipped_tests[:5]:
                report.append(f"  #{test['test_number']}: {test['test_name']}")
        
        report.append("")
    
    return "\n".join(report)

def generate_performance_report(results: List[Dict[str, Any]]) -> str:
    """Generate performance analysis report"""
    if not results:
        return "No results to report"
    
    report = []
    report.append("╔" + "="*78 + "╗")
    report.append("║" + " "*18 + "QMANN PERFORMANCE ANALYSIS" + " "*34 + "║")
    report.append("╚" + "="*78 + "╝")
    report.append("")
    
    for result in results:
        report.append(f"📊 Session: {result['session_id']}")
        report.append(f"{'─'*80}")
        
        durations = [t['duration_seconds'] for t in result['tests']]
        
        if durations:
            report.append(f"Total Duration: {sum(durations):.2f}s")
            report.append(f"Average Duration: {statistics.mean(durations):.4f}s")
            report.append(f"Median Duration: {statistics.median(durations):.4f}s")
            report.append(f"Min Duration: {min(durations):.4f}s")
            report.append(f"Max Duration: {max(durations):.4f}s")
            
            if len(durations) > 1:
                report.append(f"Std Dev: {statistics.stdev(durations):.4f}s")
            
            # Slowest tests
            report.append(f"\n🐢 Slowest Tests:")
            sorted_tests = sorted(result['tests'], key=lambda t: t['duration_seconds'], reverse=True)
            for test in sorted_tests[:5]:
                report.append(f"  {test['duration_seconds']:.4f}s - {test['test_name'].split('::')[-1]}")
        
        report.append("")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Generate test reports")
    parser.add_argument("--latest", action="store_true", help="Show latest results only")
    parser.add_argument("--version", help="Filter by version")
    parser.add_argument("--format", choices=["summary", "detailed", "performance", "all"], 
                       default="all", help="Report format")
    
    args = parser.parse_args()
    
    results = load_test_results(version=args.version, latest=args.latest)
    
    if not results:
        print("❌ No test results found!")
        return
    
    print()
    
    if args.format in ["summary", "all"]:
        print(generate_summary_report(results))
        print()
    
    if args.format in ["detailed", "all"]:
        print(generate_detailed_report(results))
        print()
    
    if args.format in ["performance", "all"]:
        print(generate_performance_report(results))
        print()

if __name__ == "__main__":
    main()

