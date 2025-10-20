"""
Pytest configuration and fixtures for QMANN v2.0 test suite.
Automatically logs all test results with timestamp, version, and test number.
"""

import pytest
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Get version from pyproject.toml
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

VERSION = get_version()
TEST_RESULTS_DIR = Path(__file__).parent.parent / "test_results"
TEST_RESULTS_DIR.mkdir(exist_ok=True)

class TestResultsLogger:
    """Logs test results with timestamp, version, and test number"""
    
    def __init__(self):
        self.test_number = 0
        self.results = []
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
    def log_test_result(self, test_name, status, duration, error_msg=None):
        """Log individual test result"""
        self.test_number += 1
        
        result = {
            "test_number": self.test_number,
            "test_name": test_name,
            "status": status,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "error": error_msg
        }
        
        self.results.append(result)
        return result
    
    def save_results(self, summary_stats):
        """Save all results to JSON and TXT files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON format
        json_file = TEST_RESULTS_DIR / f"test_results_{VERSION}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "version": VERSION,
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "summary": summary_stats,
                "tests": self.results
            }, f, indent=2)
        
        # TXT format (human readable)
        txt_file = TEST_RESULTS_DIR / f"test_results_{VERSION}_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"QMANN v{VERSION} TEST RESULTS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total Tests: {len(self.results)}\n")
            f.write(f"\nSummary:\n")
            for key, value in summary_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"DETAILED RESULTS\n")
            f.write(f"{'='*80}\n\n")
            
            for result in self.results:
                f.write(f"Test #{result['test_number']}: {result['test_name']}\n")
                f.write(f"  Status: {result['status']}\n")
                f.write(f"  Duration: {result['duration_seconds']:.4f}s\n")
                f.write(f"  Timestamp: {result['timestamp']}\n")
                if result['error']:
                    f.write(f"  Error: {result['error']}\n")
                f.write("\n")
        
        print(f"\n✅ Test results saved:")
        print(f"   JSON: {json_file}")
        print(f"   TXT:  {txt_file}")
        
        return json_file, txt_file

# Global logger instance
test_logger = TestResultsLogger()

@pytest.fixture(scope="session")
def test_results_logger():
    """Provide test logger to tests"""
    return test_logger

class TestResultsPlugin:
    """Pytest plugin to capture and log test results"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        
    def pytest_runtest_logreport(self, report):
        """Called after each test"""
        if report.when == "call":
            test_name = report.nodeid
            duration = report.duration
            
            if report.passed:
                status = "PASSED"
                self.passed += 1
            elif report.failed:
                status = "FAILED"
                self.failed += 1
                error_msg = str(report.longrepr) if report.longrepr else "Unknown error"
            elif report.skipped:
                status = "SKIPPED"
                self.skipped += 1
                error_msg = None
            else:
                status = "ERROR"
                self.errors += 1
                error_msg = str(report.longrepr) if report.longrepr else "Unknown error"
            
            error_msg = str(report.longrepr) if report.longrepr else None
            test_logger.log_test_result(test_name, status, duration, error_msg)
    
    def pytest_sessionfinish(self, session, exitstatus):
        """Called at end of test session"""
        summary_stats = {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": self.passed + self.failed + self.skipped + self.errors,
            "pass_rate": f"{(self.passed / (self.passed + self.failed + self.errors) * 100):.1f}%" if (self.passed + self.failed + self.errors) > 0 else "N/A"
        }
        
        test_logger.save_results(summary_stats)

# Register plugin
pytest_plugins = []

def pytest_configure(config):
    """Register custom plugin"""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as hardware test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    
    plugin = TestResultsPlugin()
    config.pluginmanager.register(plugin, "test_results_plugin")

