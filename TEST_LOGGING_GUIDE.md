# QMANN v2.0 Test Logging & Reporting Guide

## Overview

QMANN v2.0 includes an **automatic test logging system** that captures all test results with:
- ✅ **Timestamp** (YYYYMMDD_HHMMSS)
- ✅ **Version** (from pyproject.toml)
- ✅ **Test Number** (sequential numbering)
- ✅ **Detailed Metrics** (duration, status, errors)
- ✅ **Multiple Formats** (JSON, TXT, XML, HTML)

## Quick Start

### Run Tests with Automatic Logging

```bash
# Run all tests with logging
make test-logged

# Run specific test category
make test-simulators-logged
make test-benchmarks-logged
make test-applications-logged
make test-error-mitigation-logged
make test-ablation-logged
make test-continual-logged
make test-integration-logged
```

### View Results

```bash
# View latest summary report
make test-report-summary

# View detailed results
make test-report-detailed

# View performance analysis
make test-report-performance

# View all reports
make test-report-all

# View latest test results
make test-results-view
```

## File Structure

### Test Results Directory

```
test_results/
├── README.md                                    # Documentation
├── test_results_2.0.0_20251020_022406.json    # Structured data
├── test_results_2.0.0_20251020_022406.txt     # Human-readable
├── junit_2.0.0_20251020_022406.xml            # JUnit format
└── report_2.0.0_20251020_022406.html          # HTML report
```

### File Naming Convention

```
test_results_{VERSION}_{TIMESTAMP}.{FORMAT}
```

- **VERSION**: QMANN version (e.g., 2.0.0)
- **TIMESTAMP**: YYYYMMDD_HHMMSS format
- **FORMAT**: json, txt, xml, or html

### Example Files

```
test_results_2.0.0_20251020_143022.json
test_results_2.0.0_20251020_143022.txt
junit_2.0.0_20251020_143022.xml
report_2.0.0_20251020_143022.html
```

## Output Formats

### JSON Format

**File**: `test_results_2.0.0_20251020_143022.json`

```json
{
  "session_id": "20251020_143022",
  "version": "2.0.0",
  "timestamp": "2025-10-20T14:30:22.123456",
  "total_tests": 147,
  "summary": {
    "passed": 123,
    "failed": 16,
    "skipped": 8,
    "errors": 0,
    "total": 147,
    "pass_rate": "88.5%"
  },
  "tests": [
    {
      "test_number": 1,
      "test_name": "tests/simulators/test_quantum_backend.py::test_ideal_simulator_creation",
      "status": "PASSED",
      "duration_seconds": 0.1234,
      "timestamp": "2025-10-20T14:30:22.234567",
      "version": "2.0.0",
      "error": null
    }
  ]
}
```

**Use Cases**:
- Parse programmatically
- Integrate with CI/CD
- Generate custom reports
- Track metrics over time

### TXT Format

**File**: `test_results_2.0.0_20251020_143022.txt`

```
================================================================================
QMANN v2.0.0 TEST RESULTS
================================================================================
Session ID: 20251020_143022
Timestamp: 2025-10-20T14:30:22.123456
Total Tests: 147

Summary:
  passed: 123
  failed: 16
  skipped: 8
  errors: 0
  total: 147
  pass_rate: 88.5%

================================================================================
DETAILED RESULTS
================================================================================

Test #1: tests/simulators/test_quantum_backend.py::test_ideal_simulator_creation
  Status: PASSED
  Duration: 0.1234s
  Timestamp: 2025-10-20T14:30:22.234567
```

**Use Cases**:
- Quick review
- Email reports
- Documentation
- Human-readable logs

### XML Format (JUnit)

**File**: `junit_2.0.0_20251020_143022.xml`

**Use Cases**:
- Jenkins integration
- GitLab CI
- GitHub Actions
- CI/CD pipelines

### HTML Format

**File**: `report_2.0.0_20251020_143022.html`

**Use Cases**:
- Web viewing
- Team sharing
- Interactive reports
- Visual analysis

## Usage Examples

### Run All Tests with Logging

```bash
python scripts/run_tests_with_logging.py
```

Output:
```
================================================================================
QMANN v2.0.0 TEST EXECUTION
================================================================================
Timestamp: 2025-10-20T14:30:22.123456
Test Path: all tests
Marker: none
================================================================================

[... pytest output ...]

✅ Test results saved:
   JSON: test_results/test_results_2.0.0_20251020_143022.json
   TXT:  test_results/test_results_2.0.0_20251020_143022.txt

================================================================================
TEST EXECUTION COMPLETED
================================================================================
Exit Code: 0
Results saved to: test_results/
  - junit_2.0.0_20251020_143022.xml
  - report_2.0.0_20251020_143022.html
  - test_results_2.0.0_20251020_143022.json
  - test_results_2.0.0_20251020_143022.txt
================================================================================
```

### Run Specific Test Category

```bash
# Simulator tests
python scripts/run_tests_with_logging.py simulators

# Benchmark tests
python scripts/run_tests_with_logging.py benchmarks

# Application tests
python scripts/run_tests_with_logging.py applications
```

### Run Tests with Marker

```bash
# Hardware tests
python scripts/run_tests_with_logging.py --marker hardware

# Benchmark tests
python scripts/run_tests_with_logging.py --marker benchmark

# Integration tests
python scripts/run_tests_with_logging.py --marker integration
```

### Generate Reports

```bash
# Summary report
python scripts/generate_test_report.py --latest --format summary

# Detailed report
python scripts/generate_test_report.py --latest --format detailed

# Performance analysis
python scripts/generate_test_report.py --latest --format performance

# All reports
python scripts/generate_test_report.py --latest --format all

# Reports for specific version
python scripts/generate_test_report.py --version 2.0.0 --format all
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Run tests with logging
  run: python scripts/run_tests_with_logging.py

- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_results/
```

### GitLab CI

```yaml
test:
  script:
    - python scripts/run_tests_with_logging.py
  artifacts:
    paths:
      - test_results/
    reports:
      junit: test_results/junit_*.xml
```

### Jenkins

```groovy
stage('Test') {
    steps {
        sh 'python scripts/run_tests_with_logging.py'
        junit 'test_results/junit_*.xml'
        archiveArtifacts artifacts: 'test_results/**'
    }
}
```

## Analyzing Results

### Parse JSON Results

```python
import json
from pathlib import Path

# Load latest results
results_dir = Path("test_results")
latest_json = sorted(results_dir.glob("test_results_*.json"))[-1]

with open(latest_json) as f:
    results = json.load(f)

# Print summary
print(f"Version: {results['version']}")
print(f"Total Tests: {results['total_tests']}")
print(f"Pass Rate: {results['summary']['pass_rate']}")

# Find failed tests
failed_tests = [t for t in results['tests'] if t['status'] == 'FAILED']
for test in failed_tests:
    print(f"❌ {test['test_name']}")
    print(f"   Error: {test['error']}")
```

### Track Performance Over Time

```python
import json
from pathlib import Path
import statistics

results_dir = Path("test_results")
json_files = sorted(results_dir.glob("test_results_*.json"))

durations = []
for json_file in json_files[-10:]:  # Last 10 runs
    with open(json_file) as f:
        data = json.load(f)
        total_duration = sum(t['duration_seconds'] for t in data['tests'])
        durations.append(total_duration)

print(f"Average Duration: {statistics.mean(durations):.2f}s")
print(f"Trend: {'📈 Slower' if durations[-1] > statistics.mean(durations) else '📉 Faster'}")
```

## Best Practices

1. **Regular Testing**: Run tests regularly to catch regressions
2. **Archive Results**: Keep test_results directory clean by archiving old results
3. **Version Tracking**: Use version numbers to track changes
4. **Timestamp Precision**: Timestamps help identify when issues occurred
5. **Error Analysis**: Review error messages for patterns
6. **Performance Monitoring**: Track test duration trends
7. **CI/CD Integration**: Integrate with your CI/CD pipeline
8. **Documentation**: Document test results for paper/publication

## Troubleshooting

### No Test Results Generated

**Problem**: Test results files not created

**Solution**:
1. Check `test_results/` directory exists
2. Verify `tests/conftest.py` is present
3. Ensure pytest.ini is configured
4. Check file permissions

### Missing Plugins

**Problem**: HTML or XML reports not generated

**Solution**:
```bash
# Install optional plugins
pip install pytest-html pytest-json-report
```

### Timestamp Issues

**Problem**: Incorrect timestamps in results

**Solution**:
1. Check system time is correct
2. Verify timezone settings
3. Check Python datetime module

## Configuration

### pytest.ini

```ini
[pytest]
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    benchmark: Performance benchmark tests
    integration: Integration tests
    hardware: Hardware-specific tests
```

### conftest.py

Located in `tests/conftest.py`, handles:
- Test result logging
- Session management
- Fixture configuration
- Plugin registration

## Support

For issues or questions:
1. Check test_results/README.md
2. Review pytest.ini configuration
3. Verify conftest.py setup
4. Check Python version (3.10+)
5. Review GitHub issues

---

**Last Updated**: 2025-10-20
**QMANN Version**: 2.0.0
**Python**: 3.10+

