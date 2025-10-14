#!/usr/bin/env python3
"""
Quantum Security Scanner for QMANN
Scans for quantum-specific security vulnerabilities and credential exposure.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any


class QuantumSecurityScanner:
    """Scanner for quantum-specific security issues."""
    
    def __init__(self):
        self.patterns = {
            'credentials': [
                r'QISKIT_IBM_TOKEN\s*=\s*["\'][^"\']+["\']',
                r'qiskit.*token\s*=\s*["\'][^"\']+["\']',
                r'quantum.*api.*key\s*=\s*["\'][^"\']+["\']',
                r'ibm.*quantum.*token\s*=\s*["\'][^"\']+["\']',
            ],
            'hardcoded_urls': [
                r'quantum-computing\.ibm\.com',
                r'auth\.quantum-computing\.ibm\.com',
                r'api\.quantum-computing\.ibm\.com',
            ],
            'circuit_data': [
                r'\.qasm\b',
                r'\.qpy\b',
                r'quantum_circuit\s*=',
                r'QuantumCircuit\(',
            ]
        }
    
    def scan_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Scan a single file for security issues."""
        issues = {'credentials': [], 'hardcoded_urls': [], 'circuit_data': []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for category, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues[category].append({
                            'file': str(file_path),
                            'line': line_num,
                            'match': match.group(),
                            'pattern': pattern
                        })
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            
        return issues
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan entire directory for security issues."""
        all_issues = {'credentials': [], 'hardcoded_urls': [], 'circuit_data': []}
        
        python_files = list(directory.rglob('*.py'))
        
        for file_path in python_files:
            file_issues = self.scan_file(file_path)
            for category in all_issues:
                all_issues[category].extend(file_issues[category])
        
        return {
            'summary': {
                'files_scanned': len(python_files),
                'total_issues': sum(len(issues) for issues in all_issues.values()),
                'high_risk_issues': len(all_issues['credentials']),
                'medium_risk_issues': len(all_issues['hardcoded_urls']),
                'low_risk_issues': len(all_issues['circuit_data'])
            },
            'issues': all_issues
        }


def main():
    parser = argparse.ArgumentParser(description='Quantum Security Scanner')
    parser.add_argument('--output-dir', default='quantum_security', 
                       help='Output directory for scan results')
    parser.add_argument('--check-credentials', action='store_true',
                       help='Check for exposed credentials')
    parser.add_argument('--check-circuits', action='store_true',
                       help='Check for circuit data exposure')
    parser.add_argument('--check-data-leakage', action='store_true',
                       help='Check for data leakage')
    parser.add_argument('src_dir', nargs='?', default='src',
                       help='Source directory to scan')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize scanner
    scanner = QuantumSecurityScanner()
    
    # Scan source directory
    src_path = Path(args.src_dir)
    if not src_path.exists():
        print(f"Error: Source directory {src_path} does not exist")
        sys.exit(1)
    
    print(f"Scanning {src_path} for quantum security issues...")
    results = scanner.scan_directory(src_path)
    
    # Save results
    results_file = output_dir / 'quantum_security_scan.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results['summary']
    print(f"\nQuantum Security Scan Results:")
    print(f"Files scanned: {summary['files_scanned']}")
    print(f"Total issues: {summary['total_issues']}")
    print(f"High risk (credentials): {summary['high_risk_issues']}")
    print(f"Medium risk (URLs): {summary['medium_risk_issues']}")
    print(f"Low risk (circuits): {summary['low_risk_issues']}")
    
    # Exit with error if high-risk issues found
    if summary['high_risk_issues'] > 0:
        print("\n❌ High-risk quantum security issues detected!")
        sys.exit(1)
    else:
        print("\n✅ No high-risk quantum security issues found")


if __name__ == '__main__':
    main()
