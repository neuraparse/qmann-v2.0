#!/usr/bin/env python3
"""
Security Report Generator for QMANN
Generates comprehensive security reports from various security scan results.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class SecurityReportGenerator:
    """Generate comprehensive security reports."""
    
    def __init__(self):
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'high_issues': 0,
                'medium_issues': 0,
                'low_issues': 0
            },
            'scans': {}
        }
    
    def load_json_report(self, file_path: Path) -> Dict:
        """Load a JSON report file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}
    
    def process_safety_report(self, safety_file: Path):
        """Process Safety vulnerability scan results."""
        data = self.load_json_report(safety_file)
        if not data:
            return
        
        vulnerabilities = data.get('vulnerabilities', [])
        self.report_data['scans']['dependency_vulnerabilities'] = {
            'tool': 'Safety',
            'total_vulnerabilities': len(vulnerabilities),
            'vulnerabilities': vulnerabilities
        }
        
        # Count by severity
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'unknown').lower()
            if severity in ['critical', 'high', 'medium', 'low']:
                self.report_data['summary'][f'{severity}_issues'] += 1
            self.report_data['summary']['total_issues'] += 1
    
    def process_bandit_report(self, bandit_file: Path):
        """Process Bandit security scan results."""
        data = self.load_json_report(bandit_file)
        if not data:
            return
        
        results = data.get('results', [])
        self.report_data['scans']['code_security'] = {
            'tool': 'Bandit',
            'total_issues': len(results),
            'issues': results
        }
        
        # Count by severity
        for issue in results:
            severity = issue.get('issue_severity', 'unknown').lower()
            if severity in ['critical', 'high', 'medium', 'low']:
                self.report_data['summary'][f'{severity}_issues'] += 1
            self.report_data['summary']['total_issues'] += 1
    
    def process_quantum_security_report(self, quantum_file: Path):
        """Process quantum-specific security scan results."""
        data = self.load_json_report(quantum_file)
        if not data:
            return
        
        summary = data.get('summary', {})
        issues = data.get('issues', {})
        
        self.report_data['scans']['quantum_security'] = {
            'tool': 'Quantum Security Scanner',
            'files_scanned': summary.get('files_scanned', 0),
            'total_issues': summary.get('total_issues', 0),
            'high_risk_issues': summary.get('high_risk_issues', 0),
            'medium_risk_issues': summary.get('medium_risk_issues', 0),
            'low_risk_issues': summary.get('low_risk_issues', 0),
            'issues': issues
        }
        
        # Add to summary
        self.report_data['summary']['critical_issues'] += summary.get('high_risk_issues', 0)
        self.report_data['summary']['medium_issues'] += summary.get('medium_risk_issues', 0)
        self.report_data['summary']['low_issues'] += summary.get('low_risk_issues', 0)
        self.report_data['summary']['total_issues'] += summary.get('total_issues', 0)
    
    def process_license_report(self, license_file: Path):
        """Process license compatibility report."""
        data = self.load_json_report(license_file)
        if not data:
            return
        
        summary = data.get('summary', {})
        self.report_data['scans']['license_compliance'] = {
            'tool': 'License Checker',
            'total_packages': summary.get('total_packages', 0),
            'compatible_count': summary.get('compatible_count', 0),
            'incompatible_count': summary.get('incompatible_count', 0),
            'unknown_count': summary.get('unknown_count', 0),
            'incompatible_packages': data.get('incompatible', []),
            'unknown_packages': data.get('unknown', [])
        }
        
        # Incompatible licenses are high priority
        incompatible_count = summary.get('incompatible_count', 0)
        self.report_data['summary']['high_issues'] += incompatible_count
        self.report_data['summary']['total_issues'] += incompatible_count
    
    def generate_html_report(self, output_file: Path):
        """Generate HTML security report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>QMANN Security Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { background-color: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .critical { color: #d32f2f; }
        .high { color: #f57c00; }
        .medium { color: #fbc02d; }
        .low { color: #388e3c; }
        .scan-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>QMANN Security Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>Security Summary</h2>
        <p><strong>Total Issues:</strong> {total_issues}</p>
        <p><span class="critical">Critical:</span> {critical_issues}</p>
        <p><span class="high">High:</span> {high_issues}</p>
        <p><span class="medium">Medium:</span> {medium_issues}</p>
        <p><span class="low">Low:</span> {low_issues}</p>
    </div>
    
    {scan_sections}
    
</body>
</html>
        """
        
        # Generate scan sections
        scan_sections = ""
        for scan_name, scan_data in self.report_data['scans'].items():
            scan_sections += f"""
    <div class="scan-section">
        <h3>{scan_data.get('tool', scan_name)}</h3>
        <p>Total Issues: {scan_data.get('total_issues', 0)}</p>
        <!-- Add more detailed scan results here -->
    </div>
            """
        
        html_content = html_template.format(
            timestamp=self.report_data['timestamp'],
            total_issues=self.report_data['summary']['total_issues'],
            critical_issues=self.report_data['summary']['critical_issues'],
            high_issues=self.report_data['summary']['high_issues'],
            medium_issues=self.report_data['summary']['medium_issues'],
            low_issues=self.report_data['summary']['low_issues'],
            scan_sections=scan_sections
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def generate_json_report(self, output_file: Path):
        """Generate JSON security report."""
        with open(output_file, 'w') as f:
            json.dump(self.report_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Security Report Generator')
    parser.add_argument('--input-dir', default='.',
                       help='Directory containing security scan results')
    parser.add_argument('--output', required=True,
                       help='Output file for the report')
    parser.add_argument('--format', choices=['html', 'json'], default='html',
                       help='Output format')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    
    # Initialize report generator
    generator = SecurityReportGenerator()
    
    # Process various security scan results
    scan_files = {
        'safety-report.json': generator.process_safety_report,
        'bandit-report.json': generator.process_bandit_report,
        'quantum_security/quantum_security_scan.json': generator.process_quantum_security_report,
        'license-compliance-report.json': generator.process_license_report,
    }
    
    for filename, processor in scan_files.items():
        file_path = input_dir / filename
        if file_path.exists():
            processor(file_path)
        else:
            print(f"Warning: {filename} not found in {input_dir}")
    
    # Generate report
    if args.format == 'html':
        generator.generate_html_report(output_file)
    else:
        generator.generate_json_report(output_file)
    
    print(f"Security report generated: {output_file}")
    
    # Print summary
    summary = generator.report_data['summary']
    print(f"\nSecurity Summary:")
    print(f"Total Issues: {summary['total_issues']}")
    print(f"Critical: {summary['critical_issues']}")
    print(f"High: {summary['high_issues']}")
    print(f"Medium: {summary['medium_issues']}")
    print(f"Low: {summary['low_issues']}")


if __name__ == '__main__':
    main()
