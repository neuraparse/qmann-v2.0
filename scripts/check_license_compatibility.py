#!/usr/bin/env python3
"""
License Compatibility Checker for QMANN
Validates that all dependencies use compatible licenses.
"""

import argparse
import json
import sys
from typing import Dict, List, Set


class LicenseChecker:
    """Check license compatibility for Python packages."""
    
    def __init__(self):
        # Define compatible licenses for Apache 2.0 project
        self.compatible_licenses = {
            'Apache Software License',
            'Apache-2.0',
            'Apache 2.0',
            'MIT License',
            'MIT',
            'BSD License',
            'BSD-2-Clause',
            'BSD-3-Clause',
            'BSD',
            'Python Software Foundation License',
            'PSF',
            'ISC License',
            'ISC',
            'Mozilla Public License 2.0 (MPL 2.0)',
            'MPL-2.0',
            'UNKNOWN'  # Allow unknown for now, but flag for review
        }
        
        # Licenses that are incompatible with Apache 2.0
        self.incompatible_licenses = {
            'GNU General Public License v2 (GPLv2)',
            'GNU General Public License v3 (GPLv3)',
            'GPL-2.0',
            'GPL-3.0',
            'LGPL-2.1',
            'LGPL-3.0',
            'AGPL-3.0',
            'Copyleft'
        }
    
    def check_license(self, license_name: str) -> str:
        """Check if a license is compatible."""
        if not license_name or license_name.strip() == '':
            return 'unknown'
        
        license_clean = license_name.strip()
        
        # Check for exact matches first
        if license_clean in self.compatible_licenses:
            return 'compatible'
        
        if license_clean in self.incompatible_licenses:
            return 'incompatible'
        
        # Check for partial matches
        license_lower = license_clean.lower()
        
        if any(compat.lower() in license_lower for compat in self.compatible_licenses):
            return 'compatible'
        
        if any(incompat.lower() in license_lower for incompat in self.incompatible_licenses):
            return 'incompatible'
        
        return 'unknown'
    
    def analyze_licenses(self, licenses_data: List[Dict]) -> Dict:
        """Analyze license compatibility from pip-licenses output."""
        results = {
            'compatible': [],
            'incompatible': [],
            'unknown': [],
            'summary': {
                'total_packages': len(licenses_data),
                'compatible_count': 0,
                'incompatible_count': 0,
                'unknown_count': 0
            }
        }
        
        for package in licenses_data:
            name = package.get('Name', 'Unknown')
            license_name = package.get('License', 'UNKNOWN')
            version = package.get('Version', 'Unknown')
            
            compatibility = self.check_license(license_name)
            
            package_info = {
                'name': name,
                'version': version,
                'license': license_name,
                'compatibility': compatibility
            }
            
            results[compatibility].append(package_info)
            results['summary'][f'{compatibility}_count'] += 1
        
        return results


def main():
    parser = argparse.ArgumentParser(description='License Compatibility Checker')
    parser.add_argument('--input', required=True,
                       help='Input JSON file from pip-licenses')
    parser.add_argument('--output', required=True,
                       help='Output JSON file for results')
    parser.add_argument('--allowed-licenses', 
                       help='Comma-separated list of allowed licenses')
    
    args = parser.parse_args()
    
    # Load license data
    try:
        with open(args.input, 'r') as f:
            licenses_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input}")
        sys.exit(1)
    
    # Initialize checker
    checker = LicenseChecker()
    
    # Override allowed licenses if provided
    if args.allowed_licenses:
        allowed = set(license.strip() for license in args.allowed_licenses.split(','))
        checker.compatible_licenses.update(allowed)
    
    # Analyze licenses
    results = checker.analyze_licenses(licenses_data)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results['summary']
    print(f"License Compatibility Analysis:")
    print(f"Total packages: {summary['total_packages']}")
    print(f"Compatible: {summary['compatible_count']}")
    print(f"Incompatible: {summary['incompatible_count']}")
    print(f"Unknown: {summary['unknown_count']}")
    
    # Print incompatible licenses
    if results['incompatible']:
        print(f"\n❌ Incompatible licenses found:")
        for pkg in results['incompatible']:
            print(f"  - {pkg['name']} ({pkg['version']}): {pkg['license']}")
    
    # Print unknown licenses for review
    if results['unknown']:
        print(f"\n⚠️  Unknown licenses (need review):")
        for pkg in results['unknown']:
            print(f"  - {pkg['name']} ({pkg['version']}): {pkg['license']}")
    
    # Exit with error if incompatible licenses found
    if summary['incompatible_count'] > 0:
        print(f"\n❌ Found {summary['incompatible_count']} packages with incompatible licenses!")
        sys.exit(1)
    else:
        print(f"\n✅ All licenses are compatible with Apache 2.0")


if __name__ == '__main__':
    main()
