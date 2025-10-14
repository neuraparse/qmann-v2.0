"""
Quantum Security Utilities for QMANN
Provides security validation and scanning capabilities for quantum computing applications.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings


logger = logging.getLogger(__name__)


class QuantumSecurityValidator:
    """
    Validates quantum computing code for security vulnerabilities.

    This class provides methods to scan quantum computing code for:
    - Exposed quantum credentials
    - Hardcoded quantum backend URLs
    - Potential quantum information leakage
    - Insecure quantum circuit handling
    """

    def __init__(self):
        """Initialize the quantum security validator."""
        self.security_patterns = {
            "credentials": [
                r'QISKIT_IBM_TOKEN\s*=\s*["\'][^"\']+["\']',
                r'qiskit.*token\s*=\s*["\'][^"\']+["\']',
                r'quantum.*api.*key\s*=\s*["\'][^"\']+["\']',
                r'ibm.*quantum.*token\s*=\s*["\'][^"\']+["\']',
                r'quantum.*password\s*=\s*["\'][^"\']+["\']',
            ],
            "hardcoded_backends": [
                r"quantum-computing\.ibm\.com",
                r"auth\.quantum-computing\.ibm\.com",
                r"api\.quantum-computing\.ibm\.com",
                r'backend\s*=\s*["\'][^"\']*quantum[^"\']*["\']',
            ],
            "circuit_exposure": [
                r"print\s*\(\s*.*circuit.*\)",
                r"logger\.\w+\s*\(\s*.*circuit.*\)",
                r'open\s*\(\s*["\'][^"\']*\.qasm["\']',
                r'save\s*\(\s*["\'][^"\']*\.qpy["\']',
            ],
            "quantum_data_leakage": [
                r"measurement.*print",
                r"result.*print",
                r"counts.*print",
                r"statevector.*print",
            ],
        }

        self.risk_levels = {
            "credentials": "HIGH",
            "hardcoded_backends": "MEDIUM",
            "circuit_exposure": "LOW",
            "quantum_data_leakage": "MEDIUM",
        }

    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Scan a single file for quantum security issues.

        Args:
            file_path: Path to the file to scan

        Returns:
            Dictionary containing scan results
        """
        results = {"file": str(file_path), "issues": [], "risk_score": 0}

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(
                        pattern, content, re.IGNORECASE | re.MULTILINE
                    )
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        issue = {
                            "category": category,
                            "risk_level": self.risk_levels[category],
                            "line": line_num,
                            "match": match.group()[:100],  # Truncate long matches
                            "pattern": pattern,
                            "description": self._get_issue_description(category),
                        }
                        results["issues"].append(issue)

                        # Calculate risk score
                        risk_scores = {"HIGH": 10, "MEDIUM": 5, "LOW": 1}
                        results["risk_score"] += risk_scores.get(issue["risk_level"], 0)

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            results["error"] = str(e)

        return results

    def scan_codebase(self, directory: str) -> Dict[str, Any]:
        """
        Scan entire codebase for quantum security issues.

        Args:
            directory: Root directory to scan

        Returns:
            Dictionary containing comprehensive scan results
        """
        directory_path = Path(directory)

        if not directory_path.exists():
            raise ValueError(f"Directory {directory} does not exist")

        results = {
            "summary": {
                "files_scanned": 0,
                "total_issues": 0,
                "high_risk_issues": [],
                "medium_risk_issues": [],
                "low_risk_issues": [],
                "total_risk_score": 0,
            },
            "file_results": [],
        }

        # Find all Python files
        python_files = list(directory_path.rglob("*.py"))

        for file_path in python_files:
            file_results = self.scan_file(file_path)
            results["file_results"].append(file_results)
            results["summary"]["files_scanned"] += 1
            results["summary"]["total_issues"] += len(file_results["issues"])
            results["summary"]["total_risk_score"] += file_results["risk_score"]

            # Categorize issues by risk level
            for issue in file_results["issues"]:
                risk_level = issue["risk_level"].lower()
                if f"{risk_level}_risk_issues" in results["summary"]:
                    results["summary"][f"{risk_level}_risk_issues"].append(
                        {
                            "file": file_results["file"],
                            "line": issue["line"],
                            "category": issue["category"],
                            "description": issue["description"],
                        }
                    )

        return results

    def _get_issue_description(self, category: str) -> str:
        """Get human-readable description for issue category."""
        descriptions = {
            "credentials": "Potential quantum credentials exposure",
            "hardcoded_backends": "Hardcoded quantum backend URLs",
            "circuit_exposure": "Potential quantum circuit data exposure",
            "quantum_data_leakage": "Potential quantum measurement data leakage",
        }
        return descriptions.get(category, "Unknown security issue")

    def validate_quantum_environment(self) -> Dict[str, Any]:
        """
        Validate the quantum computing environment for security.

        Returns:
            Dictionary containing environment validation results
        """
        validation_results = {
            "environment_secure": True,
            "warnings": [],
            "recommendations": [],
        }

        # Check for exposed environment variables
        sensitive_env_vars = [
            "QISKIT_IBM_TOKEN",
            "IBM_QUANTUM_TOKEN",
            "QUANTUM_API_KEY",
        ]

        for var in sensitive_env_vars:
            if os.getenv(var):
                validation_results["warnings"].append(
                    f"Sensitive environment variable {var} is set"
                )

        # Check for secure token handling
        if not os.getenv("QISKIT_IBM_TOKEN") and not os.getenv("IBM_QUANTUM_TOKEN"):
            validation_results["recommendations"].append(
                "Consider using environment variables for quantum tokens instead of hardcoding"
            )

        return validation_results


def validate_quantum_circuit_security(circuit) -> Dict[str, Any]:
    """
    Validate quantum circuit for security issues.

    Args:
        circuit: Quantum circuit to validate

    Returns:
        Dictionary containing validation results
    """
    validation_results = {"secure": True, "issues": [], "recommendations": []}

    try:
        # Check circuit size (large circuits might be resource intensive)
        if hasattr(circuit, "num_qubits") and circuit.num_qubits > 50:
            validation_results["issues"].append(
                f"Large circuit with {circuit.num_qubits} qubits - consider resource implications"
            )

        # Check for measurement operations
        if hasattr(circuit, "data"):
            has_measurements = any(
                "measure" in str(instruction[0]).lower() for instruction in circuit.data
            )
            if not has_measurements:
                validation_results["recommendations"].append(
                    "Circuit has no measurements - ensure this is intentional"
                )

    except Exception as e:
        validation_results["secure"] = False
        validation_results["issues"].append(f"Error validating circuit: {e}")

    return validation_results


def secure_quantum_token_handler():
    """
    Secure handler for quantum computing tokens.

    Returns:
        Quantum token from secure source or None
    """
    # Try environment variable first (most secure)
    token = os.getenv("QISKIT_IBM_TOKEN")
    if token:
        return token

    # Try alternative environment variable
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if token:
        return token

    # Warn if no secure token found
    warnings.warn(
        "No quantum token found in environment variables. "
        "Consider setting QISKIT_IBM_TOKEN for secure access.",
        UserWarning,
    )

    return None
