# Security Policy

## Supported Versions

We actively support the following versions of QMANN with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

The QMANN team takes security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT create a public GitHub issue

Security vulnerabilities should be reported privately to allow us to fix them before they can be exploited.

### 2. Send a detailed report to our security team

Email: security@qmann-research.org

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if available)
- Your contact information for follow-up

### 3. Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Assessment**: Within 7 days, we will assess the vulnerability and provide an initial classification
- **Fix Development**: Critical vulnerabilities will be addressed within 30 days
- **Public Disclosure**: After a fix is available, we will coordinate responsible disclosure

### 4. Security Considerations for Quantum Computing

QMANN involves quantum computing components that may have unique security considerations:

#### Quantum-Specific Risks
- **Quantum State Leakage**: Sensitive information encoded in quantum states
- **Measurement Attacks**: Unauthorized quantum measurements
- **Decoherence Exploitation**: Attacks leveraging quantum decoherence
- **Classical-Quantum Interface**: Vulnerabilities at the hybrid boundary

#### IBM Quantum Cloud Security
- Always use secure authentication tokens
- Never commit IBM Quantum credentials to version control
- Use environment variables for sensitive configuration
- Regularly rotate access tokens

#### Data Protection
- Encrypt sensitive training data
- Use secure channels for quantum-classical communication
- Implement proper access controls for quantum resources
- Monitor for unusual quantum circuit execution patterns

### 5. Security Best Practices

When using QMANN in production:

1. **Environment Security**
   - Use virtual environments for isolation
   - Keep dependencies updated
   - Regularly scan for known vulnerabilities

2. **Quantum Resource Security**
   - Limit quantum backend access
   - Monitor quantum job submissions
   - Use least-privilege access principles

3. **Data Handling**
   - Encrypt data at rest and in transit
   - Implement proper data sanitization
   - Use secure random number generation

4. **Network Security**
   - Use HTTPS for all communications
   - Implement proper firewall rules
   - Monitor network traffic for anomalies

### 6. Known Security Considerations

#### Current Limitations
- NISQ devices may have limited security guarantees
- Quantum error correction is not yet available
- Classical components follow standard ML security practices

#### Ongoing Research
- Post-quantum cryptography integration
- Quantum-safe authentication methods
- Secure multi-party quantum computation

### 7. Security Updates

Security updates will be:
- Released as patch versions (e.g., 2.0.1)
- Documented in the CHANGELOG.md
- Announced through our security mailing list
- Available through standard package managers

### 8. Acknowledgments

We appreciate the security research community's efforts to improve QMANN's security. Researchers who responsibly disclose vulnerabilities will be:
- Credited in our security advisories (with permission)
- Listed in our Hall of Fame
- Eligible for our bug bounty program (when available)

### 9. Contact Information

- **Security Team**: security@qmann-research.org
- **General Contact**: qmann@research.org
- **Emergency Contact**: +1-XXX-XXX-XXXX (for critical vulnerabilities only)

### 10. Legal

This security policy is subject to our terms of service and privacy policy. By reporting vulnerabilities, you agree to:
- Not publicly disclose the vulnerability until we've had a chance to address it
- Not exploit the vulnerability for malicious purposes
- Provide reasonable time for us to investigate and fix the issue

Thank you for helping keep QMANN and our users safe!

---

**Last Updated**: October 2025
**Version**: 1.0
