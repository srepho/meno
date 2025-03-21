# GitHub Workflows for Meno

This directory contains GitHub Actions workflows for continuous integration, testing, and deployment of the Meno package. These workflows help ensure code quality, compatibility, and proper functioning across different environments.

## Available Workflows

### 1. CI Tests (CPU) - `ci-cpu.yml`

The primary CI workflow runs on standard GitHub-hosted runners with CPU-only environments. This workflow:

- Runs on push to `main` and on pull requests
- Tests on Python 3.8 and 3.10
- Installs all required dependencies
- Runs the full test suite
- Reports code coverage to Codecov

```yaml
name: CI Tests (CPU)

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

### 2. CI Tests (GPU) - `ci-gpu.yml`

This workflow tests GPU-specific functionality on self-hosted runners with NVIDIA GPUs:

- Runs on push to `main` and on pull requests
- Tests on Python 3.10 with CUDA support
- Verifies CUDA availability and compatibility
- Runs GPU-specific tests to ensure acceleration works properly
- Includes alternative Azure ML-based GPU testing

```yaml
name: CI Tests (GPU)

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 3'  # Run weekly on Wednesday at midnight
```

### 3. Secret Scanning - `secret-scanning.yml`

This security workflow scans the codebase for accidentally committed secrets:

- Uses detect-secrets to scan all files for potential secrets
- Runs TruffleHog to detect secrets in git history
- Flags any potential security issues
- Runs on push, pull request, and weekly schedule

```yaml
name: Secret Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly scan on Sunday at midnight
```

### 4. Scheduled Testing - `scheduled-testing.yml`

This workflow performs comprehensive testing across multiple Python versions and installation configurations:

- Runs weekly on Monday at midnight
- Tests on Python 3.8, 3.9, 3.10, 3.11, and 3.12
- Adapts test suite based on Python version
- Tests different installation options (minimal, CPU-optimized, full)
- Reports code coverage to Codecov

```yaml
name: Scheduled Testing

on:
  schedule:
    - cron: '0 0 * * 1'  # Run weekly on Monday at midnight
  workflow_dispatch:  # Allow manual triggering
```

## Workflow Infrastructure

### Self-Hosted GPU Runners

For GPU testing, we use self-hosted runners with NVIDIA GPUs. See `.github/self-hosted-runner-setup.md` for detailed setup instructions.

### Azure ML GPU Testing

As an alternative to self-hosted runners, we also support GPU testing on Azure ML:

- Configuration in `.azure/gpu-test-job.yml`
- Triggered manually or on schedule
- Uses Azure ML GPU compute for running tests
- Requires `AZURE_CREDENTIALS` secret with appropriate access

## Secret Management

The following secrets are required for these workflows:

- `CODECOV_TOKEN`: Token for uploading coverage reports to Codecov
- `AZURE_CREDENTIALS`: JSON credentials for Azure ML GPU testing (only if using Azure ML)

## Badges

Workflow status badges can be added to README.md using:

```markdown
[![Build Status](https://github.com/srepho/meno/actions/workflows/ci-cpu.yml/badge.svg)](https://github.com/srepho/meno/actions)
[![Code Coverage](https://codecov.io/gh/srepho/meno/branch/main/graph/badge.svg)](https://codecov.io/gh/srepho/meno)
```

## Local Workflow Testing

To test workflows locally before pushing:

1. Install [act](https://github.com/nektos/act)
2. Run: `act -j test`

## Troubleshooting

Common issues and solutions:

- **Workflow fails on dependency installation**: Check that all dependencies are correctly specified in `setup.py` and `pyproject.toml`.
- **GPU tests fail**: Ensure self-hosted runners have correct CUDA configuration. See `.github/self-hosted-runner-setup.md`.
- **Coverage reporting fails**: Make sure test command includes `--cov=meno --cov-report=xml` and the Codecov token is correctly set.
- **Secret scanning false positives**: Add exceptions to `detect-secrets` config if needed.

## Making Changes

When modifying workflows:

1. Test changes locally using [act](https://github.com/nektos/act) if possible
2. Create a separate PR for workflow changes
3. Document changes in this README
4. Ensure all workflows pass on your PR before merging