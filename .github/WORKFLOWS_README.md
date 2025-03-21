# Meno CI/CD and Security Workflows

[![Build Status](https://github.com/srepho/meno/actions/workflows/ci-cpu.yml/badge.svg)](https://github.com/srepho/meno/actions)
[![Code Coverage](https://codecov.io/gh/srepho/meno/branch/main/graph/badge.svg)](https://codecov.io/gh/srepho/meno)
[![Secret Scanning](https://github.com/srepho/meno/actions/workflows/secret-scanning.yml/badge.svg)](https://github.com/srepho/meno/actions)

This directory contains GitHub Actions workflows for Meno's continuous integration, delivery, and security scanning.

## Workflows

### Secret Scanning (`secret-scanning.yml`)

This workflow scans the codebase for accidentally committed credentials, API keys, and other secrets.

Features:
- Uses multiple scanning tools: detect-secrets, truffleHog, and custom regex patterns
- Checks for common patterns like API keys, passwords, and AWS credentials
- Runs on every push and pull request to prevent secrets from being merged

### CI/CD CPU (`ci-cpu.yml`)

This workflow runs tests and builds on CPU environments.

Features:
- Tests on multiple Python versions (3.8, 3.10, 3.12)
- Runs code quality checks: ruff, black, mypy
- Executes the full test suite for CPU compatibility
- Builds and validates the Python package
- Generates documentation (when configured)

### CI/CD GPU (`ci-gpu.yml`)

This workflow tests on GPU environments.

Features:
- Supports multiple GPU testing options:
  - Self-hosted runners with GPU
  - Azure ML GPU compute
- Runs tests that specifically require GPU
- Generates coverage reports
- Publishes to PyPI when a tag is created

## Setting Up

### Secret Scanning

No additional setup required. This will run automatically on push and pull requests.

### CPU Testing

No additional setup required beyond ensuring all dependencies are in the repository.

### GPU Testing

Choose one of the following options:

1. **Self-hosted runner with GPU:**
   - Follow the setup instructions in `self-hosted-runner-setup.md`
   - Uncomment the appropriate lines in `ci-gpu.yml`

2. **Azure ML for GPU testing:**
   - Create an Azure ML workspace
   - Add your Azure credentials to GitHub repository secrets as `AZURE_CREDENTIALS`
   - Adjust the `.azure/gpu-test-job.yml` file as needed

3. **GitHub-hosted runners with GPU** (additional cost):
   - Uncomment the appropriate lines in `ci-gpu.yml`
   - Be aware of the additional costs for GPU runners

### PyPI Publishing

To enable automatic publishing to PyPI:
1. Generate a PyPI API token
2. Add the token to GitHub repository secrets as `PYPI_API_TOKEN`
3. Create a tag to trigger the release:
   ```bash
   git tag v1.3.4
   git push origin v1.3.4
   ```

## Customizing the Workflows

These workflows are designed to be customizable for specific needs:

- Adjust the Python versions in the matrix strategy
- Modify the test commands or add additional test groups
- Change the build and publishing steps
- Add additional linting or security checks

## Troubleshooting

If you encounter issues with the workflows:

1. Check the workflow run logs in the GitHub Actions tab
2. Verify that all required secrets are properly set
3. For GPU workflows, ensure the compute environment is correctly configured
4. Test the commands locally before pushing to GitHub