# Setting Up a Self-Hosted GitHub Actions Runner with GPU

This guide provides instructions for setting up a self-hosted GitHub Actions runner with GPU support for Meno CI/CD workflows.

## Prerequisites

- Linux machine with NVIDIA GPU
- CUDA drivers installed
- Docker installed (optional, for containerized setup)
- GitHub account with admin access to the repository

## Basic Setup

1. On GitHub, navigate to your repository
2. Go to Settings > Actions > Runners
3. Click "New self-hosted runner" and select Linux
4. Follow the instructions provided by GitHub to download and configure the runner

## GPU Configuration

Ensure the machine has properly installed NVIDIA drivers and CUDA:

```bash
# Check NVIDIA driver installation
nvidia-smi

# Check CUDA installation
nvcc --version
```

## Runner Labels

When configuring the runner, add specific labels to identify it as a GPU runner:

```bash
./config.sh --url https://github.com/yourusername/meno --labels "self-hosted,linux,gpu,cuda"
```

This ensures that GitHub Actions workflows can target this runner specifically with:

```yaml
runs-on: [self-hosted, linux, gpu]
```

## Docker Setup (Optional)

For a containerized setup, ensure Docker and NVIDIA Container Toolkit are installed:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Test the NVIDIA Docker setup:

```bash
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Running the Runner as a Service

Set up the runner as a service so it starts automatically:

```bash
cd actions-runner
sudo ./svc.sh install
sudo ./svc.sh start
```

## Security Considerations

- Create a dedicated user for the runner
- Limit the permissions of this user
- Consider using a dedicated machine for the runner
- Regularly update the runner software

## Monitoring

Set up monitoring to ensure the runner is functioning properly:

```bash
# Check runner status
sudo ./svc.sh status

# View recent logs
tail -f ~/actions-runner/_diag/Runner_*.log
```

## Resources

- [GitHub Docs: Self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [GitHub Actions: GPU Acceleration](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)