# Setting Up Self-Hosted GPU Runners for Meno

This guide explains how to set up a self-hosted GitHub Actions runner with GPU support for testing the Meno package.

## Requirements

- A machine with CUDA-capable GPU (NVIDIA)
- CUDA Toolkit installed (11.7+ recommended)
- cuDNN installed
- Python 3.8+ installed
- Git installed

## Installation Steps

### 1. Install CUDA and cuDNN

Follow the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for your operating system.

After installing CUDA, install cuDNN by following the [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

Verify installation:
```bash
nvidia-smi
nvcc --version
```

### 2. Create a Python Environment

```bash
# Create a virtual environment
python -m venv meno-gpu-env

# Activate the environment
source meno-gpu-env/bin/activate  # Linux/macOS
# OR
.\meno-gpu-env\Scripts\activate    # Windows

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 3. Install the GitHub Actions Runner

1. In your repository, go to "Settings" > "Actions" > "Runners" > "New self-hosted runner"
2. Follow the instructions to download and configure the runner for your operating system
3. Add the runner to a specific group: `meno-gpu-runners`

Example for Linux:
```bash
# Create a directory
mkdir actions-runner && cd actions-runner

# Download the runner package
curl -o actions-runner-linux-x64-2.314.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.314.1/actions-runner-linux-x64-2.314.1.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-2.314.1.tar.gz

# Configure and specify group
./config.sh --url https://github.com/srepho/meno --token YOUR_TOKEN --runnergroup meno-gpu-runners

# Install and start the service
./svc.sh install
./svc.sh start
```

### 4. Configure the Runner Service

It's recommended to run the GitHub Actions runner as a service so it starts automatically:

#### For Linux (systemd)

Create a systemd service file:
```bash
sudo nano /etc/systemd/system/github-actions-runner.service
```

Add the following content:
```
[Unit]
Description=GitHub Actions Runner
After=network.target

[Service]
User=YOUR_USERNAME
WorkingDirectory=/path/to/actions-runner
ExecStart=/path/to/actions-runner/run.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable github-actions-runner
sudo systemctl start github-actions-runner
```

#### For Windows

Install the runner as a service:
```
.\svc.sh install
.\svc.sh start
```

### 5. Tag the Runner with GPU Capabilities

Add labels to your runner to indicate GPU capabilities:

```bash
./config.sh --url https://github.com/srepho/meno --token YOUR_TOKEN --labels gpu,cuda
```

Or edit the `.runner` file in the runner directory to add these labels manually.

## Workflow Configuration

Now you can use this runner in your workflow by specifying:

```yaml
jobs:
  test-gpu:
    runs-on: self-hosted
    # Or to be more specific:
    # runs-on: [self-hosted, gpu, cuda]
    
    steps:
    # Your workflow steps here
```

## Testing the GPU Runner

You can create a simple test workflow to verify that your runner is working correctly with GPU access:

```yaml
name: GPU Test

on:
  workflow_dispatch:

jobs:
  test-gpu:
    runs-on: self-hosted
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Verify CUDA is available
      run: |
        nvidia-smi
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
```

## Maintenance

- Keep the runner updated regularly:
  ```bash
  cd /path/to/actions-runner
  ./svc.sh stop
  git pull
  ./svc.sh start
  ```

- Monitor the runner's logs:
  ```bash
  tail -f /path/to/actions-runner/_diag/Runner_*.log
  ```

- Check the runner's status:
  ```bash
  ./svc.sh status
  ```

## Troubleshooting

- If the runner has network connectivity issues, check your firewall settings.
- If CUDA is not detected, ensure that the environment variables are set correctly:
  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```
- For permission issues, make sure the user running the service has appropriate permissions.

For detailed troubleshooting, refer to the [GitHub Actions Runner documentation](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners).