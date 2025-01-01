---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''

---
## Bug
### Describe the bug
- A clear and concise description of what the bug is.

### Steps to Reproduce
Steps to reproduce the behavior:
1. Go to '...'
1. Install '....'
1. Run commands '....'

### Expected vs. actual behavior
A clear and concise description of what you expected to happen.

### Screenshots
If applicable, add screenshots to help explain your problem.

## Environment

### operating system and version?
e.g. Ubuntu 23.10 LTS


### NVIDIA drivers and GPUs
please paste the output of (or a more suitable base version for your system):
```sh
nvidia-smi
```

You should see something like:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A5000                Off| 00000000:01:00.0 Off |                  Off |
| 30%   17C    P8               12W / 230W|      6MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Quadro RTX 8000                 Off| 00000000:C1:00.0 Off |                  Off |
| 33%   17C    P8                9W / 260W|      6MiB / 49152MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```


### Python environment and version?
e.g. Conda environment with Python 3.10. Check your Python version with:
```sh
python --version
```

### Docker version
please specify your docker version, You can find it by running: 
```sh
docker --version
```

### Version of brats ?
please specify your version of brats (please make sure you run the latest version):
```sh
pip freeze | grep brats
```

**Additional context**
Add any other context about the problem here.
