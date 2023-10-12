---
id: development
---

# Development

If you want to develop new feature, please follow below steps to set up development environment.

We suggest you to use [Visual Studio Code](https://vscode.github.com/) and install the recommended extensions for this project.
You can also develop online with [GitHub Codespaces](https://github.com/codespaces).

## Check Environment

Follow [System Requirements](../getting-started/installation.mdx).

## Set up

Clone code.
```bash
git clone --recurse-submodules https://github.com/azure/MS-AMP
cd MS-AMP
```

Install MS-AMP.
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .[test] 
make postinstall
```

Install MSCCL and preload msamp_dist library
```bash
cd third_party/msccl
# H100
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
apt-get update
apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build
dpkg -i build/pkg/deb/libnccl2_*.deb
dpkg -i build/pkg/deb/libnccl-dev_2*.deb

cd -
NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so # Change as needed
export LD_PRELOAD="/usr/local/lib/libmsamp_dist.so:${NCCL_LIBRARY}:${LD_PRELOAD}"
```

## Lint and Test

Format code using yapf.
```bash
python3 setup.py format
```

Check code style with mypy and flake8
```bash
python3 setup.py lint
```

Run unit tests.
```bash
python3 setup.py test
```

Open a pull request to main branch on GitHub.
