# Installation

symphony-dt needs Python 3.11 to 3.13 and CmdStan, the command-line interface to Stan.

## The package

```bash
pip install "symphony-dt[io]"
```

The `io` extra adds Parquet, Arrow, Excel, polars and netCDF support. Without it, CSV still works.

## CmdStan

Stan models are compiled C++ programs, so CmdStan needs a C++ toolchain. On macOS, install the command line tools
with `xcode-select --install`. On Linux, install `g++` and `make`. Then install CmdStan with cmdstanpy:

```bash
python -m cmdstanpy.install_cmdstan --version 2.39.0
```

The SYMPHONY v2 study used CmdStan 2.39.0. symphony-dt looks for CmdStan in three places, in order: the `CMDSTAN`
environment variable, cmdstanpy's default folder and `./.cmdstan` in the working folder.

## Check the installation

```bash
symphony-dt check --compile
```

The check confirms the frozen model files by their sha256, finds CmdStan and compiles pd_v2. Compiled programs
are kept in `~/.cache/symphony-dt`, or in the folder named by `SYMPHONY_DT_CACHE`.

## From source

```bash
git clone https://github.com/a7med7emedan/symphony-dt
cd symphony-dt
pip install -e ".[dev]"
pytest
```
