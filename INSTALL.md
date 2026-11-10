# Building mem4py from source

mem4py is a set of Cython/C++ extension modules that use the header-only
[Eigen](https://eigen.tuxfamily.org) library. Building it means compiling those
modules against Python, NumPy and Eigen, then installing them into a Python
environment.

## Tested configuration

The steps below were tested on Windows 11 with:

| Component  | Version                         |
|------------|---------------------------------|
| Python     | 3.12 (64-bit)                   |
| Cython     | 0.29.37                         |
| NumPy      | 1.26.4                          |
| setuptools | 84.x                            |
| Eigen      | 3.4.0                           |
| Compiler   | Microsoft Visual C++ (MSVC)     |

Cython 3.x and NumPy 2.x have not been tested. If you run into build errors,
install the versions listed above first.

## 1. Prerequisites

- **C++ compiler**
  - Windows: install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    with the *Desktop development with C++* workload.
  - Linux: `sudo apt-get install build-essential python3-dev`
- **Eigen 3.4**: Eigen is header-only, so you only need its source tree.
  - Download and unpack it from <https://eigen.tuxfamily.org> (or
    `git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git eigen3`).
  - Linux alternative: `sudo apt-get install libeigen3-dev`, which installs it
    to `/usr/include/eigen3`.

  The folder you point the build at must contain the `Eigen/` subfolder.

## 2. Create a virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "cython==0.29.37" "numpy==1.26.4" setuptools wheel scipy matplotlib colorama
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "cython==0.29.37" "numpy==1.26.4" setuptools wheel scipy matplotlib colorama
```

## 3. Tell the build where Eigen is

`setup.py` reads the Eigen location from the `EIGEN_PATH` environment variable.
If it is not set, the build looks in `C:\Program Files\eigen3` on Windows and
`/usr/include/eigen3` on Linux.

Windows (PowerShell):

```powershell
$env:EIGEN_PATH = "C:\path\to\eigen3"
```

Linux:

```bash
export EIGEN_PATH=/path/to/eigen3
```

## 4. Build and install

From the root of this repository, with the virtual environment active:

```
python -m pip install . --no-build-isolation --no-deps
```

- `--no-build-isolation` builds with the Cython and NumPy already installed in
  your environment. Without it, pip downloads the latest versions into a
  temporary environment, and those may be incompatible.
- `--no-deps` stops pip from changing the packages you installed in step 2.

Rebuilding after you edit a `.pyx` file uses the same command. It replaces the
installed copy.

To build in place without installing (the compiled `.pyd`/`.so` files are
written next to the sources):

```
python setup.py build_ext --inplace
```

Compiler warnings about the deprecated NumPy API or signed/unsigned
conversions are expected and harmless.

## 5. Check the install

Run this from a directory **outside** the repository, so Python imports the
installed copy and not the sources:

```
python -c "import mem4py, solver.DR, assembler.M, readMsh, writeOutput; print(mem4py.__file__)"
```

Because of how `setup.py` lays out the packages, `mem4py` is a single
top-level module, and `solver`, `assembler`, `elements`, `helper` and `ceygen`
are separate top-level packages. Import them as `solver.DR`, not
`mem4py.solver.DR`.

To run the example cases:

```
cd tutorials
python main.py
```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `fatal error C1083: Cannot open include file: 'Eigen/Dense'` | `EIGEN_PATH` is unset or wrong. It must point to the folder that *contains* `Eigen/`. |
| `Microsoft Visual C++ 14.0 or greater is required` | Install the Visual Studio Build Tools (step 1), then open a new terminal. |
| `No module named 'Cython'`, or the build compiles stale `.cpp` files | Install Cython in the active environment. Without it, `setup.py` falls back to the generated `.cpp` files. |
| `ModuleNotFoundError: No module named 'mem4py.solver'` | Import `solver.DR` instead (see step 5). |
| Changes to a `.pyx` file have no effect | Run step 4 again, and make sure you are not running Python from a folder that holds an older in-place build. |
