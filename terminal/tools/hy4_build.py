"""Build Hy4's native source targets for the cards selected by ordinary setup."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_control import acquisition, hy4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    p.add_argument("--arch", action="append", default=[])
    a = p.parse_args()
    acquisition.require("building Hy4 from source")
    # A source setup may have installed cmake beside this interpreter without
    # changing the parent shell's PATH (especially desktop/Windows installs).
    search = os.pathsep.join([str(Path(sys.executable).parent), str(Path(sys.executable).parent / "Scripts"), os.environ.get("PATH", "")])
    cmake = shutil.which("cmake", path=search)
    if not cmake:
        raise ValueError("CMake is missing from the Hy4 source-build environment")
    for command in hy4.build_commands(Path(__file__).resolve().parents[1], a.backend, a.arch):
        subprocess.run([cmake, *command[1:]], check=True)
    print(hy4.engine_path(Path(__file__).resolve().parents[1], a.backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
