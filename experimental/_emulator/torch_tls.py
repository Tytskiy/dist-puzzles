from functools import cache
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from subprocess import run
from sys import executable, version_info
from tempfile import gettempdir

import torch


@cache
def _extension() -> object:
    torch_version = torch.__version__.replace(".", "_").replace("+", "_")
    source = Path(__file__).with_suffix(".cpp")
    source_version = sha256(source.read_bytes()).hexdigest()[:8]
    suffix = f"{torch_version}_{version_info.major}_{version_info.minor}_{source_version}"
    build_directory = Path(gettempdir()) / f"greenlet_torch_tls_{suffix}"
    build_directory.mkdir(exist_ok=True)
    library = next(build_directory.glob("_greenlet_torch_tls*.so"), None)

    if library is None:
        run(
            [
                executable,
                str(Path(__file__).with_name("build_torch_tls.py")),
                "build_ext",
                "--build-lib",
                str(build_directory),
                "--build-temp",
                str(build_directory / "objects"),
            ],
            check=True,
            cwd=Path(__file__).parent,
        )
        library = next(build_directory.glob("_greenlet_torch_tls*.so"))

    spec = spec_from_file_location("_greenlet_torch_tls", library)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {library}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture() -> object:
    return _extension().ThreadLocalState()
