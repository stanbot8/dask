"""Compare Dask reshape paths at 2 exact revisions."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BENCHMARK = r"""
import json
import statistics
import sys
import timeit

import dask
import dask.array as da
import numpy as np
from dask.utils import M

role = sys.argv[1]
repetitions = int(sys.argv[2])
source = np.arange(262_144)
array = da.from_array(source, chunks=4_096)
shape = (256, 1_024)


def default_graph():
    return array.reshape(shape)


def no_copy_graph():
    if role == "candidate":
        return array.reshape(shape, copy=False)
    return array.reshape(shape)


def copy_graph():
    if role == "candidate":
        return array.reshape(shape, copy=True)
    return array.reshape(shape).map_blocks(M.copy, meta=array._meta)


def default_compute():
    return default_graph().compute(scheduler="sync")


def copy_compute():
    return copy_graph().compute(scheduler="sync")


def measure(function, number):
    values = timeit.repeat(function, repeat=repetitions, number=number)
    return statistics.median(values) * 1_000_000_000 / number


expected = source.reshape(shape)
checks = {
    "default_graph": np.array_equal(default_compute(), expected),
    "no_copy_graph": np.array_equal(no_copy_graph().compute(scheduler="sync"), expected),
    "copy_graph": np.array_equal(copy_compute(), expected),
}
results = {
    "no_copy_graph": measure(no_copy_graph, 200),
    "default_compute": measure(default_compute, 3),
    "copy_compute": measure(copy_compute, 3),
}
print(
    json.dumps(
        {
            "checks": checks,
            "dask": dask.__version__,
            "numpy": np.__version__,
            "results": results,
        },
        sort_keys=True,
    )
)
"""


def _archive(checkout: Path, revision: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", revision, "dask"],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(destination, filter="data")


def _measure(
    python: Path,
    source: Path,
    role: str,
    repetitions: int,
) -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    completed = subprocess.run(
        [str(python), "-c", BENCHMARK, role, str(repetitions)],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="dask-reshape-benchmark-") as temporary:
        root = Path(temporary)
        source = root / "source"
        candidate = root / "candidate"
        source.mkdir()
        candidate.mkdir()
        _archive(args.checkout, args.source_revision, source)
        _archive(args.checkout, args.candidate_revision, candidate)
        baseline = _measure(Path(sys.executable), source, "source", args.repetitions)
        changed = _measure(Path(sys.executable), candidate, "candidate", args.repetitions)

    cases = []
    for name in sorted(baseline["results"]):
        baseline_ns = float(baseline["results"][name])
        candidate_ns = float(changed["results"][name])
        cases.append(
            {
                "baseline_ns": baseline_ns,
                "candidate_ns": candidate_ns,
                "correct": all(baseline["checks"].values())
                and all(changed["checks"].values()),
                "correctness": {
                    "output_matches": all(baseline["checks"].values())
                    and all(changed["checks"].values())
                },
                "name": name,
                "ratio": candidate_ns / baseline_ns,
            }
        )

    print(
        json.dumps(
            {
                "candidate_revision": args.candidate_revision,
                "cases": cases,
                "command": [
                    "python",
                    "benchmark.py",
                    "--source-revision",
                    args.source_revision,
                    "--candidate-revision",
                    args.candidate_revision,
                    "--repetitions",
                    str(args.repetitions),
                ],
                "environment": {
                    "candidate_dask": changed["dask"],
                    "numpy": changed["numpy"],
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "source_dask": baseline["dask"],
                },
                "framework": "timeit",
                "repetitions": args.repetitions,
                "source_revision": args.source_revision,
                "statistic": "median",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
