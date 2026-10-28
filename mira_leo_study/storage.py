from __future__ import annotations

import csv
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .errors import OptionalDependencyError


def atomic_write_json(path: str | Path, value) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_csv(path: str | Path, rows: Iterable[Mapping], fieldnames: list[str] | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = list(materialized[0].keys()) if materialized else []
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)


class AppendCsvWriter:
    def __init__(self, path: str | Path, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.fieldnames: list[str] | None = None
        if not resume:
            self.path.unlink(missing_ok=True)
        if resume and self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                self.fieldnames = next(reader, None)

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.fieldnames is None:
            self.fieldnames = list(rows[0].keys())
        mode = "a" if self.path.exists() and (self.resume or self.path.stat().st_size > 0) else "w"
        write_header = mode == "w" or self.path.stat().st_size == 0
        with self.path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self.resume = True


def csv_to_parquet(csv_path: str | Path, parquet_path: str | Path, chunk_size: int = 10000) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.csv as pacsv  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Parquet export requires pyarrow; install requirements.txt"
        ) from exc
    source = Path(csv_path)
    destination = Path(parquet_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = pacsv.open_csv(source, read_options=pacsv.ReadOptions(block_size=max(1 << 20, chunk_size * 256)))
    writer = None
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def write_parquet(path: str | Path, rows: Iterable[Mapping]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Parquet export requires pyarrow; install requirements.txt"
        ) from exc
    materialized = list(rows)
    table = pa.Table.from_pylist(materialized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")


def atomic_pickle(path: str | Path, value) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
