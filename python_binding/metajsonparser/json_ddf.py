import io
import os
from glob import glob
from itertools import islice
from typing import Iterable, Optional, Union

from dask import compute, dataframe as dd, delayed
from dask.base import tokenize
from dask.utils import apply, parse_bytes

import cudf

from metajsonparser._lib import json_cudf


@delayed
def _preprocess_block_pickable(
        fname: str, byte_range: tuple[int, int], force_host_read: bool
) -> dict:
    return json_cudf.preprocess_block_wrapper(fname, byte_range, force_host_read)

def _resolve_filenames(path) -> list[str]:
    if isinstance(path, list):
        filenames = path
    elif isinstance(path, str):
        filenames = sorted(glob(path))
    elif hasattr(path, "__fspath__"):
        filenames = sorted(glob(path.__fspath__()))
    else:
        raise TypeError(f"Path type not understood:{type(path)}")

    if not filenames:
        msg = f"A file in: {filenames} does not exist."
        raise FileNotFoundError(msg)

    return filenames

def _get_blocks_metadata(preprocessed_blocks_data: Iterable[dict]) -> list[tuple[tuple[int, int], int, str]]:
    result = []
    end = 0
    for bd in preprocessed_blocks_data:
        start = end
        end = bd["last_eol"] + 1
        count = end - start
        if count:
            result.append(((start, count), bd["num_newlines"], "windows" if bd["win_eol"] else "linux"))
    return result

def _get_meta_df(fname: str, force_host_read: bool) -> cudf.DataFrame:
    with io.open(fname, "r", newline="") as f:
        line = next(iter(f))
        byte_range = (0, len(line))
        win_eol = len(line) >= 2 and line[-2] == '\r'
    meta = json_cudf.read_json(fname, 1, byte_range, win_eol, force_host_read)
    meta.drop(meta.index, inplace=True)
    return meta


def read_json_ddf(
        path, meta: Optional[cudf.DataFrame] = None, blocksize: Union[str, int] = "default", force_host_read: bool = False
) -> dd.DataFrame:
    """
    Read JSON files into a dask_cudf.DataFrame

    It supports loading many files at once using globstrings:

    >>> import metajsonparser as mp
    >>> df = mp.read_json_ddf("myfiles.*.jsonl")

    In some cases it can break up large files:

    >>> df = mp.read_json_ddf("largefile.jsonl", blocksize="256 MiB")

    Parameters
    ----------
    path : str, path object, or file-like object
        A path to a file (a str, pathlib.Path, or
        py._path.local.LocalPath)
    meta : cudf.DataFrame
        If None, schema for the DataFrame will be inferred
    blocksize : int or str, default "256 MiB"
        The target task partition size. If `None`, a single block
        is used for each file.

    Examples
    --------
    >>> import metajsonparser as mp
    >>> ddf = mp.read_json_ddf("sample.jsonl")
    >>> ddf.compute()
       a      b
    0  1     hi
    1  2  hello
    2  3     ai
    """

    # Set default `blocksize`
    if blocksize == "default":
        blocksize = "256 MiB"

    if isinstance(blocksize, str):
        blocksize = parse_bytes(blocksize)

    filenames = _resolve_filenames(path)

    if meta is None:
        meta = _get_meta_df(filenames[0], force_host_read)

    if blocksize is None:
        return _read_json_without_blocksize(path, meta, force_host_read)

    group_sizes = []
    preprocess_tasks = []

    for fn in filenames:
        size = os.path.getsize(fn)
        for start in range(0, size, blocksize):
            byte_range = (
                start,
                blocksize,
            )
            preprocess_tasks.append(_preprocess_block_pickable(fn, byte_range, force_host_read))
        group_sizes.append((size + blocksize - 1) // blocksize)

    assert sum(group_sizes) == len(preprocess_tasks)

    preprocessed_infos: list[dict] = compute(*preprocess_tasks)
    preprocessed_infos_iter = iter(preprocessed_infos)
    file_infos_groups = [
        islice(preprocessed_infos_iter, group_size) for group_size in group_sizes
    ]

    dsk = {}
    i = 0
    name = "read-json-ddf-" + tokenize(path, tokenize)

    for fn, file_info in zip(filenames, file_infos_groups):
        blocks_metadata = _get_blocks_metadata(file_info)
        for byte_range, num_lines, eol in blocks_metadata:
            dsk[(name, i)] = (apply, json_cudf.read_json, [fn, num_lines, byte_range, eol, force_host_read])
            i += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)

def _read_json_without_blocksize(path, meta: cudf.DataFrame, force_host_read: bool):
    filenames = _resolve_filenames(path)

    preprocess_tasks = []

    for fn in filenames:
        preprocess_tasks.append(_preprocess_block_pickable(fn, (0, 0), force_host_read))

    preprocessed_infos: list[dict] = compute(*preprocess_tasks)

    name = "read-json-ddf-" + tokenize(path)

    graph = {(name, i): (apply, json_cudf.read_json, [fn, info["num_newlines"], None, "windows" if info["win_eol"] else "linux", force_host_read])
             for
             i, (fn, info) in enumerate(zip(filenames, preprocessed_infos))}

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
