import os
from glob import glob
from itertools import islice
from typing import Iterator, Optional, Tuple, Union

import cudf
from dask import compute, dataframe as dd, delayed
from dask.base import tokenize
from dask.utils import apply, parse_bytes

cimport cudf._lib.utils
cimport cudf._lib.cpp.io.types as cudf_io_types

from cudf._lib.utils cimport data_from_unique_ptr
from libcpp.utility cimport move


cdef extern from "parser.cuh" namespace "end_of_line":
    cdef enum end_of_line "end_of_line":
        unknown
        uniks
        win

cdef extern from "parser.cuh":
    cudf_io_types.table_with_metadata generate_example_metadata(const char * filename, size_t offset, size_t size,
                                                                int count, end_of_line eol, bint force_host_read)

def read_json(fname: str, count: int, byte_range: tuple[int, int] = (0, 0), eol: Optional[str] = None,
              force_host_read: Optional[bool] = False):
    cdef end_of_line c_eol

    if eol == "windows":
        c_eol = end_of_line.win
    elif eol == "unix":
        c_eol = end_of_line.uniks
    else:
        c_eol = end_of_line.unknown

    cdef size_t c_offset = (
        byte_range[0] if byte_range is not None else 0
    )
    cdef size_t c_size = (
        byte_range[1] if byte_range is not None else 0
    )

    cdef cudf_io_types.table_with_metadata c_out_table
    py_byte_string = fname.encode('ASCII')
    cdef const char * c_string = py_byte_string
    print(fname, count)
    c_out_table = generate_example_metadata(c_string, c_offset, c_size, count, c_eol, force_host_read)

    column_names = [x.name.decode() for x in c_out_table.metadata.schema_info]
    df = data_from_unique_ptr(move(c_out_table.tbl), column_names=column_names)
    return cudf.DataFrame._from_data(*df)

# cdef struct block_data:
#     ssize_t last_newline
#     ssize_t newlines_count
#     bint win_eol

block_data = Tuple[int, int, bool]

@delayed
def _preprocess_block(
        filename: str, byte_range: tuple[int, int]
) -> block_data:
    with open(filename, "r") as f:
        f.seek(byte_range[0])
        content = f.read(byte_range[1])
    last_eol = -1
    num_lines = 0
    win_eol = False
    for i, c in enumerate(reversed(content)):
        if c == '\n':
            if last_eol == -1:
                last_eol = byte_range[0] + len(content) - 1 - i
            num_lines += 1
        elif c == '\r':
            win_eol = True
    return last_eol, num_lines, win_eol

def _resolve_filenames(path):
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

def _get_blocks_metadata(
        preprocessed_blocks_data: Iterator[block_data]
) -> list[tuple[tuple[int, int], int, str]]:
    result = []
    end = 0
    for last_eol, num_lines, win_eol in preprocessed_blocks_data:
        start = end
        end = last_eol + 1
        count = end - start
        if count:
            result.append(((start, count), num_lines, "windows" if win_eol else "linux"))
    return result

def read_json_ddf(
        path, meta: cudf.DataFrame, blocksize: Union[str, int] = "default", force_host_read: bool = False
) -> dd.DataFrame:
    """
    Read JSON files into a dask_cudf.DataFrame

    This API parallelizes the ``metajsonparser.read_json`` function in the following ways:

    It supports loading many files at once using globstrings:

    >>> import metajsonparser as mp
    >>> df = mp.read_json_ddf("myfiles.*.jsonl")

    In some cases it can break up large files:

    >>> df = mp.read_json_ddf("largefile.jsonl", blocksize="256 MiB")

    It can read CSV files from external resources (e.g. S3, HTTP, FTP)

    Parameters
    ----------
    path : str, path object, or file-like object
        Either a path to a file (a str, pathlib.Path, or
        py._path.local.LocalPath)
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

    if blocksize is None:
        return _read_json_without_blocksize(path, meta, force_host_read)

    filenames = _resolve_filenames(path)

    group_sizes = []
    preprocess_tasks = []

    for fn in filenames:
        size = os.path.getsize(fn)
        for start in range(0, size, blocksize):
            byte_range = (
                start,
                blocksize,
            )
            preprocess_tasks.append(apply(_preprocess_block, [fn, byte_range]))
        group_sizes.append((size + blocksize - 1) // blocksize)

    assert sum(group_sizes) == len(preprocess_tasks)

    # List of tuples
    preprocessed_infos: list[block_data] = compute(*preprocess_tasks)
    preprocessed_infos_iter = iter(preprocessed_infos)
    file_infos_groups: list[islice] = [
        islice(preprocessed_infos_iter, group_size) for group_size in group_sizes
    ]

    dsk = {}
    i = 0
    name = "read-json-ddf-" + tokenize(path, tokenize)

    for fn, file_info in zip(filenames, file_infos_groups):
        blocks_metadata = _get_blocks_metadata(file_info)
        for byte_range, num_lines, eol in blocks_metadata:
            dsk[(name, i)] = (apply, read_json, [fn, num_lines, byte_range, eol, force_host_read])
            i += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)

def _read_json_without_blocksize(path, meta: cudf.DataFrame, force_host_read: bool):
    filenames = _resolve_filenames(path)

    preprocess_tasks = []

    for fn in filenames:
        size = os.path.getsize(fn)
        preprocess_tasks.append(apply(_preprocess_block, [fn, [0, size]]))

    # List of tuples
    preprocessed_infos: list[block_data] = compute(*preprocess_tasks)

    name = "read-json-ddf-" + tokenize(path)

    graph = {(name, i): (apply, read_json, [fn, num_lines, (0, 0), "windows" if win_eol else "linux", force_host_read])
             for
             i, (fn, (_, num_lines, win_eol)) in enumerate(zip(filenames, preprocessed_infos))}

    divisions = [None] * (len(filenames) + 1)

    return dd.core.new_dd_object(graph, name, meta, divisions)
