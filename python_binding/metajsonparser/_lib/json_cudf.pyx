from typing import Optional

import cudf


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
                                                                int count, end_of_line eol, bint force_host_read,
                                                                bint pinned_read)

def read_json(fname: str, count: int, byte_range: Optional[tuple[int, int]] = None, eol: Optional[str] = None,
              force_host_read: bool = False, pinned_read: bool = True):
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
    c_out_table = generate_example_metadata(c_string, c_offset, c_size, count, c_eol, force_host_read, pinned_read)

    column_names = [x.name.decode() for x in c_out_table.metadata.schema_info]
    df = data_from_unique_ptr(move(c_out_table.tbl), column_names=column_names)
    return cudf.DataFrame._from_data(*df)

cdef extern from "dask_integration.cuh":
    cdef struct block_data:
        ssize_t last_eol
        ssize_t num_newlines
        bint win_eol
    block_data preprocess_block_device(const char * filename, size_t offset, size_t size, bint force_host_read)
    block_data preprocess_block_host(const char * filename, size_t offset, size_t size)

def preprocess_block_device_wrapper(
        fname: str, byte_range: tuple[int, int], force_host_read: bool
) -> dict:
    py_byte_string = fname.encode('ASCII')
    cdef const char * c_string = py_byte_string
    cdef size_t c_offset = byte_range[0]
    cdef size_t c_size = byte_range[1]
    return preprocess_block_device(c_string, c_offset, c_size, force_host_read)

def preprocess_block_host_wrapper(
        fname: str, byte_range: tuple[int, int]
) -> dict:
    py_byte_string = fname.encode('ASCII')
    cdef const char * c_string = py_byte_string
    cdef size_t c_offset = byte_range[0]
    cdef size_t c_size = byte_range[1]
    return preprocess_block_host(c_string, c_offset, c_size)
