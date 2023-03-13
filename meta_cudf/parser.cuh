#ifndef META_CUDF_PARSER_CUH
#define META_CUDF_PARSER_CUH
#include <cudf/io/types.hpp>

enum class end_of_line {
    unknown,
    uniks, //< LF, or "\n": end-of-line convention used by Unix
    win   //< CRLF, or "\r\n": end-of-line convention used by MS Windows
};


cudf::io::table_with_metadata generate_example_metadata(const char* filename, size_t offset, size_t size, int count, end_of_line eol, bool force_host_read = false);

#endif //META_CUDF_PARSER_CUH
