#ifndef META_JSON_PARSER_DASK_INTEGRATION_CUH
#define META_JSON_PARSER_DASK_INTEGRATION_CUH
#include <sys/types.h>

struct block_data
{
    ssize_t last_eol;
    ssize_t num_newlines;
    bool win_eol;
};

block_data preprocess_block_host(const char* filename, size_t offset, size_t size = 0);
block_data preprocess_block_device(const char* filename, size_t offset, size_t size = 0, bool force_host_read = false);

#endif //META_JSON_PARSER_DASK_INTEGRATION_CUH
