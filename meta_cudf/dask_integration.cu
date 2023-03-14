#include "dask_integration.h"
#include <filesystem>

#include <cudf/io/datasource.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

using namespace std;

// Contains the last index of '\n', number of '\n' and whether '\r' is present
typedef thrust::tuple<ssize_t, ssize_t, bool> mapped_data;
// Contains the index inside the container and the corresponding value
typedef thrust::tuple<ssize_t, uint32_t> indexed_data;

class transform_op
{
    ssize_t block_offset; // offset in bytes

public:
    transform_op(ssize_t block_offset) : block_offset{block_offset} {}

    __device__ mapped_data operator()(indexed_data val)
    {
        ssize_t max_index = -1, count = 0;  // indices are in bytes
        bool win_eol_present = false;
        uint32_t byte_word = val.get<1>();
        uint32_t mask = __vcmpeq4(byte_word, '\n\n\n\n');
        if (mask) {
            ssize_t inblock_offset = val.get<0>() << 2;
            max_index = block_offset + inblock_offset + ((31 - __clz(static_cast<int>(mask))) >> 3);
            ++count;
        }
        if (__vcmpeq4(byte_word, '\r\r\r\r')) {
            win_eol_present = true;
        }
        return mapped_data(max_index, count, win_eol_present);
    }
};

struct reduce_op
{
    __host__ __device__ mapped_data operator()(mapped_data val1, mapped_data val2)
    {
        return mapped_data(
                max(val1.get<0>(), val2.get<0>()),
                val1.get<1>() + val2.get<1>(),
                val1.get<2>() || val2.get<2>()
        );
    }
};

size_t get_size_to_read(const char* filename, size_t offset, size_t size)
{
    size_t file_size = static_cast<size_t>(filesystem::file_size(filename));
    if (size == 0)
        size = file_size;
    return min(size, file_size - offset);
}

block_data preprocess_block(const char* filename, size_t offset, size_t size, bool force_host_read)
{
    rmm::cuda_stream stream;
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

    size = get_size_to_read(filename, offset, size);

    unique_ptr<cudf::io::datasource> source = cudf::io::datasource::create(filename, offset, size);

    rmm::device_uvector<uint32_t> d_buffer((size + 3) / 4, stream, mr);
    d_buffer.set_element_to_zero_async(d_buffer.size() - 1, stream);

    if (!force_host_read && source->supports_device_read() && source->is_device_read_preferred(size))
    {
        source->device_read_async(offset, size, reinterpret_cast<uint8_t*>(d_buffer.data()), stream);
    }
    else
    {
        thrust::host_vector<uint32_t, thrust::cuda::experimental::pinned_allocator<char>> h_data(size);
        source->host_read(offset, size, reinterpret_cast<uint8_t*>(h_data.data()));
        cudaMemcpyAsync(d_buffer.data(), h_data.data(), size, cudaMemcpyHostToDevice, stream.value());
    }

    auto index_iter_begin = thrust::make_zip_iterator(thrust::make_counting_iterator(ssize_t{0}), d_buffer.cbegin());
    auto index_iter_end = thrust::make_zip_iterator(
            thrust::make_counting_iterator(static_cast<ssize_t>(d_buffer.size())), d_buffer.cend());


    auto ret = thrust::transform_reduce(
            rmm::exec_policy(stream, mr),
            index_iter_begin,
            index_iter_end,
            transform_op(offset),
            thrust::make_tuple(ssize_t{-1}, ssize_t{0}, false),
            reduce_op()
    );

    return block_data{ret.get<0>(), ret.get<1>(), ret.get<2>()};
}
