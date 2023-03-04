#include <thrust/transform_reduce.h>
#include <iostream>

#include <chrono>
using namespace std::chrono;
using namespace std;

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cudf/io/datasource.hpp>
#include <filesystem>

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/exec_policy.hpp>

#include <typeinfo>

// Contains the first index of '\n', the last index of '\n', number of '\n' and whether '\r' is present
typedef thrust::tuple<ssize_t, ssize_t, ssize_t, bool> mapped_data;
// Contains the index inside the container and the corresponding value
typedef thrust::tuple<ssize_t, uint32_t> indexed_data;

class transform_op
{
    ssize_t block_offset; // offset in bytes

public:
    transform_op(ssize_t block_offset) : block_offset{block_offset} {}

    __device__
        mapped_data
        operator()(indexed_data val)
    {
        ssize_t min_index = SSIZE_MAX, max_index = -1, count = 0;  // indices are in bytes
        bool present = false;
        uint32_t byte_word = val.get<1>();
        uint32_t mask = __vcmpeq4(byte_word, '\n\n\n\n');
        if (mask)
        {
            ssize_t inblock_offset = val.get<0>() << 2;
            min_index = block_offset + inblock_offset + (__ffs(static_cast<int>(mask)) >> 3);
            max_index = block_offset + inblock_offset + ((31 - __clz(static_cast<int>(mask))) >> 3);
            ++count;
        }
        if (__vcmpeq4(byte_word, '\r\r\r\r'))
        {
            present = true;
        }
        return mapped_data(min_index, max_index, count, present);
    }
};

class transform_op_2
{
    ssize_t block_offset; // offset in bytes

public:
    transform_op_2(ssize_t block_offset) : block_offset{block_offset} {}

    __host__
        mapped_data
        operator()(indexed_data val)
    {
        ssize_t min_index = SSIZE_MAX, max_index = -1, count = 0;  // indices are in bytes
        bool present = false;
        uint32_t byte_word = val.get<1>();
        char *c = reinterpret_cast<char *>(&byte_word);
        ssize_t inblock_offset = val.get<0>() << 2;
        for (ssize_t i = 0; i < 4; i++)
        {
            if (c[i] == '\n')
            {
                min_index = std::min(min_index, block_offset + inblock_offset + i);
                max_index = std::max(max_index, block_offset + inblock_offset + i);
                ++count;
            }
            if (c[i] == '\r')
                present = true;
        }
        return mapped_data(min_index, max_index, count, present);
    }
};

struct reduce_op
{
    __host__ __device__
        mapped_data
        operator()(mapped_data val1, mapped_data val2)
    {
        return mapped_data(
            min(val1.get<0>(), val2.get<0>()),
            max(val1.get<1>(), val2.get<1>()),
            val1.get<2>() + val2.get<2>(),
            val1.get<3>() || val2.get<3>());
    }
};

int main()
{
    std::cout << typeid(*rmm::mr::get_current_device_resource()).name() << "\n";

        {

    auto start_total = high_resolution_clock::now();


    ssize_t offset = 0;
    ssize_t size = 270433020;


    auto start = high_resolution_clock::now();

    thrust::device_vector<uint32_t> d_buffer((size + 3) / 4, 0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << "\n";


    start = high_resolution_clock::now();
    const string filepath = "../meta_cudf/test.jsonl";
    auto source = cudf::io::datasource::create(filepath);
    size = source->device_read(static_cast<size_t>(offset), static_cast<size_t>(size), reinterpret_cast<uint8_t *>(d_buffer.data().get()), rmm::cuda_stream_default);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << " Read: " << size << "\n";

    auto index_iter_begin = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(0)), d_buffer.cbegin());
    auto index_iter_end = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(d_buffer.size())), d_buffer.cend());

    start = high_resolution_clock::now();

    auto ret = thrust::transform_reduce(
        rmm::exec_policy(rmm::cuda_stream_default),
        index_iter_begin,
        index_iter_end,
        transform_op(offset),
        thrust::make_tuple((ssize_t)SSIZE_MAX, (ssize_t)-1, (size_t)0, false),
        reduce_op()
    );

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << ret.get<0>() << ' ' << ret.get<1>() << ' '  << ret.get<2>() << ' '  << ret.get<3>() << ' ' << duration.count() << "\n";

    auto stop_total = high_resolution_clock::now();
    auto duration_total = duration_cast<microseconds>(stop_total - start_total);
    std::cout <<  "Duraction tot: " << duration_total.count() << "\n";

    }



        {

    auto start_total = high_resolution_clock::now();


    ssize_t offset = 0;
    ssize_t size = 270433020;


    auto start = high_resolution_clock::now();

    rmm::device_vector<uint32_t> d_buffer((size + 3) / 4, 0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << "\n";


    start = high_resolution_clock::now();
    const string filepath = "../meta_cudf/test.jsonl";
    auto source = cudf::io::datasource::create(filepath);
    size = source->device_read(static_cast<size_t>(offset), static_cast<size_t>(size), reinterpret_cast<uint8_t *>(d_buffer.data().get()), rmm::cuda_stream_default);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << " Read: " << size << "\n";

    auto index_iter_begin = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(0)), d_buffer.cbegin());
    auto index_iter_end = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(d_buffer.size())), d_buffer.cend());

    start = high_resolution_clock::now();

    auto ret = thrust::transform_reduce(
        rmm::exec_policy(rmm::cuda_stream_default),
        index_iter_begin,
        index_iter_end,
        transform_op(offset),
        thrust::make_tuple((ssize_t)SSIZE_MAX, (ssize_t)-1, (size_t)0, false),
        reduce_op()
    );

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << ret.get<0>() << ' ' << ret.get<1>() << ' '  << ret.get<2>() << ' '  << ret.get<3>() << ' ' << duration.count() << "\n";

    auto stop_total = high_resolution_clock::now();
    auto duration_total = duration_cast<microseconds>(stop_total - start_total);
    std::cout <<  "Duraction tot: " << duration_total.count() << "\n";

    }



    rmm::mr::cuda_memory_resource cuda_mr;
    // Construct a resource that uses a coalescing best-fit pool allocator
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
    rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`

    // std::cout << typeid(*rmm::mr::get_current_device_resource()).name() << "\n";
    // return 0;


    std::cout<< std::filesystem::current_path() << '\n';
    {

    auto start_total = high_resolution_clock::now();


    ssize_t offset = 0;
    ssize_t size = 270433020;


    auto start = high_resolution_clock::now();

    rmm::device_vector<uint32_t> d_buffer((size + 3) / 4, 0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << "\n";


    start = high_resolution_clock::now();
    const string filepath = "../meta_cudf/test.jsonl";
    auto source = cudf::io::datasource::create(filepath);
    size = source->device_read(static_cast<size_t>(offset), static_cast<size_t>(size), reinterpret_cast<uint8_t *>(d_buffer.data().get()), rmm::cuda_stream_default);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << " Read: " << size << "\n";

    auto index_iter_begin = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(0)), d_buffer.cbegin());
    auto index_iter_end = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(d_buffer.size())), d_buffer.cend());

    start = high_resolution_clock::now();

    auto ret = thrust::transform_reduce(
        rmm::exec_policy(rmm::cuda_stream_default),
        index_iter_begin,
        index_iter_end,
        transform_op(offset),
        thrust::make_tuple((ssize_t)SSIZE_MAX, (ssize_t)-1, (size_t)0, false),
        reduce_op()
    );

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << ret.get<0>() << ' ' << ret.get<1>() << ' '  << ret.get<2>() << ' '  << ret.get<3>() << ' ' << duration.count() << "\n";

    auto stop_total = high_resolution_clock::now();
    auto duration_total = duration_cast<microseconds>(stop_total - start_total);
    std::cout <<  "Duraction tot: " << duration_total.count() << "\n";

    }



    {

    auto start_total = high_resolution_clock::now();

    ssize_t offset = 0;
    ssize_t size = 1229241;


    auto start = high_resolution_clock::now();

    rmm::device_uvector<uint32_t> d_buffer((size + 3) / 4, rmm::cuda_stream_default);
    thrust::uninitialized_fill(rmm::exec_policy(rmm::cuda_stream_default), d_buffer.begin(), d_buffer.end(), 0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << "\n";


    start = high_resolution_clock::now();
    const string filepath = "../meta_cudf/test.jsonl";
    auto source = cudf::io::datasource::create(filepath);
    size = source->device_read(static_cast<size_t>(offset), static_cast<size_t>(size), reinterpret_cast<uint8_t *>(d_buffer.data()), rmm::cuda_stream_default);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << " Read: " << size << "\n";

    auto index_iter_begin = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(0)), d_buffer.cbegin());
    auto index_iter_end = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(d_buffer.size())), d_buffer.cend());

    start = high_resolution_clock::now();

    auto ret = thrust::transform_reduce(
        rmm::exec_policy(rmm::cuda_stream_default),
        index_iter_begin,
        index_iter_end,
        transform_op(offset),
        thrust::make_tuple((ssize_t)SSIZE_MAX, (ssize_t)-1, (size_t)0, false),
        reduce_op()
    );

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << ret.get<0>() << ' ' << ret.get<1>() << ' '  << ret.get<2>() << ' '  << ret.get<3>() << ' ' << duration.count() << "\n";


    auto stop_total = high_resolution_clock::now();
    auto duration_total = duration_cast<microseconds>(stop_total - start_total);
    std::cout <<  "Duraction tot: " << duration_total.count() << "\n";

    }



    /// CPUU
    {

    auto start_total = high_resolution_clock::now();

    ssize_t offset = 0;
    ssize_t size = 1229241;


    auto start = high_resolution_clock::now();


    thrust::host_vector<uint32_t> h_buffer((size + 3) / 4, 0);
    const string filepath = "../meta_cudf/test.jsonl";
    auto source = cudf::io::datasource::create(filepath);
    size = source->host_read(static_cast<size_t>(offset), static_cast<size_t>(size), reinterpret_cast<uint8_t *>(h_buffer.data()));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);


    std::cout <<  "Duraction: " << duration.count() << " Read: " << size << "\n";

    auto index_iter_begin = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(0)), h_buffer.cbegin());
    auto index_iter_end = thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<ssize_t>(h_buffer.size())), h_buffer.cend());

    start = high_resolution_clock::now();

    auto ret = thrust::transform_reduce(
        index_iter_begin,
        index_iter_end,
        transform_op_2(offset),
        thrust::make_tuple((ssize_t)SSIZE_MAX, (ssize_t)-1, (size_t)0, false),
        reduce_op()
    );

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << ret.get<0>() << ' ' << ret.get<1>() << ' '  << ret.get<2>() << ' '  << ret.get<3>() << ' ' << duration.count() << "\n";


    auto stop_total = high_resolution_clock::now();
    auto duration_total = duration_cast<microseconds>(stop_total - start_total);
    std::cout <<  "Duraction tot: " << duration_total.count() << "\n";

    }


    return 0;
}
