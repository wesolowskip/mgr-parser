//
// Created by pwesolowski on 2/23/23.
//

#include <iostream>
#include "parser.cuh"
#include <filesystem>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <chrono>
#include <cudf/io/json.hpp>
using namespace std::chrono;

using namespace std;

int main() {

    rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
    rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`




    cout << std::filesystem::current_path();
    const char* fname = "../python_binding/test/sample_2000.json";
    int lines = 440000;

    auto start = high_resolution_clock::now();
    auto test = generate_example_metadata(fname, lines);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout <<  "Duraction: " << duration.count() << "\n";


//    auto test = cudf::io::read_json()

    cout << "Heloo world";
    return 0;
}