//
// Created by pwesolowski on 2/23/23.
//

#include <iostream>
#include "parser.cuh"
#include <filesystem>

using namespace std;

int main() {
    cout << std::filesystem::current_path();
    const char* fname = "../python_binding/test/sample_2000.json";
    int lines = 2000;
    auto test = generate_example_metadata(fname, lines);
    cout << "Heloo world";
    return 0;
}