from timeit import default_timer as timer

import rmm
rmm.reinitialize(pool_allocator=True)

import cudf
import metajsonparser as mp

start = timer()
cudf.read_json("../meta_cudf/test.jsonl", lines=True).head()
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282
