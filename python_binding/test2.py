from timeit import default_timer as timer

import rmm
rmm.reinitialize(pool_allocator=True)
from time import sleep

sleep(2)

import cudf
import metajsonparser as mp

start = timer()
mp.read_json("../meta_cudf/test.jsonl", 440000).head()
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282

start = timer()
mp.read_json("../meta_cudf/test.jsonl", 440000).head()
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282
start = timer()
mp.read_json("../meta_cudf/test.jsonl", 440000).head()
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282
start = timer()
mp.read_json("../meta_cudf/test.jsonl", 440000).head()
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282
