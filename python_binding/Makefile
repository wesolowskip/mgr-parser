LIB_DIR = .
PATH:=/opt/conda/envs/rapids/bin/:${PATH}

.PHONY: metajsonparser clean run_test

default: metajsonparser

metajsonparser: setup.py metajsonparser/_lib/*.pyx metajsonparser/*.py
	python3 -m pip install cython
	python3 setup.py build_ext --inplace && rm -f metajsonparser/_lib/json_cudf.cpp && rm -Rf build

clean:
	rm metajsonparser/_lib/*.so

run_test:
	LIBCUDF_CUFILE_POLICY=GDS python3 test/test.py test/sample_2000.json 2000
