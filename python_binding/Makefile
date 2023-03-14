LIB_DIR = .
PATH:=/opt/conda/envs/rapids/bin/:${PATH}

default: metajsonparser

metajsonparser: setup.py *.pyx
	python3 -m pip install cython
	LIBRARY_PATH=/root/meta-json-parser/cmake-build-debug:${LIBRARY_PATH} CPLUS_INCLUDE_PATH=/root/meta-json-parser/meta_cudf:${CPLUS_INCLUDE_PATH} python3 setup.py build_ext --inplace && rm -f metajsonparser.c && rm -Rf build

clean:
	rm *.so

run_test:
	LD_LIBRARY_PATH=../cmake-build-debug:. PYTHONPATH=. LIBCUDF_CUFILE_POLICY=GDS python3 test/test.py test/sample_2000.json 2000
