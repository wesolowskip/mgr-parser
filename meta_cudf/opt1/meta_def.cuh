#ifndef META_CUDF_META_DEF_CUH
#define META_CUDF_META_DEF_CUH

// INCLUDES
#include <boost/mp11.hpp>

#include <meta_json_parser/mp_string.h>
#include <meta_json_parser/meta_utility/metastring.h>

// TYPE-DEPENDENT INCLUDES
#include <meta_json_parser/action/jdict.cuh>
#include <meta_json_parser/action/jnumber.cuh>
#include <meta_json_parser/action/jstring.cuh>
#include <meta_json_parser/action/jstring_custom.cuh>
#include <meta_json_parser/action/string_transform_functors/polynomial_rolling_hash.cuh>
#include <meta_json_parser/action/jrealnumber.cuh>
//#include <meta_json_parser/action/datetime/jdatetime.cuh>
#include <meta_json_parser/action/jbool.cuh>

//#include <meta_json_parser/action/jstring_custom.cuh>
//#include <meta_json_parser/action/string_transform_functors/polynomial_rolling_hash_matcher.cuh>
#include <meta_json_parser/action/decorators/null_default_value.cuh>
#include <meta_json_parser/action/string_functors/letter_case.cuh>

using namespace boost::mp11;
using namespace std;

// SETTINGS
using WorkGroupSize = mp_int<32>;


// KEYS
using K_L1_user_id = metastring("user_id");
using K_L1_gmap_id = metastring("gmap_id");
using K_L1_rating = metastring("rating");
using K_L1_category = metastring("category");
using K_L1_latitude = metastring("latitude");
using K_L1_longitude = metastring("longitude");

// CONFIGURE STRING PARSING
#pragma message("Always using JStringStaticCopy for parsing strings")
// NOTE: dynamic string size are dynamic configurable, but not per field
template<class Key, int Size, class Options = boost::mp11::mp_list<>>
using JStringVariant = JStringStaticCopy<mp_int<Size>, Key, Options>;


// DICT
template<template<class, int> class StringFun, class DictOpts>
using DictCreator = JDict < mp_list <
        mp_list<K_L1_user_id, JStringVariant<K_L1_user_id, 21>>,
mp_list<K_L1_gmap_id, JStringVariant<K_L1_gmap_id, 37>>,
mp_list<K_L1_rating, JNumber<uint8_t, K_L1_rating>>,
mp_list<K_L1_category, JStringVariant<K_L1_category, 470>>,
mp_list<K_L1_latitude, JRealNumber<float, K_L1_latitude>>,
mp_list<K_L1_longitude, JRealNumber<float, K_L1_longitude>>>,
DictOpts>;


// NOTE: Neither PARSER OPTIONS nor PARSER are needed for 'data_def.cuh'
// that is for inclusion in the 'benchmark/main.cu'
#ifndef BENCHMARK_MAIN_CU

// PARSER OPTIONS
template<class Key, int Size>
using StaticCopyFun = JStringStaticCopy<mp_int<Size>, Key>;

// PARSER
using BaseAction = DictCreator<StaticCopyFun, mp_list<>>;
#endif /* !defined(BENCHMARK_MAIN_CU) */

#endif /* !defined(META_CUDF_META_DEF_CUH) */
