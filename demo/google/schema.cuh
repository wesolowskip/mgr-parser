
// KEYS
using K_L1_user_id = metastring("user_id");
using K_L1_gmap_id = metastring("gmap_id");
using K_L1_rating = metastring("rating");
using K_L1_category = metastring("category");
using K_L1_latitude = metastring("latitude");
using K_L1_longitude = metastring("longitude");

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
