#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/integral.hpp>
#include "meta_math.h"
#include <cstdint>
#include <type_traits>

template<class SizeT>
struct StaticBuffer
{
	using Size = boost::mp11::mp_if<
		boost::mp11::mp_less_equal<SizeT, boost::mp11::mp_int<0>>,
		boost::mp11::mp_int<0>,
		SizeT
		>;
	uint8_t data[Size::value > 0 ? Size::value : 1];
	constexpr static int size = Size::value;

	template<class T>
	__host__ __device__ __forceinline__ T& alias()
	{
		static_assert(sizeof(T) <= Size::value, "Aliased type is greater than size of a static buffer.");
		return reinterpret_cast<T&>(*this);
	}
};

//template<class SizeT>
//struct StaticBuffer_impl<SizeT, typename std::enable_if<boost::mp11::mp_less_equal<SizeT, boost::mp11::mp_int<0>>::value>::type>
//{
//	using Size = boost::mp11::mp_int<0>;
//	uint8_t data[1];
//	constexpr static int size = 0;
//};
//
//template<class SizeT>
//using StaticBuffer = StaticBuffer_impl<SizeT>;

template<int SizeT>
using StaticBuffer_c = StaticBuffer<boost::mp11::mp_int<SizeT>>;