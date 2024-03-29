#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>
#include <meta_json_parser/output_manager.cuh>
#include <meta_json_parser/json_parse.cuh>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>

struct VoidAction
{
	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		return ParsingError::None;
	}
};

template<typename ...RequirementsT>
struct VoidActionRequirements
{
	using ParserRequirements = boost::mp11::mp_list<RequirementsT...>;

	template<class KernelContextT>
	static __device__ INLINE_METHOD ParsingError Invoke(KernelContextT& kc)
	{
		return ParsingError::None;
	}
};
