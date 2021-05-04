#pragma once
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/config.h>
#include <meta_json_parser/parsing_error.h>
#include <meta_json_parser/intelisense_silencer.h>
#include <meta_json_parser/kernel_context.cuh>
#include <meta_json_parser/meta_memory_manager.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/kernel_launcher.cuh>
#include <cstdint>
#include <type_traits>

template<class PC, class BA>
using __NewM3 = MetaMemoryManager<
	ParserConfiguration<
		typename PC::RuntimeConfiguration,
		ExtendRequests<
			typename PC::MemoryConfiguration,
			typename BA::MemoryRequests
		>
	>
>;

template<class ParserConfigurationT, class BaseActionT>
__global__ void __launch_bounds__(1024, 2)
_parser_kernel(
	typename __NewM3<ParserConfigurationT, BaseActionT>::ReadOnlyBuffer* readOnlyBuffer,
	const char* input,
	const InputIndex* indices,
	ParsingError* err,
	void** output,
	const uint32_t count);

template<class ParserConfigurationT, class BaseActionT>
struct ParserKernel
{
	using OC = OutputConfiguration<typename BaseActionT::OutputRequests>;
	using MC = ExtendRequests<typename ParserConfigurationT::MemoryConfiguration, typename BaseActionT::MemoryRequests>;
	using PC = ParserConfiguration<
		typename ParserConfigurationT::RuntimeConfiguration,
		MC
	>;
	using M3 = MetaMemoryManager<PC>;
  	using ROB = typename M3::ReadOnlyBuffer;
	static_assert(std::is_same_v<M3, __NewM3<ParserConfigurationT, BaseActionT>>, "__NewM3 inconsistent with implementation of ParserKernel.");
	using RT = typename PC::RuntimeConfiguration;
	using KC = KernelContext<PC, OC>;
	using Launcher = KernelLauncherFixedResources<
		typename RT::BlockDimX,
		typename RT::BlockDimY,
		typename RT::BlockDimZ,
		boost::mp11::mp_int<0>,
		typename __NewM3<ParserConfigurationT, BaseActionT>::ReadOnlyBuffer*,
		const char*,
		const InputIndex*,
		ParsingError*,
		void**,
		const uint32_t
	>;

	ROB* m_d_rob;

	ParserKernel(cudaStream_t stream = 0)
	{
		cudaMalloc(&m_d_rob, sizeof(ROB));
		ROB rob;
		M3::FillReadOnlyBuffer(rob);
		cudaMemcpyAsync(m_d_rob, &rob, sizeof(ROB), cudaMemcpyHostToDevice, stream);
		//Wait for copying of "rob"
		cudaStreamSynchronize(stream);
	}

	~ParserKernel()
	{
		cudaFree(m_d_rob);
	}

	static thrust::host_vector<uint64_t> OutputSizes()
	{
		thrust::host_vector<uint64_t> result;
		boost::mp11::mp_for_each<typename OC::RequestList>([&](auto i){
			using Request = decltype(i);
			result.push_back(sizeof(typename Request::OutputType));
		});
		return std::move(result);
	}
};

template<class ParserConfigurationT, class BaseActionT>
	/// <summary>
	/// Main kernel responsible for parsing json.
	/// </summary>
	/// <param name="block_data">Read-only data stored in shared memory.</param>
	/// <param name="input">Pointer to input bytes array.</param>
	/// <param name="indices">Pointer to an array of indices of object beginings. Requires guard at the end! indices[count] == length(input)</param>
	/// <param name="err">Output array for error codes.</param>
	/// <param name="output">Pointer to array of pointers. Each points to distinct output array.</param>
	/// <param name="count">Number of objects.</param>
	/// <returns></returns>
__global__ void __launch_bounds__(1024, 2)
	_parser_kernel(
		typename __NewM3<ParserConfigurationT, BaseActionT>::ReadOnlyBuffer* readOnlyBuffer,
		const char* input,
		const InputIndex* indices,
		ParsingError* err,
		void** output,
		const uint32_t count
	)
{
	using PK = ParserKernel<ParserConfigurationT, BaseActionT>;
	using KC = typename PK::KC;
	using RT = typename PK::RT;
	__shared__ typename PK::M3::SharedBuffers sharedBuffers;
	KC kc(readOnlyBuffer, sharedBuffers, input, indices, output);
	if (RT::InputId() >= count)
		return;
	ParsingError e = BaseActionT::Invoke(kc);
	if (RT::WorkerId() == 0)
		err[RT::InputId()] = e;
}

