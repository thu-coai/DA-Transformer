// ##########################################################################
// Copyright (C) 2022 COAI @ Tsinghua University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//         http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ###########################################################################

// Parts of the codes are modified from https://github.com/Oneflow-Inc/oneflow/blob/ed8d26bad3764145a24a54e7e767765aaa87d7f8/oneflow/core/cuda/softmax.cuh
// The used codes are also licensed under Apache 2.0 with the copyright of the OneFlow Authors

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cassert>
#include <tuple>
#include <type_traits>

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <math_constants.h>

#include <torch/extension.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.type().is_cpu(), #x " must be a CPU tensor")

#define MY_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)        \
  case enum_type: {                                                              \
    using HINT = type;                                                           \
    return __VA_ARGS__();                                                        \
  }

#define MY_DISPATCH_FLOATING_TYPES_AND_HALF_WITH_HINT(TYPE, NAME, HINT, ...)         \
  [&] {                                                                        \
    const auto& the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    switch (_st) {                                                             \
      MY_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Double, double, HINT, __VA_ARGS__)  \
      MY_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Float, float, HINT, __VA_ARGS__)    \
      MY_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Half, at::Half, HINT, __VA_ARGS__)  \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");         \
    }                                                                          \
  }()

#define MY_PRIVATE_VALUE(val, HINT, ...)                                        \
  case val: {                                                                   \
    const int HINT = val;                                                           \
    return __VA_ARGS__();                                                       \
  }

#define MY_DISPATCH_VALUE(VAL, NAME, HINT, ...)                                \
  [&] {                                                                        \
    switch (VAL) {                                                             \
	  MY_PRIVATE_VALUE(1, HINT, __VA_ARGS__)                                   \
      MY_PRIVATE_VALUE(2, HINT, __VA_ARGS__)                                   \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for this value");                    \
    }                                                                          \
  }()

#define MY_DISPATCH_BOOL(VAL, NAME, HINT, ...)                                 \
  [&] {                                                                        \
    if (VAL) {                                                                 \
		const bool HINT = true;                                                \
    	return __VA_ARGS__();                                                  \
	}else{                                                                     \
		const bool HINT = false;                                               \
		return __VA_ARGS__();                                                  \
	}                                                                          \
  }()


template<typename T>
struct SumOp {
	__device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
	__device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
	typedef cub::BlockReduce<T, block_size> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ T result_broadcast;
	T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
	if (threadIdx.x == 0) { result_broadcast = result; }
	__syncthreads();
	return result_broadcast;
}

template<typename T>
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
	return CUDART_INF_F;
}

template<>
__inline__ __device__ double Inf<double>() {
	return CUDART_INF;
}

template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
	return __expf(x);
#else
	return exp(x);
#endif
}

template<>
__inline__ __device__ double Exp<double>(double x) {
	return exp(x);
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
	return __fdividef(a, b);
#else
	return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
	return a / b;
}

template<typename T>
__inline__ __device__ T Log(T x);

template<>
__inline__ __device__ float Log<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
	return __logf(x);
#else
	return log(x);
#endif
}
template<>
__inline__ __device__ double Log<double>(double x) {
	return log(x);
}

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
																int* num_blocks) {
	int dev;
	{
		cudaError_t err = cudaGetDevice(&dev);
		if (err != cudaSuccess) { return err; }
	}
	int sm_count;
	{
		cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
		if (err != cudaSuccess) { return err; }
	}
	int tpm;
	{
		cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
		if (err != cudaSuccess) { return err; }
	}
	*num_blocks =
			std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
	return cudaSuccess;
}

template<typename T>
struct DefaultComputeType{
	using type = T;
};

template<>
struct DefaultComputeType<at::Half> {
	using type = float;
};


template<typename T, int N>
struct GetPackType {
	using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
	static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
	__device__ Pack() {
		// do nothing
	}
	PackType<T, N> storage;
	T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad {
	DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
	template<int N>
	__device__ void load(DST* dst, int64_t row, int64_t col) const {
		Pack<SRC, N> pack;
		const int64_t offset = (row * row_size + col) / N;
		pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
		for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
	}
	const SRC* src;
	int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
	DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
	template<int N>
	__device__ void store(const SRC* src, int64_t row, int64_t col) {
		Pack<DST, N> pack;
		const int64_t offset = (row * row_size + col) / N;
#pragma unroll
		for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
		*(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
	}
	DST* dst;
	int64_t row_size;
};




template<typename scalar_t, typename LOAD, typename STORE,
			typename ComputeType, int pack_size, int block_size, bool require_grad, typename Accessor, typename Accessor2, typename Accessor3>
__global__ void logsoftmax_gather_kernel(
		LOAD load, STORE store, int64_t rows, int64_t cols,
		Accessor word_ins_out, Accessor2 selected_result, Accessor3 select_idx,
		int bsz, int prelen, int slen, int vocabsize)
{
	const int tid = threadIdx.x;
	// assert(cols % pack_size == 0);
	static_assert(pack_size == 1, "pack_size should not be 1");
	const int num_packs = cols / pack_size;

	for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
		ComputeType thread_max = -Inf<ComputeType>();
		for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
			ComputeType pack[pack_size];
			load.template load<pack_size>(pack, row, pack_id * pack_size);

			#pragma unroll
			for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
		}

		const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
		ComputeType thread_sum = 0;
		for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
			ComputeType pack[pack_size];
			load.template load<pack_size>(pack, row, pack_id * pack_size);

			#pragma unroll
			for (int i = 0; i < pack_size; ++i) { thread_sum += Exp(pack[i] - row_max); }
		}

		const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
		int batch_id = row / prelen;
		int prepos = row % prelen;
		for(int sid = tid; sid < slen; sid += block_size){
			int64_t target_idx = select_idx[batch_id][prepos][sid];
			selected_result[batch_id][prepos][sid] = (static_cast<ComputeType>(word_ins_out[batch_id][prepos][target_idx]) - row_max) - Log(row_sum);
		}

		if (require_grad){
			__syncthreads();
			for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
				ComputeType pack[pack_size];
				load.template load<pack_size>(pack, row, pack_id * pack_size);

				#pragma unroll
				for (int i = 0; i < pack_size; ++i) { pack[i] = Div(Exp(pack[i] - row_max), row_sum); }

				store.template store<pack_size>(pack, row, pack_id * pack_size);
			}
		}

	}
}


torch::Tensor logsoftmax_gather(torch::Tensor word_ins_out, const torch::Tensor &select_idx, bool require_gradient)
{
	CHECK_CUDA(word_ins_out);  // bsz * prelen * vocabsize
	CHECK_CUDA(select_idx);  // bsz * prelen * slen
	TORCH_CHECK(word_ins_out.dim() == 3, "word_ins_out dim != 3");
	TORCH_CHECK(select_idx.dim() == 3, "select_idx dim != 3");

    auto bsz = word_ins_out.size(0);
    auto prelen = word_ins_out.size(1);
    auto vocabsize = word_ins_out.size(2);
	auto slen = select_idx.size(2);
    TORCH_CHECK(select_idx.size(0) == bsz, "batch size not match");
	TORCH_CHECK(select_idx.size(1) == prelen, "prelen size not match");
    TORCH_CHECK(select_idx.scalar_type() == at::kLong, "select_idx should be long");
	TORCH_CHECK(word_ins_out.is_contiguous(), "word_ins_out is not contiguous");

	constexpr int block_size = 1024;
	constexpr int waves = 32;
	int grid_dim_x;
	{
		cudaError_t err = GetNumBlocks(block_size, bsz * prelen, waves, &grid_dim_x);
		assert(err == cudaSuccess);
	}

	torch::Tensor selected_result;
	cudaStream_t stream = 0;

	MY_DISPATCH_FLOATING_TYPES_AND_HALF_WITH_HINT(
		word_ins_out.scalar_type(), "logsoftmax_gather_kernel_scalar_t", scalar_t, [&] {
			using ComputeType = typename DefaultComputeType<scalar_t>::type;
			if (std::is_same<ComputeType, float>::value){
				selected_result = at::zeros({bsz, prelen, slen}, word_ins_out.options().dtype(at::kFloat));
			}else{
				selected_result = at::zeros({bsz, prelen, slen}, word_ins_out.options().dtype(at::kDouble));
			}
			using Load = DirectLoad<scalar_t, ComputeType>;
			using Store = DirectStore<ComputeType, scalar_t>;
			Load load(word_ins_out.data_ptr<scalar_t>(), vocabsize);
			Store store(word_ins_out.data_ptr<scalar_t>(), vocabsize);
			int64_t cols = vocabsize;
			int64_t rows = bsz * prelen;
			const int PackSize = 1;
			// MY_DISPATCH_VALUE(
			// 	pack_size, "GatherVocabLogitsKernel_pack_size", PackSize, [&]{
					MY_DISPATCH_BOOL(
						require_gradient, "logsoftmax_gather_kernel_require_gradient", RequireGrad, [&]{
							logsoftmax_gather_kernel<scalar_t, Load, Store, ComputeType, PackSize, block_size, RequireGrad>
								<<<grid_dim_x, block_size, 0, stream>>>
								(
									load, store, rows, cols,
									word_ins_out.packed_accessor64<scalar_t, 3>(),
									selected_result.packed_accessor64<ComputeType, 3>(),
									select_idx.packed_accessor64<int64_t, 3>(),
									bsz, prelen, slen, vocabsize
								);
							assert(cudaPeekAtLastError() == cudaSuccess);
						}
					);
			// 	}
			// );
		}
	);

	return selected_result;
}
