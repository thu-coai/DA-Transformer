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

#include <torch/extension.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.type().is_cpu(), #x " must be a CPU tensor")

#define BLOCK_BUCKET 16

// static float fninf = -std::numeric_limits<float>::infinity();

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_maxalpha_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue,
    Accessor1 alpha, Accessor2 trace,
    Accessor1 match_all,
    Accessor1 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen, int translen, int n_seg)
{
    int bucket_idx = blockIdx.y % BLOCK_BUCKET;
    __shared__ volatile int bucket_no;

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;
    if (main_thread){
        // obtain task id
        bucket_no = atomicAdd((int*)bucket_queue + bucket_idx, 1);
    }
    __syncthreads();

    int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int pos = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;

    int output_len = output_length[batch_id];
    int target_len = target_length[batch_id];

    CUDA_KERNEL_ASSERT(target_len >= 2 && output_len >= 2 && "dag_best_alignment: target/output length should at least 2");
    CUDA_KERNEL_ASSERT(output_len >= target_len && "dag_best_alignment: graph size is too small (smaller than target length)");
    CUDA_KERNEL_ASSERT((target_len - 1) * translen + 1 >= output_len && "dag_best_alignment: target length is too short or graph size is too large. \
        Please increase max_transition_length or remove samples that are too short");

    // t = 0
    {
        if(pos >= output_len) return;
        if(main_thread){
            alpha[batch_id][0][0] = match_all[batch_id][0][0];
        }
        __threadfence();
        __syncthreads();
        if(main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
    }

    for(int t = 1; t < target_len; t++){
        if(pos + t >= output_len) return;
        if (main_thread && seg_id != 0){
            while(accomplish_queue[batch_id * n_seg + seg_id - 1] < t); // wait for previous segment to accomplish
        }
        __syncthreads();
        // if(main_thread && blockIdx.x == 0 && blockIdx.y == 0){
        //     printf("alpha %d\n", t);
        // }

        scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
        int maxidx = -1;

        int nowpos = pos + t;
        int maxdelta = min(nowpos, translen);
        for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
            int lastpos = nowpos - delta;
            scalar_t nextval = alpha[batch_id][t - 1][lastpos] + links[batch_id][lastpos][delta - 1];
            if(nextval > maxval) {maxval = nextval; maxidx = lastpos;}
        }
        unsigned shfl_mask = __activemask();
        #pragma unroll
        for(int delta = TRANS_BLOCK_SIZE >> 1; delta > 0; delta >>= 1){
            scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, delta, TRANS_BLOCK_SIZE);
            scalar_t nextmaxidx = __shfl_down_sync(shfl_mask, maxidx, delta, TRANS_BLOCK_SIZE);
            if(nextval > maxval) { maxval = nextval; maxidx = nextmaxidx; }
        }
        
        if(threadIdx.x == 0){
            alpha[batch_id][t][nowpos] = maxval + match_all[batch_id][t][nowpos];
            trace[batch_id][t][nowpos] = maxidx;

            if(t == target_len-1 && nowpos == output_len-1){
                CUDA_KERNEL_ASSERT(alpha[batch_id][t][nowpos] > -std::numeric_limits<scalar_t>::infinity() && "dag_best_alignment: no valid path")
            }
            // printf("%d %d %d %d: alpha[%d][%d][%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, batch_id, t, nowpos, maxval);
        }

        __threadfence();
        __syncthreads();
        if (main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
        //__syncthreads();
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_maxalpha(cudaStream_t stream, torch::Tensor &alpha, torch::Tensor &trace, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int translen)
{
    int n_seg = (prelen - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);

    int *bucket_queue, *accomplish_queue;
    // bucket_queue = QueueAllocator<0>::allo((BLOCK_BUCKET + bsz * n_seg) * sizeof(int));
    // cudaMemsetAsync(bucket_queue, 0, (BLOCK_BUCKET + bsz * n_seg) * sizeof(int), stream);
    auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz * n_seg}, match_all.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    accomplish_queue = bucket_queue + BLOCK_BUCKET;
    // printf("alpha start\n");
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_alpha", [&] {
            alpha.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_maxalpha_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue,
                alpha.packed_accessor64<scalar_t, 3>(),
                trace.packed_accessor64<int32_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, translen, n_seg
            );
        }
    );
    // cudaDeviceSynchronize();
    // printf("alpha end\n");
}


template<int BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_backtrace_kernel(
    Accessor1 path,
    Accessor2 trace,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen)
{
    int batch_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if(batch_id >= bsz) return;
    int nowpos = output_length[batch_id] - 1;
    for(int i = target_length[batch_id] - 1; i >= 0; i--){
        path[batch_id][nowpos] = i;
        nowpos = trace[batch_id][i][nowpos];
    }
}


template<int BLOCK_SIZE>
void invoke_calculate_backtrace(cudaStream_t stream, torch::Tensor &path, const torch::Tensor &trace, const torch::Tensor &output_length, const torch::Tensor &target_length,
        int bsz, int prelen, int tarlen)
{
    int n_block = (bsz - 1) / BLOCK_SIZE + 1;
    dim3 dimGrid(n_block);
    dim3 dimBlock(BLOCK_SIZE);

    path.fill_(-1);
    calculate_backtrace_kernel<BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
        path.packed_accessor64<int32_t, 2>(),
        trace.packed_accessor64<int32_t, 3>(),
        output_length.packed_accessor64<int64_t, 1>(),
        target_length.packed_accessor64<int64_t, 1>(),
        bsz, prelen, tarlen
    );
    // cudaDeviceSynchronize();
    // printf("alpha end\n");
}


std::tuple<torch::Tensor, torch::Tensor> dag_best_alignment(const torch::Tensor &match_all, const torch::Tensor &links, 
    const torch::Tensor &output_length, const torch::Tensor &target_length, int config)
{
    CHECK_CUDA(match_all);  // bsz * tarlen * prelen
    CHECK_CUDA(links);   // bsz * prelen * translen
    CHECK_CUDA(output_length); // bsz
    CHECK_CUDA(target_length); // bsz
    TORCH_CHECK(match_all.dim() == 3, "match_all dim != 3");
    TORCH_CHECK(links.dim() == 3, "links dim != 3");
    TORCH_CHECK(output_length.dim() == 1, "output_length dim != 3");
    TORCH_CHECK(target_length.dim() == 1, "target_length dim != 3");

    auto bsz = match_all.size(0);
    auto prelen = match_all.size(2);
    auto tarlen = match_all.size(1);
    auto translen = links.size(2);
    TORCH_CHECK(links.size(0) == bsz && output_length.size(0) == bsz && target_length.size(0) == bsz, "batch size not match");
    TORCH_CHECK(links.size(1) == prelen, "prelen not match");
    TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

    cudaStream_t current_stream = 0;
    // printf("alpha0\n");

    // calculate alpha
    // printf("%d %d %d\n", bsz, tarlen, prelen);
    auto alpha = at::zeros({bsz, tarlen, prelen}, match_all.options());
    auto trace = at::zeros({bsz, tarlen, prelen}, match_all.options().dtype(at::kInt));
    auto path = at::zeros({bsz, prelen}, match_all.options().dtype(at::kInt));


    // printf("alpha1\n");
    // printf("%d %d %d\n", alpha.size(0), alpha.size(1), alpha.size(2));
    // printf("%f\n", alpha[0][0][0]);
    switch(config){ //DEBUG
        case 1: invoke_calculate_maxalpha<4, 256>(current_stream, alpha, trace, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 2: invoke_calculate_maxalpha<8, 128>(current_stream, alpha, trace, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 3: invoke_calculate_maxalpha<16, 64>(current_stream, alpha, trace, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 4: invoke_calculate_maxalpha<32, 32>(current_stream, alpha, trace, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }

    invoke_calculate_backtrace<512>(current_stream, path, trace, output_length, target_length, bsz, prelen, tarlen);

    return std::make_tuple(alpha, path);
}