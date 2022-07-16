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
#include "utilities.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.type().is_cpu(), #x " must be a CPU tensor")

#define BLOCK_BUCKET 16

// static float fninf = -std::numeric_limits<float>::infinity();

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2>
__global__ void calculate_alpha_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue,
    Accessor1 alpha,
    Accessor1 match_all,
    Accessor1 links,
    Accessor2 output_length,
    Accessor2 target_length,
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
        int nowpos = pos + t;
        int maxdelta = min(nowpos, translen);
        for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
            int lastpos = nowpos - delta;
            scalar_t nextval = alpha[batch_id][t - 1][lastpos] + links[batch_id][lastpos][delta - 1];
            if(nextval > maxval) maxval = nextval;
        }
        unsigned shfl_mask = __activemask();
        if_constexpr (TRANS_BLOCK_SIZE > 16) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 8) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 4) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 2) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        if_constexpr (TRANS_BLOCK_SIZE > 1) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
        maxval = __shfl_sync(shfl_mask, maxval, 0, TRANS_BLOCK_SIZE);
        // if(t == 1 && threadIdx.y == 1) printf("%d %d %d %d: aft1_maxval = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, maxval);
        
        shfl_mask = __ballot_sync(shfl_mask, !isinf(maxval));
        float res;
        if (isinf(maxval)){
            res = maxval;
        }else{
            scalar_t sumval = 0;
            for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
                int lastpos = nowpos - delta;
                sumval += exp(alpha[batch_id][t - 1][lastpos] + links[batch_id][lastpos][delta - 1] - maxval);
            }
            if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
            if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
            res = log(sumval) + maxval + match_all[batch_id][t][nowpos];
        }
        if(threadIdx.x == 0){
            alpha[batch_id][t][nowpos] = res;
            // printf("%d %d %d %d: alpha[%d][%d][%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, batch_id, t, nowpos, res);
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
void invoke_calculate_alpha(cudaStream_t stream, torch::Tensor &alpha, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
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
            calculate_alpha_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue,
                alpha.packed_accessor64<scalar_t, 3>(),
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

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2>
__global__ void calculate_beta_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue,
    Accessor1 beta,
    Accessor1 match_all,
    Accessor1 links,
    Accessor2 output_length,
    Accessor2 target_length,
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
    int pos = (n_seg - 1 - seg_id) * SEQ_BLOCK_SIZE + threadIdx.y;

    int output_len = output_length[batch_id];
    int target_len = target_length[batch_id];

    // t = target_len - 1
    {
        int nowpos = pos + target_len - 1;
        if(nowpos == output_len - 1){
            beta[batch_id][target_len - 1][nowpos] = match_all[batch_id][target_len - 1][nowpos];
        }
        __threadfence();
        __syncthreads();
        if(main_thread){
            atomicAdd((int*)accomplish_queue + batch_id * n_seg + seg_id, 1);
        }
    }

    for(int t = target_len - 2; t >= 0; t--){
        if (main_thread && seg_id != 0){
            while(accomplish_queue[batch_id * n_seg + seg_id - 1] < target_len - 1 - t); // wait for previous segment to accomplish
        }
        __syncthreads();
        // if(main_thread && blockIdx.x == 0 && blockIdx.y == 0){
        //     printf("beta: %d\n", t);
        // }

        int nowpos = pos + t;
        if(nowpos < output_len){
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
            int maxdelta = min(output_len - 1 - nowpos, translen);
            for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
                int lastpos = nowpos + delta;
                scalar_t nextval = beta[batch_id][t + 1][lastpos] + links[batch_id][nowpos][delta - 1];
                if(nextval > maxval) maxval = nextval;
            }
            unsigned shfl_mask = __activemask();
            if_constexpr (TRANS_BLOCK_SIZE > 16) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 8) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 4) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 2) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            if_constexpr (TRANS_BLOCK_SIZE > 1) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
            maxval = __shfl_sync(shfl_mask, maxval, 0, TRANS_BLOCK_SIZE);
            
            shfl_mask = __ballot_sync(shfl_mask, !isinf(maxval));        
            float res;
            if (isinf(maxval)){
                res = maxval;
            }else{
                scalar_t sumval = 0;
                for(int delta = threadIdx.x + 1; delta <= maxdelta; delta += TRANS_BLOCK_SIZE){
                    int lastpos = nowpos + delta;
                    sumval += exp(beta[batch_id][t + 1][lastpos] + links[batch_id][nowpos][delta - 1] - maxval);
                }
                if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                res = log(sumval) + maxval + match_all[batch_id][t][nowpos];
            }
            if(threadIdx.x == 0){
                beta[batch_id][t][nowpos] = res;
            }
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
void invoke_calculate_beta(cudaStream_t stream, torch::Tensor &beta, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int translen)
{
    int n_seg = (prelen - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);

    int *bucket_queue, *accomplish_queue;
    // bucket_queue = QueueAllocator<1>::allo((BLOCK_BUCKET + bsz * n_seg) * sizeof(int));
    // cudaMemsetAsync(bucket_queue, 0, (BLOCK_BUCKET + bsz * n_seg) * sizeof(int), stream);
    auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz * n_seg}, match_all.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    accomplish_queue = bucket_queue + BLOCK_BUCKET;
    // printf("beta start\n");
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_beta", [&] {
            beta.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_beta_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue,
                beta.packed_accessor64<scalar_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, translen, n_seg
            );
        }
    );
    // cudaDeviceSynchronize();
    // printf("beta end\n");
}


std::tuple<torch::Tensor, torch::Tensor> dag_loss(const torch::Tensor &match_all, const torch::Tensor &links, 
    const torch::Tensor &output_length, const torch::Tensor &target_length, bool require_gradient,
    int config)
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
    torch::Tensor beta = at::zeros({bsz, tarlen, prelen}, match_all.options());


    // printf("alpha1\n");
    // printf("%d %d %d\n", alpha.size(0), alpha.size(1), alpha.size(2));
    // printf("%f\n", alpha[0][0][0]);
    switch(config){
        case 1: invoke_calculate_alpha<4, 256>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 2: invoke_calculate_alpha<8, 128>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 3: invoke_calculate_alpha<16, 64>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 4: invoke_calculate_alpha<32, 32>(current_stream, alpha, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }

    // calculate beta
    if(require_gradient){
        cudaStream_t new_stream;
        cudaEvent_t cuda_event;
        cudaStreamCreate(&new_stream);
        cudaEventCreate(&cuda_event);
        switch(config){
            case 1: invoke_calculate_beta<4, 256>(new_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            case 2: invoke_calculate_beta<8, 128>(new_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            case 3: invoke_calculate_beta<16, 64>(new_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            case 4: invoke_calculate_beta<32, 32>(new_stream, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
            default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
        }
        cudaEventRecord(cuda_event, new_stream); // new_stream triggers cuda_event
        cudaStreamWaitEvent(current_stream, cuda_event, 0); // current stream waits for event
        cudaEventDestroy(cuda_event);
        cudaStreamDestroy(new_stream);
    }

    // printf("alpha4\n");
    return std::make_tuple(alpha, beta);
}


template<class scalar_t, int TAR_BLOCK_SIZE, int PRE_BLOCK_SIZE, int SWEEP_SIZE, class Accessor1, class Accessor2>
__global__ void calculate_grad_match_all_kernel(
    Accessor1 grad_match_all,
    Accessor2 grad_output,
    Accessor1 alpha,
    Accessor1 beta,
    Accessor1 match_all,
    int prelen, int tarlen)
{
    int y = blockIdx.y * TAR_BLOCK_SIZE + threadIdx.y;
    int batch_id = blockIdx.z;

    if(y >= tarlen) return;
    #pragma unroll
    for(int i = 0; i < SWEEP_SIZE; i++){
        int x = blockIdx.x * PRE_BLOCK_SIZE * SWEEP_SIZE + threadIdx.x + PRE_BLOCK_SIZE * i;
        if(x >= prelen)  continue;
        if(isinf(match_all[batch_id][y][x]) || isinf(beta[batch_id][0][0])){
            grad_match_all[batch_id][y][x] = 0;
        }else{
            grad_match_all[batch_id][y][x] = exp(alpha[batch_id][y][x] + beta[batch_id][y][x] - match_all[batch_id][y][x] - beta[batch_id][0][0]) * grad_output[batch_id];
        }
    }
}

template<int TAR_BLOCK_SIZE, int PRE_BLOCK_SIZE, int SWEEP_SIZE>
void invoke_calculate_grad_match_all(cudaStream_t stream, torch::Tensor &grad_match_all, const torch::Tensor &grad_output,
    const torch::Tensor &alpha, const torch::Tensor &beta, const torch::Tensor &match_all, const torch::Tensor &links,
    const torch::Tensor &output_length, const torch::Tensor &target_length,
    int bsz, int prelen, int tarlen, int translen)
{
    int n_tar = (tarlen - 1) / TAR_BLOCK_SIZE + 1;
    int n_pre = (prelen - 1) / PRE_BLOCK_SIZE / SWEEP_SIZE + 1;
    dim3 dimGrid(n_pre, n_tar, bsz);
    dim3 dimBlock(PRE_BLOCK_SIZE, TAR_BLOCK_SIZE, 1);

    // printf("grad_match_all start\n");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_grad_match_all", [&] {
            calculate_grad_match_all_kernel<scalar_t, TAR_BLOCK_SIZE, PRE_BLOCK_SIZE, SWEEP_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                grad_match_all.packed_accessor64<scalar_t, 3>(),
                grad_output.packed_accessor64<scalar_t, 1>(),
                alpha.packed_accessor64<scalar_t, 3>(),
                beta.packed_accessor64<scalar_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                prelen, tarlen
            );
        }
    );
    // cudaDeviceSynchronize();
    // printf("grad_match_all end\n");
}


template<class scalar_t, int TRANS_BLOCK_SIZE, int PRE_BLOCK_SIZE, int WARP_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_grad_links_kernel(
    Accessor1 grad_links,
    Accessor2 grad_output,
    Accessor1 alpha,
    Accessor1 beta,
    Accessor1 match_all,
    Accessor1 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int prelen, int tarlen, int translen)
{
    int n_trans = ((translen - 1) / TRANS_BLOCK_SIZE + 1);
    int pre_block_idx = blockIdx.y / n_trans;
    int trans_block_idx = blockIdx.y % n_trans;
    int pre_thread_idx = threadIdx.y / TRANS_BLOCK_SIZE;
    int trans_thread_idx = threadIdx.y % TRANS_BLOCK_SIZE;

    int pre_id = pre_block_idx * PRE_BLOCK_SIZE + pre_thread_idx;
    int trans_id = trans_block_idx * TRANS_BLOCK_SIZE + trans_thread_idx;
    int batch_id = blockIdx.z;

    int presize = output_length[batch_id];
    int tarsize = target_length[batch_id];

    // if(threadIdx.x == 0 && blockIdx.z == 0){
    //     printf("%d %d %d %d: pre_id=%d, trans_id=%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, pre_id, trans_id);
    // }

    if(pre_id >= presize || trans_id >= translen) return;
    int next_pre_id = pre_id + trans_id + 1;
    if(next_pre_id >= presize || isinf(beta[batch_id][0][0])){
        if(threadIdx.x == 0) grad_links[batch_id][pre_id][trans_id] = 0;
        return;
    }

    float grad_tmp = 0;
    float extraadd = links[batch_id][pre_id][trans_id] - beta[batch_id][0][0];
    #pragma unroll(4)
    for(int i = 0; i < (tarsize - 1) / WARP_SIZE + 1; i++){
        int tar_id = threadIdx.x + WARP_SIZE * i;
        if(tar_id + 1 >= tarsize) continue;
        grad_tmp += exp(alpha[batch_id][tar_id][pre_id] + beta[batch_id][tar_id + 1][next_pre_id] + extraadd);
    }

    __syncwarp(0xffffffff);
    unsigned shfl_mask = __activemask();
    if_constexpr (WARP_SIZE > 16) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 16, WARP_SIZE);
    if_constexpr (WARP_SIZE > 8) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 8, WARP_SIZE);
    if_constexpr (WARP_SIZE > 4) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 4, WARP_SIZE);
    if_constexpr (WARP_SIZE > 2) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 2, WARP_SIZE);
    if_constexpr (WARP_SIZE > 1) grad_tmp += __shfl_down_sync(shfl_mask, grad_tmp, 1, WARP_SIZE);
    if(threadIdx.x == 0) grad_links[batch_id][pre_id][trans_id] = grad_tmp * grad_output[batch_id];
}

template<int TRANS_BLOCK_SIZE, int PRE_BLOCK_SIZE, int WARP_SIZE>
void invoke_calculate_grad_links(cudaStream_t stream, torch::Tensor &grad_links, const torch::Tensor &grad_output,
    const torch::Tensor &alpha, const torch::Tensor &beta, const torch::Tensor &match_all, const torch::Tensor &links,
    const torch::Tensor &output_length, const torch::Tensor &target_length,
    int bsz, int prelen, int tarlen, int translen)
{
    int n_block = ((translen - 1) / TRANS_BLOCK_SIZE + 1) * ((prelen - 1) / PRE_BLOCK_SIZE + 1);
    dim3 dimGrid(1, n_block, bsz);
    dim3 dimBlock(WARP_SIZE, TRANS_BLOCK_SIZE * PRE_BLOCK_SIZE, 1);
    static_assert(WARP_SIZE <= 32, "WARP_SIZE should be less than warp size");
    // printf("grad_links start\n");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_grad_links", [&] {
            calculate_grad_links_kernel<scalar_t, TRANS_BLOCK_SIZE, PRE_BLOCK_SIZE, WARP_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                grad_links.packed_accessor64<scalar_t, 3>(),
                grad_output.packed_accessor64<scalar_t, 1>(),
                alpha.packed_accessor64<scalar_t, 3>(),
                beta.packed_accessor64<scalar_t, 3>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 3>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                prelen, tarlen, translen
            );
        }
    );
    // cudaDeviceSynchronize();
    // printf("grad_links end\n");
}

std::tuple<torch::Tensor, torch::Tensor> dag_loss_backward(const torch::Tensor &grad_output, const torch::Tensor &alpha, const torch::Tensor &beta,
            const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, 
            int config1, int config2)
{
    // CHECK_CUDA(match_all);  // bsz * tarlen * prelen
    // CHECK_CUDA(links);   // bsz * prelen * translen
    // CHECK_CUDA(output_length); // bsz
    // CHECK_CUDA(target_length); // bsz
    // TORCH_CHECK(match_all.dim() == 3, "match_all dim != 3");
    // TORCH_CHECK(links.dim() == 3, "links dim != 3");
    // TORCH_CHECK(output_length.dim() == 1, "output_length dim != 3");
    // TORCH_CHECK(target_length.dim() == 1, "target_length dim != 3");

    // assume checked
    auto bsz = match_all.size(0);
    auto prelen = match_all.size(2);
    auto tarlen = match_all.size(1);
    auto translen = links.size(2);
    // TORCH_CHECK(links.size(0) == bsz && output_length.size(0) == bsz && target_length.size(0) == bsz, "batch size not match");
    // TORCH_CHECK(links.size(1) == prelen, "prelen not match");
    // TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

    auto grad_match_all = at::zeros({bsz, tarlen, prelen}, match_all.options());
    torch::Tensor grad_links = at::zeros({bsz, prelen, translen}, match_all.options());

    cudaStream_t current_stream = 0;
    // printf("alpha0\n");

    // calculate grad_match_all
    switch(config1){
        case 1: invoke_calculate_grad_match_all<4, 256, 4>(current_stream, grad_match_all, grad_output, alpha, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 2: invoke_calculate_grad_match_all<4, 128, 4>(current_stream, grad_match_all, grad_output, alpha, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        default: TORCH_CHECK(config1 <= 2 && config1 >= 1, "config1 should be 1~2");
    }
    

    // calculate grad_links
    cudaStream_t new_stream;
    cudaEvent_t cuda_event;
    cudaStreamCreate(&new_stream);
    cudaEventCreate(&cuda_event);
    switch(config2){
        case 1: invoke_calculate_grad_links<8, 32, 4>(new_stream, grad_links, grad_output, alpha, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 2: invoke_calculate_grad_links<4, 64, 4>(new_stream, grad_links, grad_output, alpha, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        case 3: invoke_calculate_grad_links<4, 32, 4>(new_stream, grad_links, grad_output, alpha, beta, match_all, links, output_length, target_length, bsz, prelen, tarlen, translen); break;
        default: TORCH_CHECK(config2 <= 3 && config2 >= 1, "config2 should be 1~3");
    }
    cudaEventRecord(cuda_event, new_stream); // new_stream triggers cuda_event
    cudaStreamWaitEvent(current_stream, cuda_event, 0); // current stream waits for event
    cudaEventDestroy(cuda_event);
    cudaStreamDestroy(new_stream);

    return std::make_tuple(grad_match_all, grad_links);
}
