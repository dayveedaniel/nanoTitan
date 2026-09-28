#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cuda_check.h"
#include <string>

#define BLOCK_DIM 32


template <typename T>
__global__ void naive_gemm_cu(
    const T* __restrict__ A,
    const T* __restrict__ B, 
    T* __restrict__ C,
    int M,
    int N,
    int K
){
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (row >= M || col >= N){
        return;
    }

    float acc = 0.0f;

    for (int i = 0; i < K; i++){
        acc += float(A[row * K + i]) * float(B[i * N + col]);
    }
    
    C[row * N + col] = T(acc);
}

template <typename T, int tileN, int tileM, int tileK>
__global__ void tiled_gemm_cu(
    const T* __restrict__ A, //[N, K] 
    const T* __restrict__ B, //[K, M]
    T* __restrict__ C,       //[N, M]
    int N, 
    int M, 
    int K
){
    __shared__ T shared_A[tileN][tileK];
    __shared__ T shared_B[tileK][tileM];

    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    float sum{0.0f};

    int num_threads = blockDim.x * blockDim.y;
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    for(int tile = 0; tile < K; tile += tileK){
        // Load A's tiles
        for(int i = thread_idx; i < tileN * tileK; i += num_threads){
            int row = i / tileK;
            int k = i % tileK;

            int load_global_row = blockIdx.y * tileN + row;

            if(load_global_row < N && tile + k < K){
                shared_A[row][k] = A[load_global_row * K + tile + k];
            }
            else{
                shared_A[row][k] = 0;
            }
        }

        // Load B's tiles
        for(int i = thread_idx; i < tileK * tileM; i += num_threads){
            int k = i / tileM;
            int col = i % tileM;

            int load_global_col = blockIdx.x * tileM + col;

            if(tile + k < K && load_global_col < M){
                shared_B[k][col] = B[(tile + k) * M + load_global_col];
            }
            else{
                shared_B[k][col] = 0;
            }
        }

        __syncthreads();

        for(int z = 0; z < tileK; z++){
            sum += static_cast<float>(shared_A[tile_row][z]) * static_cast<float>(shared_B[z][tile_col]);
        }
        __syncthreads();
    }

    if (global_row < N && global_col < M) {
        C[global_row * M + global_col] = static_cast<T>(sum);
    }
}




// template <typename T>
// __global__ void tiled_gemm_cu(
//     const T* __restrict__ A, // [M, K]
//     const T* __restrict__ B, // [K, N]
//     T* __restrict__ C,       // [M, N]
//     size_t M,
//     size_t N,
//     size_t K
// ){
//     __shared__ T shared_A[BLOCK_DIM][BLOCK_DIM];
//     __shared__ T shared_B[BLOCK_DIM][BLOCK_DIM];

//     size_t global_row = blockIdx.y * blockDim.y + threadIdx.y;
//     size_t global_col = blockIdx.x * blockDim.x + threadIdx.x;

//     float acc{0.0f};
//     for(size_t k = 0; k < K; k += BLOCK_DIM){
//         if(global_row >= M or k + threadIdx.x >= K){
//             shared_A[threadIdx.y][threadIdx.x] = 0;
//         }
//         else{
//             shared_A[threadIdx.y][threadIdx.x] = A[global_row * K + (k + threadIdx.x)];
//         }
        
//         if(global_col >= N or k + threadIdx.y >= K){
//             shared_B[threadIdx.y][threadIdx.x] = 0;
//         }
//         else{
//             shared_B[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + global_col];
//         }

//         __syncthreads();

//         for(int idx = 0; idx < BLOCK_DIM; idx++){
//             acc += static_cast<float>(shared_A[threadIdx.y][idx]) * static_cast<float>(shared_B[idx][threadIdx.x]);
//         }

//         __syncthreads();
//     }

//     if(global_row < M and global_col < N){
//         C[global_row * N + global_col] = static_cast<T>(acc);
//     }
// }

namespace {

void check_gemm_inputs(
    torch::Tensor A,
    torch::Tensor B
){
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");
    TORCH_CHECK(A.is_contiguous(), "A should be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B should be contiguous");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same dtype");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimension should be the same for the matrices in the matmul");
    TORCH_CHECK(
        A.scalar_type() == at::ScalarType::Float ||
        A.scalar_type() == at::ScalarType::Double ||
        A.scalar_type() == at::ScalarType::Half ||
        A.scalar_type() == at::ScalarType::BFloat16,
        "gemm_kernel supports float32, float64, float16, and bfloat16 tensors"
    );
}

torch::Tensor launch_naive_gemm(
    torch::Tensor A,
    torch::Tensor B
){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    dim3 threads (16, 16);
    dim3 blocks (
        (M + threads.y - 1) / threads.y,
        (N + threads.x - 1) / threads.x
    );

    c10::cuda::CUDAGuard guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "naive_gemm_kernel",
        [&] {
            naive_gemm_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, 
                N, 
                K
            );
        }
    );
    CUDA_KERNEL_CHECK();

    return C;
}

torch::Tensor launch_tiled_gemm(
    torch::Tensor A,
    torch::Tensor B
){
    constexpr int tileN = 16;
    constexpr int tileM = 32;
    constexpr int tileK = 8;

    size_t N = static_cast<size_t>(A.size(0));
    size_t K = static_cast<size_t>(A.size(1));
    size_t M = static_cast<size_t>(B.size(1));

    torch::Tensor C = torch::empty(
        {static_cast<int64_t>(N), static_cast<int64_t>(M)},
        A.options()
    );

    dim3 block(tileM, tileN);
    dim3 grid (
        (M + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    c10::cuda::CUDAGuard guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "tiled_gemm_kernel",
        [&] {
            tiled_gemm_cu<scalar_t, tileN, tileM, tileK><<<grid, block, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                N,
                M,
                K
            );
        }
    );
    CUDA_KERNEL_CHECK();

    return C;
}

} // namespace

torch::Tensor gemm_kernel(
    torch::Tensor A,
    torch::Tensor B,
    const std::string& implementation
){
    check_gemm_inputs(A, B);

    if (implementation == "tiled"){
        return launch_tiled_gemm(A, B);
    }
    if (implementation == "naive"){
        return launch_naive_gemm(A, B);
    }

    TORCH_CHECK(false, "implementation must be either 'tiled' or 'naive'");
}

torch::Tensor tiled_gemm_kernel(
    torch::Tensor A,
    torch::Tensor B
){
    return gemm_kernel(A, B, "tiled");
}

torch::Tensor naive_gemm_kernel(
    torch::Tensor A,
    torch::Tensor B
){
    return gemm_kernel(A, B, "naive");
}
