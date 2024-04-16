#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/host_tensor.h>
#include <torch/library.h>
#include <torch/version.h>
#include "pyg_lib/csrc/utils/convert.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace pyg {
namespace ops {

namespace {

cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// Returns the bytes of shared memory available per SM on the GPU, or -1 on
// error.
cudaDeviceProp get_dev_prop() {
  cudaDeviceProp properties;
  int device_idx;
  C10_CUDA_CHECK(cudaGetDevice(&device_idx));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_idx));
  return properties;
}
cudaDeviceProp props;
bool props_queried = false;

void grouped_matmul_out_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out,
                               bool segment) {
  /* Grouped Gemm Inputs */
  const int64_t num_matrices = input.size();
  cublasHandle_t handle;
  if (!props_queried) {
    props = get_dev_prop();
    props_queried = true;
  }
  bool use_tf32;
  if (props.major >= 8) {
    use_tf32 = false;
  }  else {
#if TORCH_VERSION_MINOR >= 12 or TORCH_VERSION_MAJOR > 1
    use_tf32 = at::globalContext().float32MatmulPrecision() !=
               at::Float32MatmulPrecision::HIGHEST;
#else
    use_tf32 = at::globalContext().allowTF32CuBLAS();
#endif
  }

  std::vector<cublasOperation_t> ta(num_matrices);
  std::vector<cublasOperation_t> tb(num_matrices);
  std::vector<int> m(num_matrices);
  std::vector<int> n(num_matrices);
  std::vector<int> k(num_matrices);
  std::vector<float> alpha(num_matrices);
  std::vector<int> lda(num_matrices);
  std::vector<int> ldb(num_matrices);
  std::vector<float> beta(num_matrices);
  std::vector<int> ldc(num_matrices);
  std::vector<int> group_size(num_matrices);
  int group_count = num_matrices;


  int64_t* ptr_A_data;
  int64_t* ptr_B_data;
  int64_t* ptr_C_data;

  float** d_ptrA_data;
  float** d_ptrB_data;
  float** d_ptrC_data;

  auto size = sizeof(float*) * num_matrices;
  auto empty_tens_A = at::empty({static_cast<int64_t>(size)}, input[0].options().dtype(at::kByte));
  d_ptrA_data = (float**) empty_tens_A.data_ptr();
  auto empty_tens_B = at::empty({static_cast<int64_t>(size)}, other[0].options().dtype(at::kByte));
  d_ptrB_data = (float**) empty_tens_B.data_ptr();
  auto empty_tens_C = at::empty({static_cast<int64_t>(size)}, out[0].options().dtype(at::kByte));
  d_ptrC_data = (float**) empty_tens_C.data_ptr();
  const unsigned int bytes = num_matrices * sizeof(float);
  cudaError_t err;
  bool isPtrArrayPinned = true;
  err = cudaMallocHost((void**)&ptr_A_data, 3 * sizeof(int64_t) * num_matrices); // allocate all three in one allocation
  if (err) {
    isPtrArrayPinned = false;
    ptr_A_data = new int64_t[3 * num_matrices];
  }

  ptr_B_data = ptr_A_data + num_matrices;
  ptr_C_data = ptr_B_data + num_matrices;


  /* Auxiliary inputs needed for setup */
  for (size_t i = 0; i < num_matrices; ++i) {
    auto new_in = input[i];
    auto new_other = other[i];
    auto new_out = out[i].contiguous();
    m[i] = new_in.size(0);
    k[i] = new_other.size((int)(segment));
    n[i] = new_out.size(1);



    ptr_A_data[i] = reinterpret_cast<int64_t>(new_in.data_ptr<float>());
    ptr_B_data[i] = reinterpret_cast<int64_t>(new_other.data_ptr<float>());
    ptr_C_data[i] = reinterpret_cast<int64_t>(new_out.data_ptr<float>());

    ta[i] = CUBLAS_OP_N;
    tb[i] = CUBLAS_OP_N;
    group_size[i] = 1;
    alpha[i] = 1.0f;
    beta[i] = 0.0f;

    lda[i] = (ta[i] == CUBLAS_OP_N) ? k[i] : m[i];
    ldb[i] = (tb[i] == CUBLAS_OP_N) ? n[i] : k[i];
    ldc[i] = n[i];
  }

  cudaMemcpy(d_ptrA_data, ptr_A_data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptrB_data, ptr_B_data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptrC_data, ptr_C_data, size, cudaMemcpyHostToDevice);

  handle = at::cuda::getCurrentCUDABlasHandle();
  if (use_tf32) {
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  } else {
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  }
  // use row major instead of default column major for CUBLAS
  cublasStatus_t status = cublasSgemmGroupedBatched(handle, tb.data(), ta.data(), n.data(), m.data(), k.data(), alpha.data(), d_ptrB_data,
                                     ldb.data(), d_ptrA_data, lda.data(), beta.data(), d_ptrC_data, ldc.data(),
                                     group_count, group_size.data());
  TORCH_CUDABLAS_CHECK(status);
}

std::vector<at::Tensor> grouped_matmul_kernel(const at::TensorList input,
                                              const at::TensorList other) {
  std::vector<at::Tensor> out(input.size());
  std::vector<at::Tensor> new_input(input.size());
  std::vector<at::Tensor> new_other(input.size());
  for (size_t i = 0; i < input.size(); ++i){
    new_input[i] = input[i].contiguous();
    new_other[i] = other[i].contiguous();
    out[i] = input[i].new_empty({new_input[i].size(0), new_other[i].size(-1)});
  }
  grouped_matmul_out_kernel(new_input, new_other, out, false);

  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  // TODO (matthias) Allow for other types than `int64_t`.
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto out = input.new_empty({input.size(0), other.size(-1)});

  // TODO (matthias) Better handle non-contiguous memory layouts.
  grouped_matmul_out_kernel(
      input.contiguous().split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other.contiguous().split(/*split_size=*/1, /*dim=*/0),
      out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0), true);

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
