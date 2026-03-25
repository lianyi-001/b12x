#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 64;
constexpr int kCols = 256;
constexpr int kSubtileCols = 64;
constexpr int kSubtiles = kCols / kSubtileCols;
constexpr int kElements = kRows * kCols;
constexpr int kWords = kElements / 2;
constexpr int kStageBytes = kElements * int(sizeof(uint16_t));

__device__ __forceinline__ uint32_t smem_ptr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t expected_count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(smem_ptr(bar)), "r"(expected_count) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"
      :
      : "r"(smem_ptr(bar)), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* bar, uint32_t phase) {
  uint32_t ready = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "}\n\t"
      : "=r"(ready)
      : "r"(smem_ptr(bar)), "r"(phase)
      : "memory");
  return ready != 0;
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    void* dst,
    const void* tensor_map,
    int c0,
    int c1,
    uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(smem_ptr(dst)), "l"(tensor_map), "r"(c0), "r"(c1), "r"(smem_ptr(bar))
      : "memory");
}

__global__ void probe_tma_live_layout_kernel(
    const uint16_t* __restrict__ src,
    const CUtensorMap* __restrict__ tensor_map,
    uint32_t* __restrict__ out_words,
    bool plane_slabs) {
  __shared__ alignas(1024) uint16_t stage[kElements];
  __shared__ alignas(8) uint64_t bars[kSubtiles];

  if (threadIdx.x == 0) {
    for (int g = 0; g < kSubtiles; ++g) {
      mbarrier_init(&bars[g], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int g = 0; g < kSubtiles; ++g) {
      uint16_t* dst = plane_slabs ? (stage + g * kRows * kSubtileCols) : (stage + g * kSubtileCols);
      mbarrier_arrive_expect_tx(&bars[g], kRows * kSubtileCols * int(sizeof(uint16_t)));
      cp_async_bulk_tensor_2d(
          dst,
          tensor_map,
          g * kSubtileCols,
          0,
          &bars[g]);
    }
  }
  __syncthreads();

  for (int g = 0; g < kSubtiles; ++g) {
    while (!mbarrier_try_wait_parity(&bars[g], 0)) {
    }
  }
  __syncthreads();

  const uint32_t* stage_words = reinterpret_cast<const uint32_t*>(stage);
  for (int word_idx = threadIdx.x; word_idx < kWords; word_idx += blockDim.x) {
    out_words[word_idx] = stage_words[word_idx];
  }
}

int permuted_word_index(int row, int col_halfword_pair) {
  const int elem_col = col_halfword_pair * 2;
  const int vec_idx = elem_col / 8;
  const int lane_in_vec = elem_col % 8;
  const int permuted_vec = vec_idx ^ (row % 8);
  const int permuted_elem = permuted_vec * 8 + lane_in_vec;
  return row * kCols + permuted_elem;
}

CUtensorMapSwizzle parse_swizzle(const std::string& name) {
  if (name == "none") {
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
  if (name == "128b") {
    return CU_TENSOR_MAP_SWIZZLE_128B;
  }
  if (name == "atom32") {
    return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
  }
  if (name == "atom64") {
    return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
  }
  std::fprintf(stderr, "unsupported swizzle: %s\n", name.c_str());
  std::exit(2);
}

const char* swizzle_name(CUtensorMapSwizzle swizzle) {
  switch (swizzle) {
    case CU_TENSOR_MAP_SWIZZLE_NONE:
      return "none";
    case CU_TENSOR_MAP_SWIZZLE_128B:
      return "128b";
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B:
      return "atom32";
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B:
      return "atom64";
    default:
      return "unknown";
  }
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

void check_cu(CUresult err, const char* what) {
  if (err != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &text);
    std::fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "?", text ? text : "?");
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string swizzle_arg = argc > 1 ? argv[1] : "128b";
  const std::string layout_mode = argc > 2 ? argv[2] : "fullrow";
  const CUtensorMapSwizzle swizzle = parse_swizzle(swizzle_arg);
  const bool plane_slabs = layout_mode == "planes";
  if (!plane_slabs && layout_mode != "fullrow") {
    std::fprintf(stderr, "unsupported layout mode: %s\n", layout_mode.c_str());
    return 2;
  }

  check_cuda(cudaSetDevice(0), "cudaSetDevice");
  check_cuda(cudaFree(nullptr), "cudaFree");
  check_cu(cuInit(0), "cuInit");

  std::vector<uint16_t> host_src(kElements);
  for (int idx = 0; idx < kElements; ++idx) {
    host_src[idx] = static_cast<uint16_t>(idx);
  }

  uint16_t* dev_src = nullptr;
  uint32_t* dev_out = nullptr;
  CUtensorMap* dev_tensor_map = nullptr;
  check_cuda(cudaMalloc(&dev_src, kStageBytes), "cudaMalloc(dev_src)");
  check_cuda(cudaMalloc(&dev_out, kWords * sizeof(uint32_t)), "cudaMalloc(dev_out)");
  check_cuda(cudaMemcpy(dev_src, host_src.data(), kStageBytes, cudaMemcpyHostToDevice), "cudaMemcpy(dev_src)");

  alignas(64) CUtensorMap host_tensor_map{};
  std::array<uint64_t, 2> global_dim = {static_cast<uint64_t>(kCols), static_cast<uint64_t>(kRows)};
  std::array<uint64_t, 1> global_stride = {static_cast<uint64_t>(kCols * sizeof(uint16_t))};
  std::array<uint32_t, 2> box_dim = {static_cast<uint32_t>(kSubtileCols), static_cast<uint32_t>(kRows)};
  std::array<uint32_t, 2> element_stride = {1u, 1u};

  check_cu(
      cuTensorMapEncodeTiled(
          &host_tensor_map,
          CU_TENSOR_MAP_DATA_TYPE_UINT16,
          2,
          dev_src,
          global_dim.data(),
          global_stride.data(),
          box_dim.data(),
          element_stride.data(),
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");

  check_cuda(cudaMalloc(&dev_tensor_map, sizeof(CUtensorMap)), "cudaMalloc(dev_tensor_map)");
  check_cuda(
      cudaMemcpy(dev_tensor_map, &host_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
      "cudaMemcpy(dev_tensor_map)");

  probe_tma_live_layout_kernel<<<1, 128>>>(dev_src, dev_tensor_map, dev_out, plane_slabs);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<uint32_t> host_out(kWords);
  check_cuda(cudaMemcpy(host_out.data(), dev_out, kWords * sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy(dev_out)");

  std::vector<uint32_t> expected(kWords);
  if (plane_slabs) {
    for (int plane = 0; plane < kSubtiles; ++plane) {
      const int plane_elem_base = plane * kSubtileCols;
      const int plane_word_base = plane * (kRows * (kSubtileCols / 2));
      for (int row = 0; row < kRows; ++row) {
        for (int pair_idx = 0; pair_idx < kSubtileCols / 2; ++pair_idx) {
          const int dst_word = plane_word_base + row * (kSubtileCols / 2) + pair_idx;
          const int elem_col = pair_idx * 2;
          const int vec_idx = elem_col / 8;
          const int lane_in_vec = elem_col % 8;
          const int permuted_vec = vec_idx ^ (row % 8);
          const int src_elem = row * kCols + plane_elem_base + permuted_vec * 8 + lane_in_vec;
          const uint32_t lo = static_cast<uint32_t>(src_elem & 0xFFFF);
          const uint32_t hi = static_cast<uint32_t>((src_elem + 1) & 0xFFFF);
          expected[dst_word] = lo | (hi << 16);
        }
      }
    }
  } else {
    for (int row = 0; row < kRows; ++row) {
      for (int pair_idx = 0; pair_idx < kCols / 2; ++pair_idx) {
        const int dst_word = row * (kCols / 2) + pair_idx;
        const int src_elem = permuted_word_index(row, pair_idx);
        const uint32_t lo = static_cast<uint32_t>(src_elem & 0xFFFF);
        const uint32_t hi = static_cast<uint32_t>((src_elem + 1) & 0xFFFF);
        expected[dst_word] = lo | (hi << 16);
      }
    }
  }

  int mismatch_count = 0;
  int first_mismatch = -1;
  for (int idx = 0; idx < kWords; ++idx) {
    if (host_out[idx] != expected[idx]) {
      ++mismatch_count;
      if (first_mismatch == -1) {
        first_mismatch = idx;
      }
    }
  }

  std::printf("{\n");
  std::printf("  \"swizzle\": \"%s\",\n", swizzle_name(swizzle));
  std::printf("  \"layout_mode\": \"%s\",\n", plane_slabs ? "planes" : "fullrow");
  std::printf("  \"mismatch_count\": %d,\n", mismatch_count);
  if (first_mismatch >= 0) {
    const int row = first_mismatch / (kCols / 2);
    const int word_in_row = first_mismatch - row * (kCols / 2);
    std::printf("  \"first_mismatch\": {\n");
    std::printf("    \"word\": %d,\n", first_mismatch);
    std::printf("    \"row\": %d,\n", row);
    std::printf("    \"word_in_row\": %d,\n", word_in_row);
    std::printf("    \"got\": \"0x%08x\",\n", host_out[first_mismatch]);
    std::printf("    \"expected\": \"0x%08x\"\n", expected[first_mismatch]);
    std::printf("  },\n");
  } else {
    std::printf("  \"first_mismatch\": null,\n");
  }
  std::printf("  \"first_words\": [");
  for (int idx = 0; idx < 16; ++idx) {
    std::printf("%s\"0x%08x\"", idx == 0 ? "" : ", ", host_out[idx]);
  }
  std::printf("]\n");
  std::printf("}\n");

  cudaFree(dev_tensor_map);
  cudaFree(dev_out);
  cudaFree(dev_src);
  return mismatch_count == 0 ? 0 : 3;
}
