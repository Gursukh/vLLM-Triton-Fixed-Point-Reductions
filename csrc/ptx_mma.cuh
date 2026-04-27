// Inline-PTX wrappers for int8x int8 -> int32 tensor-core MMA.
//
// Two arch tiers:
//   sm_75 : mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
//   sm_80+: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
// The wrappers take packed int32 register operands and emit inline
// PTX. Each thread is responsible for the fragment that PTX assigns
// to its lane; the calling kernel is responsible for the
// shared-memory -> register staging that places the right int8s in
// the right lanes. See the PTX ISA, Section "Matrix multiply-and-accumulate".

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace fxpr {

// sm_75 : 8 x 8 x 16 fragment. Each thread holds:
//   a       : 1 int32 (4 int8 along K)
//   b       : 1 int32 (4 int8 along K)
//   c, d    : 2 int32 (2 int32 along N)
__device__ __forceinline__ void mma_s8s8_s32_m8n8k16(
    int32_t d[2],
    const int32_t a_packed,
    const int32_t b_packed,
    const int32_t c[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a_packed), "r"(b_packed), "r"(c[0]), "r"(c[1]));
#else
  // Fallback so the symbol exists on every arch; not intended to run.
  d[0] = c[0];
  d[1] = c[1];
#endif
}

// sm_80+ : 16 x 8 x 32 fragment. Each thread holds:
//   a       : 4 int32 (16 int8)
//   b       : 2 int32 (8 int8)
//   c, d    : 4 int32
__device__ __forceinline__ void mma_s8s8_s32_m16n8k32(
    int32_t d[4],
    const int32_t a_packed[4],
    const int32_t b_packed[2],
    const int32_t c[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a_packed[0]), "r"(a_packed[1]),
        "r"(a_packed[2]), "r"(a_packed[3]),
        "r"(b_packed[0]), "r"(b_packed[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
  d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

}  // namespace fxpr
