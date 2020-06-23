#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <cstring>

#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace qnnpack {
struct q8gemm_context {
  size_t k;
  size_t k_stride;
  size_t n;
  size_t n_stride;
  const uint8_t* a;
  size_t a_stride;
  const uint8_t* packed_w;
  uint8_t* c;
  size_t c_stride;
  union pytorch_qnnp_conv_quantization_params quantization_params;
  const pytorch_q8gemm_ukernel_function ukernel;
};

struct fpu_state {
#if defined(__SSE__) || defined(__x86_64__)
  uint32_t mxcsr;
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  uint32_t fpscr;
#elif defined(__aarch64__)
  uint64_t fpcr;
#else
  char unused;
#endif
};

static inline struct fpu_state get_fpu_state() {
  struct fpu_state state = { 0 };
#if defined(__SSE__) || defined(__x86_64__)
  state.mxcsr = (uint32_t) _mm_getcsr();
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  __asm__ __volatile__("VMRS %[fpscr], fpscr" : [fpscr] "=r" (state.fpscr));
#elif defined(__aarch64__)
  __asm__ __volatile__("MRS %[fpcr], fpcr" : [fpcr] "=r" (state.fpcr));
#endif
  return state;
}

static inline void set_fpu_state(const struct fpu_state state) {
#if defined(__SSE__) || defined(__x86_64__)
  _mm_setcsr((unsigned int) state.mxcsr);
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  __asm__ __volatile__("VMSR fpscr, %[fpscr]" : : [fpscr] "r" (state.fpscr));
#elif defined(__aarch64__)
  __asm__ __volatile__("MSR fpcr, %[fpcr]" : : [fpcr] "r" (state.fpcr));
#endif
}

static inline void disable_fpu_denormals() {
#if defined(__SSE__) || defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040);
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  uint32_t fpscr;
  __asm__ __volatile__(
      "VMRS %[fpscr], fpscr\n"
      "ORR %[fpscr], #0x1000000\n"
      "VMSR fpscr, %[fpscr]\n"
    : [fpscr] "=r" (fpscr));
#elif defined(__aarch64__)
  uint64_t fpcr;
  __asm__ __volatile__(
      "MRS %[fpcr], fpcr\n"
      "ORR %w[fpcr], %w[fpcr], 0x1000000\n"
      "ORR %w[fpcr], %w[fpcr], 0x80000\n"
      "MSR fpcr, %[fpcr]\n"
    : [fpcr] "=r" (fpcr));
#endif
}

static void compute_q8gemm(
    const struct q8gemm_context context[1],
    size_t group_index,
    size_t pixel_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t group_range /* always 1 */,
    size_t pixel_range,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t k = context->k;
  const size_t k_stride = context->k_stride;
  const size_t n = context->n;
  const size_t n_stride = context->n_stride;
  const uint8_t* a = context->a;
  const size_t a_stride = context->a_stride;
  const void* packed_w = context->packed_w;
  uint8_t* c = context->c;
  const size_t c_stride = context->c_stride;

  const fpu_state saved_fpu_state = get_fpu_state();
  disable_fpu_denormals();

  size_t output_channel_index = nr_block_start;
  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      (const void*) ((uintptr_t) packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start + group_index * n,
      c_stride,
      output_channel_index,
      &context->quantization_params);

  set_fpu_state(saved_fpu_state);
}

enum pytorch_qnnp_status qnnpackLinear(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    const size_t output_stride,
    pthreadpool_t threadpool)
{
  const size_t groups = 1;
  const size_t group_input_channels = input_channels;
  const size_t group_output_channels = output_channels;
  const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
  const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
  const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

  const size_t output_size = batch_size * 1;
  union pytorch_qnnp_conv_quantization_params conv_quantization_params =
      pytorch_qnnp_compute_conv_quantization_params(
          input_zero_point, kernel_zero_points,
          requantization_scales, output_zero_point, output_min, output_max);

  struct q8gemm_context q8gemm_context = {
      .k = group_input_channels,
      .k_stride = k_stride,
      .n = group_output_channels,
      .n_stride = n_stride,
      .a = input,
      .a_stride = input_stride,
      .packed_w = (uint8_t*) packed_weights,
      .c = output,
      .c_stride = output_stride,
      .quantization_params = conv_quantization_params,
      .ukernel = pytorch_qnnp_params.q8conv.gemm,
  };

  pthreadpool_compute_4d_tiled(
      threadpool,
      (pthreadpool_function_4d_tiled_t) compute_q8gemm,
      &q8gemm_context,
      groups,
      1 * output_size,
      output_size,
      group_output_channels,
      1,
      output_size,
      mr,
      nr);

  return pytorch_qnnp_status_success;
}
} // namespace qnnpack
