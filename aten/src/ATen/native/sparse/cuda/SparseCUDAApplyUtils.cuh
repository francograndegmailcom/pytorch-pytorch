#pragma once

#include <ATen/cuda/detail/TensorInfo.cuh>

namespace at { namespace native {

namespace apply {

using at::cuda::detail::TensorInfo;
using indexT = int64_t;

const int WARP_SIZE = 32;

template <typename IndexType, typename Real, typename Op>
__device__ void applyOp2(
    Op op, IndexType blockSize,
    TensorInfo<Real, IndexType> values1, IndexType idx1,
    TensorInfo<Real, IndexType> values2, IndexType idx2) {
  for (IndexType k = blockIdx.x * blockDim.x + threadIdx.x;
       k < blockSize;
       k += gridDim.x * blockDim.x) {
    op(values1.data + idx1 * blockSize + k, values2.data + idx2 * blockSize + k);
  }
}

template <typename IndexType, typename Real, typename Op>
__device__ void applyOp3(
    Op op, IndexType blockSize,
    TensorInfo<Real, IndexType> values1, IndexType idx1,
    TensorInfo<Real, IndexType> values2, IndexType idx2,
    TensorInfo<Real, IndexType> values3, IndexType idx3) {
  for (IndexType k = blockIdx.x * blockDim.x + threadIdx.x;
       k < blockSize;
       k += gridDim.x * blockDim.x) {
    op(values1.data + idx1 * blockSize + k,
       values2.data + idx2 * blockSize + k,
       values3.data + idx3 * blockSize + k);
  }
}

// Assume both dense and values are contiguous.
// Currently only used in add_out_dense_sparse_cuda: add(dense, sparse, scalar).
template <typename Op, typename IndexType, typename Real>
__global__ void sparseElementwiseKernel(
    Op op,
    TensorInfo<Real, IndexType> dense,
    TensorInfo<indexT, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  IndexType ind_skip = indices.strides[0];
  IndexType ind_nnz_skip = indices.strides[1];
  IndexType value_size = values.strides[0];  // numel of each slice in values
  for (IndexType linearId = blockIdx.x;
       linearId < nnz;
       linearId += gridDim.x) {
    IndexType index = 0;
    for (IndexType d = 0; d < indices.sizes[0]; d++) {
      index = dense.sizes[d] * index + indices.data[d * ind_skip + linearId * ind_nnz_skip];
    }
    Real *dst = dense.data + index * value_size;
    Real *src = values.data + linearId * value_size;
    for (IndexType linearId2 = threadIdx.x; linearId2 < value_size; linearId2 += blockDim.x) {
      op(dst + linearId2, src + linearId2);
    }
  }
}

// Assume dense is contiguous.
// Currently only used in add_out_dense_sparse_cuda: add(dense, sparse, scalar).
template <typename Op, typename IndexType, typename Real>
__global__ void sparseElementwiseKernelScalar(
    Op op,
    TensorInfo<Real, IndexType> dense,
    TensorInfo<indexT, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  IndexType ind_skip = indices.strides[0];
  IndexType ind_nnz_skip = indices.strides[1];
  IndexType value_skip = values.strides[0];
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < nnz;
       linearId += gridDim.x * blockDim.x) {
    IndexType index = 0;
    for (IndexType d = 0; d < indices.sizes[0]; d++) {
      index = dense.sizes[d] * index + indices.data[d * ind_skip + linearId * ind_nnz_skip];
    }
    op(dense.data + index, values.data + linearId * value_skip);
  }
}

template <typename OpBoth, typename OpLeft, typename OpRight, typename IndexType, typename Real>
__global__ void valueSparseUnionKernel(
    OpBoth opBoth,
    OpLeft opLeft,
    OpRight opRight,
    TensorInfo<indexT, IndexType> r_indices,
    TensorInfo<indexT, IndexType> t_indices,
    TensorInfo<indexT, IndexType> s_indices,
    TensorInfo<Real, IndexType> r_values,
    TensorInfo<Real, IndexType> t_values,
    TensorInfo<Real, IndexType> s_values,
    const IndexType t_nnz, const IndexType s_nnz) {
  IndexType t_indskip = t_indices.strides[0];
  IndexType s_indskip = s_indices.strides[0];
  int64_t cmp, d;
  int64_t nDimI = r_indices.sizes[0];
  IndexType valueSize = r_values.strides[0];
  IndexType r_i = 0, t_i = 0, s_i = 0;
  while (t_i < t_nnz || s_i < s_nnz) {
    if (t_i >= t_nnz) {
      cmp = -1;
    } else if (s_i >= s_nnz) {
      cmp = 1;
    } else {
      cmp = 0;
      for (d = 0; d < nDimI; d++) {
        if (t_indices.data[d * t_indskip + t_i] < s_indices.data[d * s_indskip + s_i]) {
          cmp = 1;
          break;
        }
        if (t_indices.data[d * t_indskip + t_i] > s_indices.data[d * s_indskip + s_i]) {
          cmp = -1;
          break;
        }
      }
    }
    if (cmp == 0) applyOp3(opBoth, valueSize, r_values, r_i, t_values, t_i++, s_values, s_i++);
    else if (cmp > 0) applyOp2(opLeft, valueSize, r_values, r_i, t_values, t_i++);
    else if (cmp < 0) applyOp2(opRight, valueSize, r_values, r_i, s_values, s_i++);
    r_i++;
  }
}

// TODO find a way to parallelize this...
template <typename IndexType, typename Real>
__global__ void indexSparseUnionKernel(
    TensorInfo<indexT, IndexType> r_indices,
    TensorInfo<indexT, IndexType> t_indices,
    TensorInfo<indexT, IndexType> s_indices,
    const IndexType t_nnz, const IndexType s_nnz, IndexType *resultNnz) {
  IndexType r_indskip = r_indices.strides[0];
  IndexType t_indskip = t_indices.strides[0];
  IndexType s_indskip = s_indices.strides[0];
  int64_t cmp, d;
  int64_t nDimI = r_indices.sizes[0];
  IndexType r_i = 0, t_i = 0, s_i = 0;
  while (t_i < t_nnz || s_i < s_nnz) {
    if (t_i >= t_nnz) {
      cmp = -1;
    } else if (s_i >= s_nnz) {
      cmp = 1;
    } else {
      cmp = 0;
      for (d = 0; d < nDimI; d++) {
        if (t_indices.data[d * t_indskip + t_i] < s_indices.data[d * s_indskip + s_i]) {
          cmp = 1;
          break;
        }
        if (t_indices.data[d * t_indskip + t_i] > s_indices.data[d * s_indskip + s_i]) {
          cmp = -1;
          break;
        }
      }
    }
    if (cmp >= 0) {
      for (d = 0; d < nDimI; d++) {
        r_indices.data[d * r_indskip + r_i] = t_indices.data[d * t_indskip + t_i];
      }
      t_i++;
    }
    if (cmp <= 0) {
      for (d = 0; d < nDimI; d++) {
        r_indices.data[d * r_indskip + r_i] = s_indices.data[d * s_indskip + s_i];
      }
      s_i++;
    }
    r_i++;
  }
  *resultNnz = r_i;
}

template <typename Op, typename IndexType, typename Real>
__global__ void valueSparseIntersectionKernel(
    Op op,
    TensorInfo<indexT, IndexType> r_indices,
    TensorInfo<indexT, IndexType> t_indices,
    TensorInfo<indexT, IndexType> s_indices,
    TensorInfo<Real, IndexType> r_values,
    TensorInfo<Real, IndexType> t_values,
    TensorInfo<Real, IndexType> s_values,
    const IndexType t_nnz, const IndexType s_nnz) {
  IndexType t_indskip = t_indices.strides[0];
  IndexType s_indskip = s_indices.strides[0];
  int64_t match, d;
  int64_t nDimI = r_indices.sizes[0];
  IndexType valueSize = r_values.strides[0];
  IndexType r_i = 0, t_i = 0, s_i = 0;
  while (t_i < t_nnz && s_i < s_nnz) {
    match = 1;
    for (d = 0; d < nDimI; d++) {
      if (t_indices.data[d * t_indskip + t_i] < s_indices.data[d * s_indskip + s_i]) {
        t_i++;
        match = 0;
        break;
      }
      if (t_indices.data[d * t_indskip + t_i] > s_indices.data[d * s_indskip + s_i]) {
        s_i++;
        match = 0;
        break;
      }
    }
    if (!match) continue;
    applyOp3(op, valueSize, r_values, r_i++, t_values, t_i++, s_values, s_i++);
  }
}

// TODO find a way to parallelize this...
template <typename IndexType, typename Real>
__global__ void indexSparseIntersectionKernel(
    TensorInfo<indexT, IndexType> r_indices,
    TensorInfo<indexT, IndexType> t_indices,
    TensorInfo<indexT, IndexType> s_indices,
    const IndexType t_nnz, const IndexType s_nnz, IndexType *resultNnz) {
  IndexType r_indskip = r_indices.strides[0];
  IndexType t_indskip = t_indices.strides[0];
  IndexType s_indskip = s_indices.strides[0];
  int64_t match, d;
  int64_t nDimI = r_indices.sizes[0];
  IndexType r_i = 0, t_i = 0, s_i = 0;
  while (t_i < t_nnz && s_i < s_nnz) {
    match = 1;
    for (d = 0; d < nDimI; d++) {
      if (t_indices.data[d * t_indskip + t_i] < s_indices.data[d * s_indskip + s_i]) {
        t_i++;
        match = 0;
        break;
      }
      if (t_indices.data[d * t_indskip + t_i] > s_indices.data[d * s_indskip + s_i]) {
        s_i++;
        match = 0;
        break;
      }
    }
    if (!match) continue;
    for (d = 0; d < nDimI; d++) {
      r_indices.data[d * r_indskip + r_i] = t_indices.data[d * t_indskip + t_i];
    }
    r_i++; t_i++; s_i++;
  }
  *resultNnz = r_i;
}

// template <typename Dtype, typename Acctype>
// __global__ void coalesceValuesKernel_gridStrided(
//   long *segment_offsets, long *value_indices,
//   Dtype *values, Dtype *newValues,
//   long nnz, long newNnz, long stride) {
//
//   long chunksPerSeg = THCCeilDiv(stride, (long) blockDim.x);
//   long numChunks = newNnz * chunksPerSeg;
//   long chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
//   long chunkStride = gridDim.x * blockDim.y;
//
//   for (long chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
//     long featureDim = (chunk % chunksPerSeg) * blockDim.x + threadIdx.x;
//     if (featureDim < stride) {
//       auto valFeat = values + featureDim;
//       long seg = chunk / chunksPerSeg;
//       auto begin = segment_offsets[seg];
//       auto end = (seg < newNnz - 1) ? segment_offsets[seg + 1] : nnz;
//       Acctype valSum = static_cast<Acctype>::to(0);
//       for (long valIdx = begin; valIdx < end; valIdx++) {
//         const long valRow = value_indices[valIdx] * stride;
//         valSum += static_cast<Acctype>::to(valFeat[valRow]);
//       }
//       newValues[seg * stride + featureDim] = static_cast<Dtype>::to(valSum);
//     }
//   }
// }

template <typename scalar_t>
__device__ scalar_t coalesce_sum_op(scalar_t a, scalar_t b) {
  return a + b;
}

template <typename scalar_t, typename func_t>
__device__ void coalesceValuesKernel(
  int64_t *segment_offsets,
  int64_t *value_indices,
  scalar_t *values,
  scalar_t *newValues,
  int64_t nnz,
  int64_t newNnz,
  int64_t stride,
  const func_t& op
) {

  int seg = blockIdx.x * 4 + threadIdx.y;

  // Number of values processed by each thread (grain size)
  const int SZ = 4;

  if (seg < newNnz) {
    const int newValueRow = seg * stride;
    const int begin = segment_offsets[seg];
    const int end = (seg < newNnz - 1) ? segment_offsets[seg + 1] : nnz;
    const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
    scalar_t tmp[SZ];
    #pragma unroll
    for (int ii = 0; ii < SZ; ii++) {
      tmp[ii] = 0;
    }
    for (int row = begin; row < end; row++) {
      const int valueRow = ((int) value_indices[row]) * stride;

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          // tmp[ii] += static_cast<cuda_accscalar_t>(values[valueRow + featureDim]);
          tmp[ii] = op(tmp[ii], static_cast<scalar_t>(values[valueRow + featureDim]));
        }
      }
    }
    #pragma unroll
    for (int ii = 0; ii < SZ; ii++)
    {
      int featureDim = startFeature + ii * WARP_SIZE;
      if (featureDim < stride)
      {
        newValues[newValueRow + featureDim] = static_cast<scalar_t>(tmp[ii]);
      }
    }
  }
}

template <typename scalar_t>
__global__ void coalesce_sum_kernel(
  int64_t *segment_offsets,
  int64_t *value_indices,
  scalar_t *values,
  scalar_t *newValues,
  int64_t nnz,
  int64_t newNnz,
  int64_t stride
) {
  coalesceValuesKernel(
    segment_offsets,
    value_indices,
    values,
    newValues,
    nnz,
    newNnz,
    stride,
    coalesce_sum_op<scalar_t>
  );
}

} // namespace apply

}} // namespace at::native
