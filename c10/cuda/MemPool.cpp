#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/MemPool.h>
#include <iostream>

namespace c10::cuda {

// uuid count starts at 1. 0 is reserved to mean "wasn't set by
// graph_pool_handle".
std::atomic<CaptureId_t> MemPool::uid_{1};

// id starts at 1:
// Ensures uuid count starts at 1. 0 is reserved to mean "not set by
// cudaStreamGetCaptureInfo". (But how do we know GetCaptureInfo never sets id_
// to 0? Because that's the current behavior, and I asked cuda devs to keep it
// that way, and they agreed.)
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    CUDACachingAllocator::CUDAAllocator* allocator,
    bool is_user_created)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our
    // mempool_id_. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    id_ = {uuid_++, 0};
  }
}

MempoolId_t MemPool::id() {
  return id_;
}

thread_local MemPool* MemPoolContext::active_mempool_ = nullptr;

MemPoolContext::MemPoolContext(MemPool* mempool)
    : prev_mempool_(active_mempool_) {
  active_mempool_ = mempool;
}

MemPoolContext::~MemPoolContext() {
  active_mempool_ = prev_mempool_;
}

MemPool* MemPoolContext::getActiveMemPool() {
  return active_mempool_;
}

} // namespace c10::cuda
