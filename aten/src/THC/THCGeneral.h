#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include "THGeneral.h"
#include "THAllocator.h"
#include "THCThreadLocal.h"
#undef log10
#undef log1p
#undef log2
#undef expm1

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cusparse.h"

#define USE_MAGMA

#ifdef __cplusplus
# define THC_EXTERNC extern "C"
#else
# define THC_EXTERNC extern
#endif

#ifdef _WIN32
# ifdef THC_EXPORTS
#  define THC_API THC_EXTERNC __declspec(dllexport)
#  define THC_CLASS __declspec(dllexport)
# else
#  define THC_API THC_EXTERNC __declspec(dllimport)
#  define THC_CLASS __declspec(dllimport)
# endif
#else
# define THC_API THC_EXTERNC
# define THC_CLASS
#endif

#ifndef THAssert
#define THAssert(exp)                                                   \
  do {                                                                  \
    if (!(exp)) {                                                       \
      _THError(__FILE__, __LINE__, "assert(%s) failed", #exp);          \
    }                                                                   \
  } while(0)
#endif

struct THCRNGState;  /* Random number generator state. */
typedef struct THCStream THCStream;
typedef struct THCState THCState;

typedef struct _THCDeviceAllocator {
   cudaError_t (*malloc)( void*, void**, size_t,         cudaStream_t);
   cudaError_t (*realloc)(void*, void**, size_t, size_t, cudaStream_t);
   cudaError_t (*free)(void*, void*);
   cudaError_t (*emptyCache)(void*);
   cudaError_t  (*cacheInfo)(void*, int, size_t*, size_t*);
   void* state;
} THCDeviceAllocator;

typedef struct _THCCudaResourcesPerDevice {
  THCStream** streams;
  /* Number of materialized cuBLAS handles */
  int numBlasHandles;
  /* Number of materialized cuSparse handles */
  int numSparseHandles;
  /* cuBLAS handes are lazily initialized */
  cublasHandle_t* blasHandles;
  /* cuSparse handes are lazily initialized */
  cusparseHandle_t* sparseHandles;
  /* Size of scratch space per each stream on this device available */
  size_t scratchSpacePerStream;
  /* Device-resident scratch space per stream, used for global memory
     reduction kernels. Lazily initialized. */
  void** devScratchSpacePerStream;
} THCCudaResourcesPerDevice;


/* Global state to be held in the cutorch table. */
struct THCState {
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  /* Set of all allocated resources. resourcePerDevice[dev]->streams[0] is NULL,
     which specifies the per-device default stream. blasHandles and
     sparseHandles do not have a default and must be explicitly initialized.
     We always initialize 1 blasHandle and 1 sparseHandle but we can use more.
  */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;
  /* Number of Torch defined resources available, indices 1 ... numStreams */
  int numUserStreams;
  int numUserBlasHandles;
  int numUserSparseHandles;

  /* Allocator using cudaMallocHost. */
  THAllocator* cudaHostAllocator;
  THAllocator* cudaUVAAllocator;
  THCDeviceAllocator* cudaDeviceAllocator;

  /* Index of the current selected BLAS handle. The actual BLAS handle used
     depends on the current device. */
  THCThreadLocal/*<int>*/ currentPerDeviceBlasHandle;
  /* Index of the current selected sparse handle. The actual sparse handle used
     depends on the current device. */
  THCThreadLocal/*<int>*/ currentPerDeviceSparseHandle;
  /* Array of thread locals containing the current stream for each device */
  THCThreadLocal* currentStreams;

  /* Table of enabled peer-to-peer access between directed pairs of GPUs.
     If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;

  /* Is direct cross-kernel p2p access allowed? Normally, only cross-GPU
     copies are allowed via p2p if p2p access is enabled at all for
     the pair of GPUs in question, but if this flag is true, then
     all cross-GPU access checks are disabled, allowing kernels to
     directly access memory on another GPUs.
     Note that p2p access must exist and be enabled for the pair of
     GPUs in question. */
  int p2pKernelAccessEnabled;

  void (*cutorchGCFunction)(void *data);
  void *cutorchGCData;
  ptrdiff_t heapSoftmax;
  ptrdiff_t heapDelta;
};

THC_API THCState* THCState_alloc(void);
THC_API void THCState_free(THCState* state);

THC_API void THCudaInit(THCState* state);
THC_API void THCudaShutdown(THCState* state);

/* If device `dev` can access allocations on device `devToAccess`, this will return */
/* 1; otherwise, 0. */
THC_API int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess);
/* Enables or disables allowed p2p access using cutorch copy. If we are */
/* attempting to enable access, throws an error if CUDA cannot enable p2p */
/* access. */
THC_API void THCState_setPeerToPeerAccess(THCState* state, int dev, int devToAccess,
                                          int enable);

/* By default, direct in-kernel access to memory on remote GPUs is
   disabled. When set, this allows direct in-kernel access to remote
   GPUs where GPU/GPU p2p access is enabled and allowed. */
THC_API int THCState_getKernelPeerToPeerAccessEnabled(THCState* state);
THC_API void THCState_setKernelPeerToPeerAccessEnabled(THCState* state, int val);

THC_API struct cudaDeviceProp* THCState_getCurrentDeviceProperties(THCState* state);
THC_API struct cudaDeviceProp* THCState_getDeviceProperties(THCState* state, int device);

THC_API struct THCRNGState* THCState_getRngState(THCState* state);
THC_API THAllocator* THCState_getCudaHostAllocator(THCState* state);
THC_API THAllocator* THCState_getCudaUVAAllocator(THCState* state);
THC_API THCDeviceAllocator* THCState_getDeviceAllocator(THCState* state);
THC_API void THCState_setDeviceAllocator(THCState* state, THCDeviceAllocator* allocator);
THC_API int THCState_isCachingAllocatorEnabled(THCState* state);

THC_API void THCMagma_init(THCState *state);

/* State manipulators and accessors */
THC_API int THCState_getNumDevices(THCState* state);
THC_API void THCState_reserveStreams(THCState* state, int numStreams, int nonBlocking);
THC_API int THCState_getNumStreams(THCState* state);

/* Stream API */
THC_API cudaStream_t THCState_getCurrentStreamOnDevice(THCState *state, int device);
THC_API cudaStream_t THCState_getCurrentStream(THCState *state);
THC_API struct THCStream* THCState_getStream(THCState *state);
THC_API void THCState_setStream(THCState *state, struct THCStream* stream);
/* deprecated stream API */
THC_API cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream);
THC_API int THCState_getCurrentStreamIndex(THCState *state);
THC_API void THCState_setCurrentStreamIndex(THCState *state, int stream);

THC_API void THCState_reserveBlasHandles(THCState* state, int numHandles);
THC_API int THCState_getNumBlasHandles(THCState* state);

THC_API void THCState_reserveSparseHandles(THCState* state, int numHandles);
THC_API int THCState_getNumSparseHandles(THCState* state);

THC_API cublasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle);
THC_API cublasHandle_t THCState_getCurrentBlasHandle(THCState *state);
THC_API int THCState_getCurrentBlasHandleIndex(THCState *state);
THC_API void THCState_setCurrentBlasHandleIndex(THCState *state, int handle);

THC_API cusparseHandle_t THCState_getDeviceSparseHandle(THCState *state, int device, int handle);
THC_API cusparseHandle_t THCState_getCurrentSparseHandle(THCState *state);
THC_API int THCState_getCurrentSparseHandleIndex(THCState *state);
THC_API void THCState_setCurrentSparseHandleIndex(THCState *state, int handle);

/* For the current device and stream, returns the allocated scratch space */
THC_API void* THCState_getCurrentDeviceScratchSpace(THCState* state);
THC_API void* THCState_getDeviceScratchSpace(THCState* state, int device, int stream);
THC_API size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state);
THC_API size_t THCState_getDeviceScratchSpaceSize(THCState* state, int device);

#define THCAssertSameGPU(expr) if (!expr) THError("arguments are located on different GPUs")
#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCudaCheckWarn(err)  __THCudaCheckWarn(err, __FILE__, __LINE__)
#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)
#define THCusparseCheck(err)  __THCusparseCheck(err,  __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCudaCheckWarn(cudaError_t err, const char *file, const int line);
THC_API void __THCublasCheck(cublasStatus_t status, const char *file, const int line);
THC_API void __THCusparseCheck(cusparseStatus_t status, const char *file, const int line);

THC_API cudaError_t THCudaMalloc(THCState *state, void **ptr, size_t size);
THC_API cudaError_t THCudaFree(THCState *state, void *ptr);
THC_API void* THCudaHostAlloc(THCState *state, size_t size);
THC_API void THCudaHostFree(THCState *state, void *ptr);
THC_API void THCudaHostRecord(THCState *state, void *ptr);

THC_API cudaError_t THCudaMemGetInfo(THCState *state, size_t* freeBytes, size_t* totalBytes);
THC_API cudaError_t THCudaMemGetInfoCached(THCState *state, size_t* freeBytes, size_t* totalBytes, size_t* largestBlock);
THC_API void THCSetGCHandler(THCState *state,
                             void (*torchGCHandlerFunction)(void *data),
                             void *data );

#endif
