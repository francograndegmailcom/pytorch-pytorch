#include "ProcessGroupNCCL.hpp"
#include "private/CUDAUtils.hpp"

#include <THC.h>

#include <map>
#include <unordered_set>

namespace c10d {

namespace {

// NCCL op mapping
std::map<ReduceOp, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
};

// NCCL type typing
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
};

// Helper function that gets the data type and issues error if not supported
ncclDataType_t getNcclDataType(at::ScalarType type) {
  try {
    return ncclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for NCCL process group");
  }
}

// Helper function that gets the device list to determine the CUDA devices
std::vector<int> getDevicesFromKey(const std::string& deviceSeq) {
  std::stringstream ss(deviceSeq);
  std::string device;
  std::vector<int> devices;
  while (std::getline(ss, device, ',')) {
    devices.push_back(stoi(device));
  }
  return devices;
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<int>& devices) {
  std::string deviceList;
  for (auto device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device);
    } else {
      deviceList += "," + std::to_string(device);
    }
  }
  return deviceList;
}

// Get the list of devices from list of tensors
std::vector<int> getDevicesOfTensors(const std::vector<at::Tensor>& tensors) {
  std::vector<int> res;
  for (auto& tensor : tensors) {
    res.push_back(tensor.get_device());
  }
  return res;
}

// Helper that lets the input ncclStreams to wait for the THC stream
void syncStreams(
    THCState* thcState,
    const std::vector<int>& devices,
    std::vector<CUDAEvent>& ncclEvents,
    std::vector<CUDAStream>& ncclStreams) {
  CUDADevice gpuGuard;
  for (auto i = 0; i < devices.size(); ++i) {
    gpuGuard.setDevice(devices[i]);
    auto currentThcStream = THCState_getCurrentStream(thcState);
    CUDAStream& ncclStream = ncclStreams[i];
    CUDAEvent& ncclEvent = ncclEvents[i];

    C10D_CUDA_CHECK(cudaEventRecord(ncclEvent.getEvent(), currentThcStream));
    C10D_CUDA_CHECK(
        cudaStreamWaitEvent(ncclStream.getStream(), ncclEvent.getEvent(), 0));
  }
}

} // namespace

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const std::vector<int>& devices)
    : devices_(devices) {
  // GPU deviceGuard
  CUDADevice gpuGuard;
  cudaEvents_.resize(devices.size());
  // Now create the CUDA events
  for (auto i = 0; i < devices.size(); ++i) {
    // Guard the GPU
    gpuGuard.setDevice(devices[i]);
    cudaEvents_[i] = CUDAEvent::create(cudaEventDisableTiming);
  }
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() {}

// Checking the work's corresponding CUDA events' status
bool ProcessGroupNCCL::WorkNCCL::isCompleted() const {
  CUDADevice gpuGuard;
  for (auto i = 0; i < devices_.size(); ++i) {
    // Get to the device
    gpuGuard.setDevice(devices_[i]);
    auto& cudaEvent = cudaEvents_[i];
    // Query the event
    auto ret = cudaEventQuery(cudaEvent.getEvent());
    if (ret != cudaSuccess && ret != cudaErrorNotReady) {
      // Throw directly
      C10D_CUDA_CHECK(ret);
    }
    if (ret == cudaErrorNotReady) {
      return false;
    }
  }
  return true;
}

// Same as synchronize(), and will always return true
bool ProcessGroupNCCL::WorkNCCL::wait() {
  synchronize();
  return true;
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupNCCL::WorkNCCL::synchronize() {
  auto thcState = ::at::globalContext().lazyInitCUDA();
  CUDADevice gpuGuard;
  for (auto i = 0; i < devices_.size(); ++i) {
    // Get to the device
    gpuGuard.setDevice(devices_[i]);
    auto thcStream = THCState_getCurrentStream(thcState);
    auto& cudaEvent = cudaEvents_[i];
    // Let THC stream wait for the NCCL stream
    C10D_CUDA_CHECK(cudaStreamWaitEvent(thcStream, cudaEvent.getEvent(), 0));
  }
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  throw std::runtime_error(
      "isSuccess() is not supported by NCCL process "
      "group's work, isCompleted() and wait() will "
      "either succeed or throw");
}

const std::exception& ProcessGroupNCCL::WorkNCCL::exception() const {
  throw std::runtime_error(
      "exception() is not supported by NCCL process "
      "group's work, isCompleted() and wait() will "
      "either succeed or throw");
}

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::shared_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store) {
  C10D_CUDA_CHECK(cudaGetDeviceCount(&numGPUs_));
  thcState_ = ::at::globalContext().lazyInitCUDA();
}

ProcessGroupNCCL::~ProcessGroupNCCL() {}

void ProcessGroupNCCL::broadcastUniqueNCCLId(
    const std::string& devicesKey,
    ncclUniqueId* ncclId) {
  // Rank 0 writes to the store as bcast
  if (rank_ == 0) {
    auto ncclIdVal = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclId),
        reinterpret_cast<uint8_t*>(ncclId) + NCCL_UNIQUE_ID_BYTES);
    store_->set(devicesKey, ncclIdVal);
    // Other ranks get to the store
  } else {
    auto ncclIdVal = store_->get(devicesKey);
    // Just a sanity check
    if (ncclIdVal.size() != NCCL_UNIQUE_ID_BYTES) {
      throw std::runtime_error(
          "Unexpected NCCL unique ID length received "
          "from the store");
    }
    // Now put the data back to the input pointer
    memcpy(ncclId, ncclIdVal.data(), NCCL_UNIQUE_ID_BYTES);
  }
}

std::vector<std::shared_ptr<NCCLComm>>& ProcessGroupNCCL::getNCCLComm(
    const std::string& devicesKey,
    const std::vector<int>& devices) {
  // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
        "Not able to create/get the Nccl Comm since "
        "the GPU devices are not known");
  }
  if (devNCCLCommMap_.find(devicesKey) != devNCCLCommMap_.end()) {
    // Reuse the cached communicator if there is one.
    return devNCCLCommMap_[devicesKey];
  }
  // NCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<NCCLComm>> ncclComms;
  ncclComms.resize(devices.size());

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclId;

  if (rank_ == 0) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&ncclId));
  }

  // Broadcast so that each process can have a unique NCCL ID
  broadcastUniqueNCCLId(devicesKey, &ncclId);

  // GPU deviceGuard
  CUDADevice gpuGuard;

  std::vector<CUDAEvent> eventVal;
  std::vector<CUDAStream> streamVal;

  eventVal.resize(devices.size());
  streamVal.resize(devices.size());

  // Create the NCCL communicators for each GPU
  C10D_NCCL_CHECK(ncclGroupStart());

  for (auto i = 0; i < devices.size(); ++i) {
    // GPU world size and GPU rank
    int numRanks = getSize() * devices.size();
    int rank = getRank() * devices.size() + i;

    // Guard the GPU
    gpuGuard.setDevice(devices[i]);
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclId);

    // Also create the NCCL streams and events
    streamVal[i] = CUDAStream::create();
    eventVal[i] = CUDAEvent::create(cudaEventDisableTiming);
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Move the NCCL resource to cache
  devNCCLCommMap_.emplace(devicesKey, std::move(ncclComms));
  ncclStreams_.emplace(devicesKey, std::move(streamVal));
  ncclEvents_.emplace(devicesKey, std::move(eventVal));

  return devNCCLCommMap_[devicesKey];
}

// Helper function that checks the input and output tensors for validity
void ProcessGroupNCCL::tensorCheckHelper(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    size_t outputOverInput) {
  if (input.size() != output.size()) {
    throw std::runtime_error(
        "Input tensor sequence should have the same "
        "number of tensors as the output tensor sequence");
  }

  if (input.size() == 0) {
    throw std::runtime_error("The number of input tensors should not be zero");
  }

  if (input.size() > numGPUs_) {
    throw std::runtime_error(
        "The number of input tensors is larger than "
        "the number of available GPUs");
  }

  // To make sure each tensor is on separate devices
  std::unordered_set<int> usedDevices;
  usedDevices.reserve(input.size());

  uint64_t inputNumElement = input[0].numel();
  auto elementType = input[0].type().scalarType();

  for (auto i = 0; i < input.size(); ++i) {
    //  Check to make sure it's a GPU dense tensor
    if (!(input[i].type().is_cuda() && !input[i].type().is_sparse() &&
          output[i].type().is_cuda() && !output[i].type().is_sparse())) {
      throw std::runtime_error(
          "Only CUDA dense tensor is supported for NCCL "
          "collective operations");
    }
    // Check the tensor type is identical
    if (input[i].type().scalarType() != elementType ||
        output[i].type().scalarType() != elementType) {
      throw std::runtime_error(
          "Expecting all GPU tensors to have identical "
          "type");
    }
    // Check the input tensor size is identical
    if (input[i].numel() != inputNumElement) {
      throw std::runtime_error(
          "Expecting all input tensors to have identical "
          "number of elements");
    }
    // Check the output tensor size equals to input tensor size
    if (output[i].numel() != inputNumElement * outputOverInput) {
      throw std::runtime_error(
          "The number of elements of output tensor does "
          "not match the number of elements of the input "
          "tensor");
    }
    // Contiguous verification
    if (!input[i].is_contiguous() || !output[i].is_contiguous()) {
      throw std::runtime_error("Expecting all GPU tensors to be contiguous");
    }

    bool inserted;
    std::tie(std::ignore, inserted) = usedDevices.insert(input[i].get_device());
    // Device verification, if the insertion didn't take place
    if (!inserted) {
      throw std::runtime_error("Expecting inputs on different GPU devices");
    }

    // Now check the output device
    if (input[i].get_device() != output[i].get_device()) {
      throw std::runtime_error(
          "Expecting input and output tensors to be on "
          "the same device");
    }
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  tensorCheckHelper(tensors, tensors);

  auto devices = getDevicesOfTensors(tensors);
  auto key = getKeyFromDevices(devices);
  // Get the NCCL communicators
  auto& ncclComms = getNCCLComm(key, devices);

  // First let NCCL streams wait for THC stream
  syncStreams(thcState_, devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  // Guard GPU device
  CUDADevice gpuGuard;

  // Queue the NCCL kernel
  C10D_NCCL_CHECK(ncclGroupStart());

  for (auto i = 0; i < tensors.size(); ++i) {
    // Get to the device
    gpuGuard.setDevice(devices[i]);
    // Use the NCCL stream
    CUDAStream& ncclStream = ncclStreams_[key][i];

    C10D_NCCL_CHECK(ncclAllReduce(
        tensors[i].data_ptr(),
        tensors[i].data_ptr(),
        tensors[i].numel(),
        getNcclDataType(tensors[i].type().scalarType()),
        ncclOp[opts.reduceOp],
        ncclComms[i]->getNcclComm(),
        ncclStream.getStream()));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Event should only be recorded after the ncclGroupEnd()
  for (auto i = 0; i < tensors.size(); ++i) {
    CUDAStream& ncclStream = ncclStreams_[key][i];
    CUDAEvent& cudaEvent = work->cudaEvents_[i];

    C10D_CUDA_CHECK(
        cudaEventRecord(cudaEvent.getEvent(), ncclStream.getStream()));
  }

  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  tensorCheckHelper(tensors, tensors);

  auto devices = getDevicesOfTensors(tensors);
  auto key = getKeyFromDevices(devices);
  // Get the NCCL communicators
  auto& ncclComms = getNCCLComm(key, devices);
  // First let NCCL streams wait for THC stream
  syncStreams(thcState_, devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  // Guard GPU device
  CUDADevice gpuGuard;

  // Queue the NCCL kernel
  C10D_NCCL_CHECK(ncclGroupStart());

  for (auto i = 0; i < tensors.size(); ++i) {
    // Get to the device
    gpuGuard.setDevice(devices[i]);
    // Use the NCCL stream
    CUDAStream& ncclStream = ncclStreams_[key][i];
    // root rank of the the GPU
    int root = opts.rootRank * tensors.size() + opts.rootTensor;

    C10D_NCCL_CHECK(ncclBcast(
        tensors[i].data_ptr(),
        tensors[i].numel(),
        getNcclDataType(tensors[i].type().scalarType()),
        root,
        ncclComms[i]->getNcclComm(),
        ncclStream.getStream()));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Event should only be recorded after the ncclGroupEnd()
  for (auto i = 0; i < tensors.size(); ++i) {
    CUDAStream& ncclStream = ncclStreams_[key][i];
    CUDAEvent& cudaEvent = work->cudaEvents_[i];

    C10D_CUDA_CHECK(
        cudaEventRecord(cudaEvent.getEvent(), ncclStream.getStream()));
  }

  return work;
}

} // namespace c10d
