#ifndef CAFFE2_CORE_NET_GPU_H_
#define CAFFE2_CORE_NET_GPU_H_

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/event.h"
#include "caffe2/core/net.h"

namespace caffe2 {

// Run an event-driven graph - before each operator chain, wait on each parent
// operator for the chain source, then execute each operator. Due to the chain
// construction mechanism, operators in the same chain implicitly runs on the
// same stream.
class AsyncDAGNet : public DAGNetBase {
 public:
  AsyncDAGNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  bool RunAt(const std::vector<int>& chain) override;
  bool Run() override;

 protected:
  // Tracks whether a given op has had an event recorded in each
  // RunAt() iteration.
  std::vector<int32_t> eventRecorded_;
  DISABLE_COPY_AND_ASSIGN(AsyncDAGNet);
};

namespace gpu_single_thread {

struct Task {
  std::vector<std::unique_ptr<OperatorBase>>* ops_;
  std::condition_variable* cv_;
  std::mutex* mtx_;
  int stream_id_;
  bool done_ = false;
};

class GPUExecutor {
 public:
  explicit GPUExecutor(int gpu_id) : gpu_id_(gpu_id) {}

  ~GPUExecutor() {
    queue_.NoMoreJobs();
    thread_.join();
  }

  void RunJob(Task* task) {
    queue_.Push(task);
  }

  void start() {
    thread_ = std::thread(&GPUExecutor::WorkerFunction, this);
  }

  static std::shared_ptr<GPUExecutor> Get(int gpu);
  static void Release(int gpu);

 private:
  void set_affinity();
  void WorkerFunction();

  std::thread thread_;
  int gpu_id_;
  SimpleQueue<Task*> queue_;
  static std::shared_ptr<GPUExecutor> executors_[CAFFE2_COMPILE_TIME_MAX_GPUS];
  static std::mutex gpu_mtx_[CAFFE2_COMPILE_TIME_MAX_GPUS];
};
}

} // namespace caffe2

#endif // CAFFE2_CORE_NET_GPU_H_
