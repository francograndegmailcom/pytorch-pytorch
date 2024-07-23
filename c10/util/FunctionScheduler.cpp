#include <c10/util/FunctionScheduler.h>

#include <iostream>

namespace c10 {

/* Job */

Job::Job(
    std::function<void()> function,
    std::chrono::microseconds interval,
    bool immediate,
    int run_limit)
    : _function(std::move(function)),
      _interval(interval),
      _run_limit(run_limit),
      _immediate(immediate) {}

void Job::run() {
  ++_counter;
  try {
    _function();
  } catch (const std::exception& e) {
    std::cerr << "Job failed: " << e.what() << std::endl;
  }
}

/* Run */

Run::Run(int job_id, std::chrono::time_point<std::chrono::steady_clock> time)
    : _job_id(job_id), _time(time) {}

/* FunctionScheduler */

FunctionScheduler::FunctionScheduler() = default;

FunctionScheduler::~FunctionScheduler() {
  stop();
}

std::chrono::microseconds FunctionScheduler::getNextWaitTime() {
  // We can't pop the next run instantly,
  // as it may still change while we're waiting.
  _next_run = _queue.front();

  // Finding the first run associated with an active job.
  auto entry = _jobs.find(_next_run.job_id());
  while (!validEntry(entry)) {
    // Only pop runs associated with an invalid job.
    std::pop_heap(_queue.begin(), _queue.end(), Run::gt);
    _queue.pop_back();

    if (_queue.empty())
      return std::chrono::microseconds(-1);

    _next_run = _queue.front();
    entry = _jobs.find(_next_run.job_id());
  }

  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
      _next_run.time() - now);
}

void FunctionScheduler::run() {
  std::unique_lock<std::mutex> lock(_mutex);

  while (_running) {
    if (_queue.empty() || _paused || _dirty) {
      _cond.wait(lock);
      continue;
    }

    std::chrono::microseconds wait_time = getNextWaitTime();
    // Check again if queue is empty after pops
    if (_queue.empty())
      continue;

    if (wait_time.count() <= 0) {
      runNextJob(lock);
      continue;
    }

    // Waiting for the next run to be ready.
    // We need to wake up if a new run is added
    // to the queue, as it may need to happen
    // before the current ´_next_run´. We also
    // need to wake up if ´_paused´ changes, to
    // pause execution.
    if (_cond.wait_for(lock, wait_time) == std::cv_status::timeout) {
      // Lock timed out, i.e., nothing happened while we waited.
      // The run selected as next is still the correct one and we
      // aren't paused.
      runNextJob(lock);
    }
  }
}

void FunctionScheduler::runNextJob(const std::unique_lock<std::mutex>& lock) {
  // This function is always called with the mutex previously acquired.
  TORCH_INTERNAL_ASSERT(lock.owns_lock(), "Mutex not acquired");
  TORCH_INTERNAL_ASSERT(
      _next_run == _queue.front(), "Next run does not match queue top.");

  // Remove this run from the queue
  std::pop_heap(_queue.begin(), _queue.end(), Run::gt);
  _queue.pop_back();

  // Check if the job was canceled in the meantime.
  auto entry = _jobs.find(_next_run.job_id());
  if (validEntry(entry)) {
    entry->second.run();
    // Add a new run associated with this job to the queue
    addRun(lock, entry->first, entry->second);
  }
}

bool FunctionScheduler::validEntry(
    const std::unordered_map<int, Job>::iterator& entry) {
  return entry != _jobs.end() &&
      entry->second.counter() != entry->second.run_limit();
}

void FunctionScheduler::addRun(int job_id, const Job& job) {
  // We can only call addRun without a mutex locked if we are not yet running.
  TORCH_INTERNAL_ASSERT(
      !_running, "Function called without a mutex while scheduler is running");
  addRunInternal(job_id, job);
}

void FunctionScheduler::addRun(
    const std::unique_lock<std::mutex>& lock,
    int job_id,
    const Job& job) {
  // This function is always called with the mutex previously acquired.
  TORCH_INTERNAL_ASSERT(lock.owns_lock(), "Mutex not acquired");
  addRunInternal(job_id, job);
}

void FunctionScheduler::addRunInternal(int job_id, const Job& job) {
  // This function should not be called directly, use addRun instead.

  auto interval = job.interval();
  if (job.immediate() && job.counter() == 0)
    interval = std::chrono::microseconds(0);

  auto time = std::chrono::steady_clock::now() + interval;

  _queue.emplace_back(job_id, time);
  std::push_heap(_queue.begin(), _queue.end(), Run::gt);
}

int FunctionScheduler::scheduleJob(const Job& job) {
  ++_dirty;
  std::unique_lock<std::mutex> lock(_mutex);
  int job_id = id();

  if (_running)
    addRun(lock, job_id, job);

  _jobs.emplace(job_id, job);
  --_dirty;
  _cond.notify_one();
  return job_id;
}

bool FunctionScheduler::removeJob(int id) {
  ++_dirty;
  std::lock_guard<std::mutex> lock(_mutex);
  --_dirty;
  _cond.notify_one();

  // The scheduler checks if the job associated
  // with a run is valid, so, to cancel a job
  // and it's run, we just need to erase
  // it (thus making it invalid).
  return _jobs.erase(id);
}

bool FunctionScheduler::start() {
  if (_running || _paused)
    return false;

  for (const auto& entry : _jobs) {
    addRun(entry.first, entry.second);
  }

  _running = true;
  _paused = false;
  _thread = std::thread(&FunctionScheduler::run, this);
  return true;
}

bool FunctionScheduler::stop() {
  if (!_running)
    return false;

  _running = false;
  _paused = false;
  // Unblock the thread executing
  // `FunctionScheduler::run` so it
  // exits the loop.
  _cond.notify_one();
  if (_thread.joinable()) {
    _thread.join();
  }

  // clear queue
  _queue.clear();

  // reset counters
  for (auto& entry : _jobs) {
    entry.second.reset_counter();
  }
  return true;
}

bool FunctionScheduler::pause() {
  if (_paused || !_running)
    return false;

  _paused_time = std::chrono::steady_clock::now();
  _paused = true;
  return true;
}

bool FunctionScheduler::resume() {
  if (!_paused)
    return false;

  // Since we're shifting the time of all elements by the same amount
  // the min-heap is still valid, no need to rebuild it.
  auto diff = std::chrono::steady_clock::now() - _paused_time;
  for (auto& run : _queue) {
    run.set_time(run.time() + diff);
  }

  _paused = false;

  // Unblock the thread executing
  // `FunctionScheduler::run` so it
  // continues execution.
  _cond.notify_one();

  return true;
}

} // namespace c10
