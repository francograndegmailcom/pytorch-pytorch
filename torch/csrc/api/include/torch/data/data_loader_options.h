#pragma once

#include <torch/arg.h>

#include <ATen/optional.h>

#include <chrono>
#include <cstddef>

namespace torch {
namespace data {
namespace detail {
struct FullDataLoaderOptions;
} // namespace detail
} // namespace data
} // namespace torch

namespace torch {
namespace data {

/// Options to configure a `DataLoader`.
struct DataLoaderOptions {
  friend struct torch::data::detail::FullDataLoaderOptions;

  DataLoaderOptions() = default;
  /* implicit */ DataLoaderOptions(size_t batch_size)
      : batch_size_(batch_size) {}

  /// The size of each batch to fetch.
  TORCH_ARG(size_t, batch_size) = 1;

  /// Whether to omit the last batch if it contains less than `batch_size`
  /// examples.
  TORCH_ARG(bool, drop_last) = false;

  /// The number of worker threads to launch. If zero, the main thread will
  /// synchronously perform the data loading.
  TORCH_ARG(size_t, workers) = 0;

  /// The maximum number of jobs to enqueue for fetching by worker threads.
  /// Defaults to two times the number of worker threads.
  TORCH_ARG(at::optional<size_t>, max_jobs);

  /// An optional limit on the time to wait for the next batch.
  TORCH_ARG(at::optional<std::chrono::milliseconds>, timeout);

  /// Whether to enforce ordering of batches when multiple are loaded
  /// asynchronously by worker threads. Set to `false` for better performance if
  /// you do not care about determinism.
  TORCH_ARG(bool, enforce_ordering) = true;
};

/// Like `DataLoaderOptions`, but without any unconfigured state.
/// `DataLoaderOptions` has some options that depend on other options
/// (`max_jobs` => `2 * workers`). In the spirit of properly using the C++ type
/// system, `DataLoaderOptions` allows only setting values. To access values,
/// you must create a `FullDataLoaderOptions` from a `DataLoaderOptions`
/// instance, which will do any necessary coalescing.
struct FullDataLoaderOptions {
  explicit FullDataLoaderOptions(DataLoaderOptions options)
      : batch_size(options.batch_size_),
        drop_last(options.drop_last_),
        workers(options.workers_),
        max_jobs(options.max_jobs_.value_or(2 * workers)),
        timeout(options.timeout_),
        enforce_ordering(options.enforce_ordering_) {}

  size_t batch_size;
  bool drop_last;
  size_t workers;
  size_t max_jobs;
  at::optional<std::chrono::milliseconds> timeout;
  bool enforce_ordering;
};
} // namespace data
} // namespace torch
