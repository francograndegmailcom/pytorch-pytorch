#pragma once

#include <torch/tensor.h>

#include <ATen/optional.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace detail {
namespace sequencers {

/// A `Sequencer` accepts a function that yields the next result of a
/// `DataLoader` and then has the opportunity to influence the order in which
/// these results are returned. The `NoSequencer` does not enforce any
/// sequencing and returns any result directly. The `OrderedSequencer` instead
/// buffers results internally to return them in order of their sequence number.
template <typename Result>
struct Sequencer {
  using ResultProducer = std::function<optional<Result>()>;
  virtual ~Sequencer() = default;
  virtual optional<Result> next(ResultProducer next_result) = 0;
};

/// A `Sequencer` that does not enforce any ordering. It is effectively the
/// identity function.
template <typename Result>
struct NoSequencer final : public Sequencer<Result> {
  using typename Sequencer<Result>::ResultProducer;
  optional<Result> next(ResultProducer next_result) override {
    return next_result();
  }
};

/// A `Sequencer` that buffers results and returns them in order of their
/// sequence number. The `OrderedSequencer` maintains an internal, monotonically
/// incrementing counter for the next sequence number it expects. If it receives
/// a result with a higher sequence number, it will buffer it for later (when
/// the sequence number reaches that of this result). Otherwise, if the sequence
/// numbers match, the result is returned.
///
/// Implementation note: The `OrderedSequencer` is implemented with a fixed-size
/// buffer. Let `m` be the maximum number of jobs in the data loader's queue and
/// `s` be the current sequence number. Assume `m` jobs are scheduled in the
/// `DataLoader`. Any new result is stored at index `job.sqn mod m` in the
/// `OrderedSequencer`. Why are we sure sequence numbers of new jobs will not
/// collide with sequence numbers of buffered jobs? The `OrderedSequencer` will
/// not return from `next()` until it receives the result with sqn `s`. This
/// means no new jobs can be scheduled in the `DataLoader` in the meantime,
/// which enforces that as long as sqn `s` has not been received, `s + m` (which
/// would cause a collision in the fixed-size buffer) will not yet be scheduled.
template <typename Result>
struct OrderedSequencer : public Sequencer<Result> {
  using typename Sequencer<Result>::ResultProducer;

  /// Constructs the `OrderedSequencer` with the maximum number of results it
  /// will ever hold at one point in time.
  explicit OrderedSequencer(size_t max_jobs) : buffer_(max_jobs) {}

  /// Buffers results until the next one in the expected order is received.
  optional<Result> next(ResultProducer next_result) override {
    // If we already have the result for the next sqn, return it.
    if (auto& maybe_result = buffer(next_sequence_number_)) {
      auto result = std::move(*maybe_result);
      buffer(next_sequence_number_++).reset();
      return result;
    }
    // Otherwise wait for the next result.
    while (true) {
      auto result = next_result();
      if (!result) {
        break;
      }
      // If it was not nullopt and the sequence numbers match, return it
      // directly and bump the sequence number.
      if (result->sequence_number == next_sequence_number_) {
        ++next_sequence_number_;
        return result;
      }
      // Stash the result for later.
      buffer(result->sequence_number) = std::move(result);
    }
    // The result was an empty optional, so we are done with this epoch.
    return nullopt;
  }

  /// Accesses the buffer at the `index` modulo the buffer size.
  optional<Result>& buffer(size_t index) {
    return buffer_.at(index % buffer_.size());
  }

  /// The monotonically increasing sequence number we expect.
  size_t next_sequence_number_ = 0;

  /// A fixed-size buffer (after construction).
  std::vector<optional<Result>> buffer_;
};
} // namespace sequencers
} // namespace detail
} // namespace data
} // namespace torch
