// Copyright 2012, Google Inc.
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above
//     copyright notice, this list of conditions and the following disclaimer
//     in the documentation and/or other materials provided with the
//     distribution.
//   * Neither the name of Google Inc. nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,           
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------
//
//
/// \file training-time.H
/// Provides the reranker::Time class, which holds the three notions of
/// training time: current epoch, current time step within the current epoch
/// and absolute time step.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_TRAINING_TIME_H_
#define RERANKER_TRAINING_TIME_H_

#include <ctime>
#include <iostream>
#include <string>
#include <sstream>

namespace reranker {

using std::string;
using std::ostream;

/// \class Time
///
/// A simple class to hold the three notions of time during training:
/// the current epoch, the current time index within the current epoch,
/// and the absolute time index.
class Time {
 public:
  /// Constructs a new time instance, ready for its indices to be incremented.
  Time() : epoch_(-1), index_(-1), absolute_index_(-1),
           absolute_time_(clock()), time_since_last_epoch_(absolute_time_) { }
  /// Constructs a new time instance with the specified time indices.
  ///
  /// \param epoch          the current epoch
  /// \param index          the current time index within the current epoch
  /// \param absolute_index the absolute time index
  Time(int epoch, int index, int absolute_index) :
      epoch_(epoch),
      index_(index),
      absolute_index_(absolute_index),
      absolute_time_(clock()), time_since_last_epoch_(absolute_time_) { }

  // accessors
  /// Returns the index of the current epoch.
  int epoch() const { return epoch_; }
  /// Returns the index of the current training example within the
  /// current epoch.
  int index() const { return index_; }
  /// Returns the total number of training examples seen in all epochs.
  int absolute_index() const { return absolute_index_; }

  double absolute_seconds() const {
    clock_t diff_time = clock() - absolute_time_;
    return ((double)diff_time) / CLOCKS_PER_SEC;
  }
  double seconds_since_last_epoch() const {
    clock_t diff_time = clock() - time_since_last_epoch_;
    return ((double)diff_time) / CLOCKS_PER_SEC;
  }

  friend ostream &operator<<(ostream &os, const Time &time) {
    return os << time.epoch() << "," << time.index() << ";"
              << time.absolute_index();
  }

  string to_string() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  // mutators
  /// Increments both the time index for the current epoch and the
  /// absolute time index.
  void Tick() {
    index_++;
    absolute_index_++;
  }
  /// Increments the epoch counter.
  void NewEpoch() {
    epoch_++;
    index_ = -1;
    time_since_last_epoch_ = clock();
  }
 private:
  int epoch_;
  int index_;
  int absolute_index_;
  clock_t absolute_time_;
  clock_t time_since_last_epoch_;
};

}  // namespace reranker

#endif
