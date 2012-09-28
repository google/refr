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
/// \file kernel-function.H
/// Provides the reranker::KernelFunction interface.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_KERNEL_FUNCTION_H_
#define RERANKER_KERNEL_FUNCTION_H_

#include "feature-vector.H"

namespace reranker {

/// \class KernelFunction
///
/// An interface specifying a kernel function for two FeatureVector instances.
/// The two index arguments can be of help to kernel function implementations
/// that either cache results, or else have access to the precomputed square
/// matrix of kernel function values for all training vectors.
///
class KernelFunction {
 public:
  /// Applies this kernel function to the specified feature vectors.
  virtual double Apply(const FeatureVector<int,double> &fv1,
                       const FeatureVector<int,double> &fv2) {
    return Apply(fv1, -1, fv2, -1);
  }

  /// Applies this kernel function to the specified feature vectors.
  ///
  /// \param fv1    the first feature vector
  /// \param index1 the training example index of the first feature vector, if
  ///               known, or else -1; concrete implementations may ignore this
  ///               argument
  /// \param fv2    the second feature vector
  /// \param index2 the training example index of the second feature vector, if
  ///               known, or else -1; concrete implementations may ignore this
  ///               argument
  virtual double Apply(const FeatureVector<int,double> &fv1, int index1,
                       const FeatureVector<int,double> &fv2, int index2)
      const = 0;
};

}  // namespace reranker

#endif
