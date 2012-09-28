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
/// \file
/// Candidate and Candidate::Comparator factory implementation.
/// Author: dbikel@google.com (Dan Bikel)

#include "candidate.H"

namespace reranker {

IMPLEMENT_FACTORY(Candidate::Comparator)

bool Candidate::Compile(Symbols *symbols,
                        bool clear_features,
                        bool clear_symbolic_features,
                        bool force) {
  bool compiled_this_invocation = false;
  if ((!compiled_ || force) && !symbolic_features_.empty()) {
    if (clear_features) {
      features_.clear();
    }
    for (FeatureVector<string, double>::const_iterator it =
             symbolic_features_.begin();
         it != symbolic_features_.end();
         ++it) {
      features_.IncrementWeight(symbols->GetIndex(it->first), it->second);
      // Only gets set to true if we compile at least one symbolic feature.
      compiled_ = true;
      compiled_this_invocation = true;
    }
    if (clear_symbolic_features) {
      symbolic_features_.clear();
    }
  }
  return compiled_this_invocation;
}

void Candidate::Decompile(Symbols *symbols,
                          bool clear_symbolic_features,
                          bool clear_features,
                          bool force) {
  if (compiled_ || force) {
    if (clear_symbolic_features) {
      symbolic_features_.clear();
    }
    for (FeatureVector<int, double>::const_iterator it =
             features_.begin();
         it != features_.end();
         ++it) {
      symbolic_features_.IncrementWeight(symbols->GetSymbol(it->first),
                                         it->second);
    }
    if (clear_features) {
      features_.clear();
    }
  }
  compiled_ = false;
}

}  // namespace reranker
