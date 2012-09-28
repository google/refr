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
/// Provides the implementation of the
/// reranker::ExecutiveFeatureExtractor class.  This class executes
/// the extraction methods of a suite of reranker::FeatureExtractor
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "candidate.H"
#include "candidate-set.H"
#include "feature-extractor.H"
#include "executive-feature-extractor.H"
#include "stream-tokenizer.H"

#define DEBUG 1

using std::cerr;
using std::endl;
using std::ifstream;
using std::istream;
using std::string;
using std::vector;

namespace reranker {

void
ExecutiveFeatureExtractor::Init(const string &filename) {
  ifstream is(filename.c_str());
  Init(is);
}

void
ExecutiveFeatureExtractor::Init(istream &is) {
  StreamTokenizer st(is);
  while (st.HasNext()) {
    extractors_.push_back(factory_.CreateOrDie(st));
  }
}

void
ExecutiveFeatureExtractor::Reset() const {
  for (vector<shared_ptr<FeatureExtractor> >::const_iterator it =
           extractors_.begin();
       it != extractors_.end();
       ++it) {
    (*it)->Reset();
  }
}

void
ExecutiveFeatureExtractor::Extract(CandidateSet &candidate_set) const {
  for (vector<shared_ptr<FeatureExtractor> >::const_iterator it =
           extractors_.begin();
       it != extractors_.end();
       ++it) {
    (*it)->Extract(candidate_set);
  }
}

}  // namespace reranker
