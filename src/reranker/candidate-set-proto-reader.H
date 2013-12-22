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
/// \file candidate-set-proto-reader.H
/// Reads CandidateSetMessage instances and converts them to
/// reranker::CandidateSet instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_SET_PROTO_READER_H_
#define RERANKER_CANDIDATE_SET_PROTO_READER_H_

#include <unordered_map>
#include <vector>

#include "candidate.H"
#include "candidate-set.H"
#include "string-canonicalizer.H"
#include "tokenizer.H"
#include "../proto/data.pb.h"
#include "../proto/model.pb.h"

namespace reranker {

using std::unordered_map;

using confusion_learning::CandidateSetMessage;

/// \class CandidateSetProtoReader
///
/// A class to fill in a CandidateSet based on a CandidateSetMessage,
/// crucially constructing new Candidate instances from each
/// CandidateMessage inside the CandidateSetMessage.
class CandidateSetProtoReader {
 public:
  CandidateSetProtoReader() { }
  virtual ~CandidateSetProtoReader() { }

  /// Fills in the specified CandidateSet based on the specified
  /// CandidateSetMessage, crucially constructing new Candidate instances
  /// from each CandidateMessage inside the CandidateSetMessage.
  ///
  /// \param[in]  m   the message from which to fill in a CandidateSet
  /// \param[out] set the CandidateSet to be filled in based on the
  ///                 specified CandidateSetMessage
  void Read(const CandidateSetMessage &m, CandidateSet &set) {
    Read(m, -1, set);
  }

  /// Identical to read, but specifies a maximum number of candidates to be read
  /// per candidate set (for debugging).
  ///
  /// \param[in]  m              the message from which to fill in a
  ///                            CandidateSet
  /// \param[in]  max_candidates the maximum number of candidates to read
  ///                            when filling in any CandidateSet, or -1 if
  ///                            there is no maximum
  /// \param[out] set            the CandidateSet to be filled in based on the
  ///                            specified CandidateSetMessage
  void Read(const CandidateSetMessage &m, int max_candidates,
            CandidateSet &set);

  void ClearStrings() {
    StringCanonicalizer::Clear();
  }
 private:
  int CountTokens(const string &s, const char *delimiters = " \t") const {
    int count = 0;
    size_t end_pos = 0;
    size_t begin_pos = 0;
    while (begin_pos != string::npos) {
      begin_pos = s.find_first_not_of(delimiters, end_pos);
      end_pos = s.find_first_of(delimiters, begin_pos);
      if (end_pos == string::npos) {
        end_pos = s.length();
      }
      if (begin_pos != string::npos) {
        ++count;
        begin_pos = end_pos;
      }
    }
    return count;
  }

  double ComputeLoss(CandidateSet &set, const string &candidate_raw_data);

  // data members
  Tokenizer tokenizer_;
};

}  // namespace reranker

#endif
