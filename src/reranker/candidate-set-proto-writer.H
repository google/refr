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
/// \file candidate-set-proto-writer.H
/// Serializer for reranker::CandidateSet instances to CandidateSetMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_SET_PROTO_WRITER_H_
#define RERANKER_CANDIDATE_SET_PROTO_WRITER_H_

#include <tr1/unordered_map>
#include <vector>

#include "../proto/data.pb.h"
#include "../proto/model.pb.h"
#include "candidate.H"
#include "candidate-set.H"
#include "feature-vector-writer.H"

namespace reranker {

using std::tr1::unordered_map;

using confusion_learning::CandidateSetMessage;
using confusion_learning::CandidateMessage;
using confusion_learning::FeatureVecMessage;
using confusion_learning::FeatureMessage;
using confusion_learning::ScoreMessage;

/// \class CandidateSetProtoWriter
///
/// A class to construct a CandidateSetMessage from a CandidateSet instance.
class CandidateSetProtoWriter {
 public:
  /// Constructs a new instance that can serialize CandidateSet instances to
  /// CandidateSetMessage protocol buffer messages.
  CandidateSetProtoWriter() { }
  /// Destroys this writer.
  virtual ~CandidateSetProtoWriter() { }

  /// Serializes a CandidateSet instance to a CandidateSetMessage.
  ///
  /// \param set                   the CandidateSet to be serialized to the
  ///                              specified CandidateSetMessage
  /// \param candidate_set_message the CandidateSetMessage to be filled in
  ///                              by this method with the serialized version
  ///                              of the specified CandidateSet
  void Write(const CandidateSet &set,
             CandidateSetMessage *candidate_set_message) const;
 private:
  FeatureVectorWriter<FeatureVector<int,double> > fv_writer_;
  FeatureVectorWriter<FeatureVector<string,double> > symbolic_fv_writer_;
};

}  // namespace reranker

#endif
