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
/// \file candidate-set-proto-writer.C
/// Implementation for serializer for reranker::CandidateSet instances to
/// CandidateSetMessage instances.
/// \author dbikel@google.com (Dan Bikel)

#include "candidate-set-proto-writer.H"

namespace reranker {

void
CandidateSetProtoWriter::Write(const CandidateSet &set,
                               CandidateSetMessage *candidate_set_message)
    const {
  for (CandidateSet::const_iterator it = set.begin(); it != set.end(); ++it) {
    // Serialize each Candidate to a CandidateMessage.
    // TODO(dbikel): Possibly create CandidateWriter class to handle this.
    CandidateMessage *candidate_message =
        candidate_set_message->add_candidate();
    const Candidate &candidate = *(*it);
    if (candidate.raw_data() != "") {
      candidate_message->set_raw_data(candidate.raw_data());
    }

    // Add loss.
    ScoreMessage *loss_message = candidate_message->add_score();
    loss_message->set_type(ScoreMessage::LOSS);
    loss_message->set_score(candidate.loss());
    // Add model score.
    ScoreMessage *model_score_message = candidate_message->add_score();
    model_score_message->set_type(ScoreMessage::OUTPUT_SCORE);
    model_score_message->set_score(candidate.score());
    // Add baseline score.
    ScoreMessage *baseline_score_message = candidate_message->add_score();
    baseline_score_message->set_type(ScoreMessage::SYSTEM_SCORE);
    baseline_score_message->set_score(candidate.baseline_score());

    // Add features.
    FeatureVecMessage *fv_message = candidate_message->mutable_feats();
    fv_writer_.Write(candidate.features(), FeatureMessage::BASIC, fv_message);
    symbolic_fv_writer_.Write(candidate.symbolic_features(),
                              FeatureMessage::BASIC, fv_message);
  }

  // Add gold index and best-scoring indices.
  candidate_set_message->set_gold_index(set.gold_index());
  candidate_set_message->set_best_scoring_index(set.best_scoring_index());

  candidate_set_message->set_source_key(set.training_key());
  candidate_set_message->set_reference_string(set.reference_string());
}

}  // namespace reranker
