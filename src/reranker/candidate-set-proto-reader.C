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
/// \file candidate-set-proto-reader.C
/// Implementation for class that reads CandidateSetMessage instances
/// and converts them to reranker::CandidateSet instances.
/// \author dbikel@google.com (Dan Bikel)

#include <algorithm>
#include <iterator>
#include <set>
#include <memory>

#include "../proto/data.pb.h"
#include "../proto/model.pb.h"
#include "symbol-table.H"
#include "candidate-set-proto-reader.H"

#define DEBUG 0

using confusion_learning::CandidateMessage;
using confusion_learning::CandidateSetMessage;
using confusion_learning::FeatureVecMessage;
using confusion_learning::FeatureMessage;
using confusion_learning::ScoreMessage;

using std::cout;
using std::endl;
using std::insert_iterator;
using std::ostream_iterator;
using std::string;
using std::unordered_map;
using std::shared_ptr;

namespace reranker {

void
CandidateSetProtoReader::Read(const CandidateSetMessage &m,
                              int max_candidates,
                              CandidateSet &set) {
  if (m.has_source_key()) {
    set.set_training_key(m.source_key());
  }
  if (m.has_reference_string()) {
    set.set_reference_string(m.reference_string());
    set.set_reference_string_token_count(CountTokens(set.reference_string()));
  }
  if (m.has_gold_index()) {
    set.set_gold_index(m.gold_index());
  }
  if (m.has_best_scoring_index()) {
    set.set_best_scoring_index(m.best_scoring_index());
  }

  int num_candidates = m.candidate_size();
  int num_candidates_to_read =
      max_candidates < 0 ? num_candidates :
      (max_candidates > num_candidates ? num_candidates : max_candidates);
  for (int i = 0; i < num_candidates_to_read; ++i) {
    const CandidateMessage &candidate_msg = m.candidate(i);
    const FeatureVecMessage &feature_vec_msg = candidate_msg.feats();

    FeatureVector<string,double> symbolic_features;
    FeatureVector<int,double> features;

    for (int j = 0; j < feature_vec_msg.feature_size(); ++j) {
      const FeatureMessage &feature_msg = feature_vec_msg.feature(j);
      if (feature_msg.has_name() && ! feature_msg.name().empty()) {
        symbolic_features.IncrementWeight(feature_msg.name(),
                                          feature_msg.value());
      } else {
        features.IncrementWeight(feature_msg.id(), feature_msg.value());
      }
    }
    bool set_loss = false;
    double loss = 0.0;
    double baseline_score = 0.0;
    for (int score_index = 0;
         score_index < candidate_msg.score_size();
         ++score_index) {
      const ScoreMessage &score_msg = candidate_msg.score(score_index);
      switch (score_msg.type()) {
        case ScoreMessage::LOSS:
          loss = score_msg.score();
          set_loss = true;
          break;
        case ScoreMessage::SYSTEM_SCORE:
          // TODO(dbikel): Deal with the fact that there could be multiple
          //               system scores one day, perhaps simply storing
          //               them in an array (right now, we just take the
          //               last one).
          baseline_score = score_msg.score();
          break;
        case ScoreMessage::OUTPUT_SCORE:
          break;
      }
    }

    int num_words = CountTokens(candidate_msg.raw_data());

    if (!set_loss) {
      cerr << "CandidateSetProtoReader: warning: computing loss by tokenizing"
           << " and counting." << endl;
      loss = ComputeLoss(set, candidate_msg.raw_data());
    }

    shared_ptr<Candidate> candidate(new Candidate(i, loss, baseline_score,
                                                  num_words,
                                                  candidate_msg.raw_data(),
                                                  features, symbolic_features));
    set.AddCandidate(candidate);
  }
}

double
CandidateSetProtoReader::ComputeLoss(CandidateSet &set,
                                     const string &candidate_raw_data) {
  // For now, find loss for candidate by doing "position-independent WER".
  vector<string> ref_toks;
  tokenizer_.Tokenize(set.reference_string(), ref_toks);
  vector<string> candidate_toks;
  tokenizer_.Tokenize(candidate_raw_data, candidate_toks);
  std::set<string> ref_toks_set;
  typedef vector<string>::const_iterator const_it;
  for (const_it it = ref_toks.begin(); it != ref_toks.end(); ++it) {
    ref_toks_set.insert(*it);
  }
  std::set<string> candidate_toks_set;
  for (const_it it = candidate_toks.begin(); it != candidate_toks.end();
       ++it) {
    candidate_toks_set.insert(*it);
  }
  std::set<string> intersection;
  insert_iterator<std::set<string> > ii(intersection, intersection.begin());
  set_intersection(ref_toks_set.begin(), ref_toks_set.end(),
                   candidate_toks_set.begin(), candidate_toks_set.end(),
                   ii);
   

  if (DEBUG) {
    ostream_iterator<string> tab_delimited(cout, "\n\t");
    cout << "Set of ref toks:\n\t";
    copy(ref_toks_set.begin(), ref_toks_set.end(), tab_delimited);
    cout << endl;
    cout << "Set of candidate toks:\n\t";
    copy(candidate_toks_set.begin(), candidate_toks_set.end(), tab_delimited);
    cout << endl;
    cout << "Intersection:\n\t";
    copy(intersection.begin(), intersection.end(), tab_delimited);
    cout << endl;
  }

  double loss = intersection.size() / (double)ref_toks_set.size();

  if (DEBUG) {
    cout << "Intersection size is " << intersection.size()
         << " and there are " << ref_toks_set.size()
         << " ref toks, so loss is " << loss << "." << endl;
  }

  return loss;
}

}  // namespace reranker
