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
/// Model and model factory implementation.  Most of the
/// implementation in this file is related to factory creation and
/// initialization of Model and Candidate::Comaprator instances.
/// Author: dbikel@google.com (Dan Bikel)

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "tokenizer.H"
#include "model.H"

using namespace std;

namespace reranker {

IMPLEMENT_FACTORY(Model)
IMPLEMENT_FACTORY(Model::UpdatePredicate)
IMPLEMENT_FACTORY(Model::Updater)

REGISTER_CANDIDATE_COMPARATOR(DefaultScoreComparator)
REGISTER_CANDIDATE_COMPARATOR(DefaultGoldComparator)

REGISTER_CANDIDATE_SET_SCORER(DefaultCandidateSetScorer)
REGISTER_CANDIDATE_SET_SCORER(RandomPairCandidateSetScorer)

void
DefaultCandidateSetScorer::Score(Model *model,
                                 CandidateSet &candidates, bool training) {
  // N.B.: We assume there is at least one candidate!
  CandidateSet::iterator it = candidates.begin();

  // Score first candidate, which is perforce the best candidate so far.
  model->ScoreCandidate(*(*it), training);

  CandidateSet::iterator best_it = it;
  CandidateSet::iterator gold_it = it;
  ++it;

  // Score any and all remaining candidates.
  for ( ; it != candidates.end(); ++it) {
    Candidate &candidate = *(*it);
    model->ScoreCandidate(candidate, training);
    if (model->score_comparator()->Compare(*model, candidate, **best_it) > 0) {
      best_it = it;
    }
    if (model->gold_comparator()->Compare(*model, candidate, **gold_it) > 0) {
      gold_it = it;
    }
  }
  candidates.set_best_scoring_index((*best_it)->index());
  candidates.set_gold_index((*gold_it)->index());
}

void
RandomPairCandidateSetScorer::Init(const string &arg) { srand(time(NULL)); }

size_t
RandomPairCandidateSetScorer::GetRandomIndex(size_t max) {
  // Get an index proportional to the reciprocal rank.
  // First, get a floating-point value that is distributed uniformly (roughly)
  /// in [0,1].
  double r = rand() / (double)RAND_MAX;
  // We compute the cummulative density function, or cdf, of the
  // reciprocal rank distribution over the fixed set of items.  As soon as
  // our uniformly-distributed r is less than the cdf at index i, we
  // return that index.
  double cdf = 0.0;
  double denominator = (max * (max + 1)) / 2;
  for (size_t i = 0; i < max; ++i) {
    cdf += (max - i) / denominator;
    if (r <= cdf) {
      return i;
    }
  }
  return max - 1;
}

void
RandomPairCandidateSetScorer::Score(Model *model,
                                    CandidateSet &candidates, bool training) {
  // First, pick two candidate indices at random.
  size_t idx1 = GetRandomIndex(candidates.size());
  size_t idx2 = GetRandomIndex(candidates.size());
  Candidate &c1 = candidates.Get(idx1);
  Candidate &c2 = candidates.Get(idx2);

  // Next, just score those two candidates.
  model->ScoreCandidate(c1, training);
  model->ScoreCandidate(c2, training);

  // Finally, set indices of best scoring and gold amongst just those two.
  int score_cmp = model->score_comparator()->Compare(*model, c1, c2);
  candidates.set_best_scoring_index(score_cmp > 0 ? c1.index() : c2.index());

  int gold_cmp = model->gold_comparator()->Compare(*model, c1, c2);
  candidates.set_gold_index(gold_cmp > 0 ? c1.index() : c2.index());
}

void
Model::CheckNumberOfTokens(const string &arg,
                           const vector<string> &tokens,
                           size_t min_expected_number,
                           size_t max_expected_number,
                           const string &class_name) const {
  if ((min_expected_number > 0 && tokens.size() < min_expected_number) ||
      (max_expected_number > 0 && tokens.size() > max_expected_number)) {
    std::stringstream err_ss;
    err_ss << class_name << "::Init: error parsing init string \""
           << arg << "\": expected between "
           << min_expected_number << " and " << max_expected_number
           << " tokens but found " << tokens.size() << " tokens";
    cerr << err_ss.str() << endl;
    throw std::runtime_error(err_ss.str());
  }
}

shared_ptr<Candidate::Comparator>
Model::GetComparator(const string &spec) const {
  Factory<Candidate::Comparator> comparator_factory;
  string err_msg = "error: model " + name() + ": could not construct " +
      "Candidate::Comparator from specification string \"" + spec + "\"";
  return  comparator_factory.CreateOrDie(spec, err_msg);
}

shared_ptr<CandidateSet::Scorer>
Model::GetCandidateSetScorer(const string &spec) const {
  Factory<CandidateSet::Scorer> candidate_set_scorer_factory;
  string err_msg = "error: model " + name() + ": could not construct " +
      "Candidate::Scorer from specification string \"" + spec + "\"";
  return candidate_set_scorer_factory.CreateOrDie(spec, err_msg);
}

shared_ptr<Model::UpdatePredicate>
Model::GetUpdatePredicate(const string &spec) const {
  Factory<Model::UpdatePredicate> update_predicate_factory;
  string err_msg = "error: model " + name() + ": could not construct " +
      "Model::UpdatePredicate from specification string \"" + spec + "\"";
  return update_predicate_factory.CreateOrDie(spec, err_msg);
}

shared_ptr<Model::Updater>
Model::GetUpdater(const string &spec) const {
  Factory<Model::Updater> updater_factory;
  string err_msg = "error: model " + name() + ": could not construct " +
      "Model::Updater from specification string \"" + spec + "\"";
  return updater_factory.CreateOrDie(spec, err_msg);
}

}  // namespace reranker
