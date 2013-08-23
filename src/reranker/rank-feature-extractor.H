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
/// Provides the reranker::RankFeatureExtractor class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_RANK_FEATURE_EXTRACTOR_H_
#define RERANKER_RANK_FEATURE_EXTRACTOR_H_

#include <cmath>
#include <sstream>

#include "feature-extractor.H"

namespace reranker {

using std::stringstream;

/// \class RankFeatureExtractor
///
/// Encodes the baseline rank as a boolean feature.
class RankFeatureExtractor : public FeatureExtractor {
 public:
  /// Constructs an instance.
  RankFeatureExtractor() : FeatureExtractor() {
  }
  /// Destroys this instance.
  virtual ~RankFeatureExtractor() { }

  virtual void RegisterInitializers(Initializers &initializers) {
    initializers.Add("add_score_factor", &add_score_factor_);
    initializers.Add("add_score_difference", &add_score_difference_);
  }

  // No extra initialization required.
  virtual void Init(const string &arg) { }

  virtual void Extract(Candidate &candidate,
		       FeatureVector<int,double> &features) { }

  /// Extracts symbolic (or <i>string</i>) features.  These will be
  /// automatically compiled when training.
  ///
  /// \param[in]  candidate         the candidate for which to extract features
  /// \param[out] symbolic_features the features extracted for the specified
  ///                               candidate
  virtual void ExtractSymbolic(Candidate &candidate,
                               FeatureVector<string,double> &symbolic_features)
  {
    stringstream ss;
    ss << "orig_rank:" << candidate.index();
    symbolic_features.IncrementWeight(ss.str(), 1.0);

    stringstream lg_rank_ss;
    double lg_rank = floor(log((double)candidate.index() + 1.0) /
                           log(2.0) + 0.5);
    lg_rank_ss << "lg_rank:" << lg_rank;
    symbolic_features.IncrementWeight(lg_rank_ss.str(), 1.0);

    if (add_score_factor_) {
      double top_baseline_score =
          top_baseline_score_ == 0.0 ? 1e-10 : top_baseline_score_;
      double score_factor = candidate.baseline_score() / top_baseline_score;
      symbolic_features.IncrementWeight("score_factor", score_factor);
    }
    if (add_score_difference_) {
      double score_difference =
          candidate.baseline_score() - top_baseline_score_;
      symbolic_features.IncrementWeight("score_difference", score_difference);
    }
  }

  /// Override the default Extract method so that it caches the score of the
  /// top-scoring hypothesis of the baseline system.
  virtual void Extract(CandidateSet &candidate_set) {
    top_baseline_score_ = candidate_set.Get(0).baseline_score();
    FeatureExtractor::Extract(candidate_set);
  }

 private:
  double top_baseline_score_;
  bool add_score_factor_;
  bool add_score_difference_;
};

}  // namespace reranker

#endif
