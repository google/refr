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
/// \file
/// Provides the reranker::MiraStyleModel reranker class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_MIRA_STYLE_MODEL_H_
#define RERANKER_MIRA_STYLE_MODEL_H_

#include <cstdlib>
#include <string>

#include "kernel-function.H"
#include "symbol-table.H"

#include "perceptron-model.H"

#define DEFAULT_MIRA_CLIP 0.1

namespace reranker {

using std::string;

/// \class DirectLossScoreComparator
///
/// A class to do &ldquo;direct loss minimization&rdquo; by considering the
/// score of a candidate to be its raw score plus its loss insofar as
/// candidate ordering is concerned.
class DirectLossScoreComparator : public Candidate::Comparator {
 public:
  /// Returns 0 if the two candidates&rsquo; scores are equal, less than
  /// zero if the score of <tt>c1</tt> is less than that of <tt>c2</tt>
  /// and more than 0 if the score of <tt>c1</tt> is greater than that
  /// of <tt>c2</tt>.  A candidate&rsquo;s score is the sum of its raw
  /// score plus its loss, as far as this overridden method is concerned.
  virtual int Compare(const Model &model,
                      const Candidate &c1, const Candidate &c2) {
    double score_diff = (c1.score() + c1.loss()) - (c2.score() + c2.loss());
    return score_diff == 0.0 ? 0 : (score_diff < 0.0 ? -1 : 1);
  }
};

/// \class MiraStyleModel
///
/// A subclass of \link PerceptronModel \endlink that differs only in the
/// way that the \link ComputeStepSize \endlink method is implemented.
/// The overridden definition here provides a &ldquo;MIRA-style&rdquo;
/// update, where the step size is related to the difference in losses
/// and scores of the gold and best-scoring candidates.
class MiraStyleModel : public PerceptronModel {
 public:
  MiraStyleModel() : PerceptronModel(), mira_clip_(DEFAULT_MIRA_CLIP) { }
  MiraStyleModel(const string &name) : PerceptronModel(name),
                                       mira_clip_(DEFAULT_MIRA_CLIP) { }
  MiraStyleModel(const string &name, KernelFunction *kernel_fn) :
      PerceptronModel(name, kernel_fn),
      mira_clip_(DEFAULT_MIRA_CLIP) { }
  MiraStyleModel(const string &name, KernelFunction *kernel_fn,
                 Symbols *symbols) :
      PerceptronModel(name, kernel_fn, symbols),
      mira_clip_(DEFAULT_MIRA_CLIP) { }


  /// Registers one additional variable that may be initialized when this object
  /// is constructed via \link
  /// Factory::CreateOrDie(StreamTokenizer&)\endlink.  The set of
  /// members includes all those handled by \link
  /// PerceptronModel::RegisterInitializers\endlink, as well as the
  /// following:
  /// <table>
  /// <tr><th>Variable name</th>
  ///     <th>Type</th>
  ///     <th>Required</th>
  ///     <th>Description</th>
  ///     <th>Default value</th>
  /// </tr>
  /// <tr><td><tt>mira_clip</tt></td>
  ///     <td><tt>double</tt></td>
  ///     <td>No</td>
  ///     <td>The maxmum value this model&rsquo;s step size may attain.</td>
  ///     <td>0.1</td>
  /// </tr>
  /// </table>
  virtual void RegisterInitializers(Initializers &initializers) {
    PerceptronModel::RegisterInitializers(initializers);
    initializers.Add("mira_clip", &mira_clip_);
  }

  /// Sets the maximum value for a step size computed by \link
  /// ComputeStepSize\endlink.
  void set_mira_clip(double mira_clip) { mira_clip_ = mira_clip; }

  /// Computes the step size for the next update, and, as a side effect, caches
  /// this value in step_size_.  The step size here is computed based on the
  /// loss and score difference between the best-scoring candidate and the
  /// gold candidate, to achieve a &ldquo;MIRA-style&rdquo; update.
  virtual double ComputeStepSize(
      const unordered_set<int> &gold_features,
      const unordered_set<int> &best_scoring_features,
      const CandidateSet &example) {
    FeatureVector<int,double> vector_diff;
    vector_diff.AddScaledSubvector(gold_features,
                                   example.GetGold().features(), 1.0);
    vector_diff.AddScaledSubvector(best_scoring_features,
                                   example.GetBestScoring().features(), -1.0);
    double loss_weight = use_weighted_loss() ? example.loss_weight() : 1.0;
    double loss_diff =
        loss_weight *
        (example.GetBestScoring().loss() - example.GetGold().loss());
    double score_diff =
        example.GetBestScoring().score() - example.GetGold().score();
    double raw_step = (loss_diff + score_diff) / vector_diff.Dot(vector_diff);
    step_size_ = raw_step > mira_clip_ ? mira_clip_ : raw_step;
    return step_size_;
  }
 private:
  // data members
  double mira_clip_;
};

}  // namespace reranker

#endif
