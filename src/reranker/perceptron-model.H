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
/// Provides the reranker::PerceptronModel reranker class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_PERCEPTRON_MODEL_H_
#define RERANKER_PERCEPTRON_MODEL_H_

#include <vector>
#include <tr1/unordered_set>
#include <tr1/memory>

#include "candidate-set.H"
#include "dot-product.H"
#include "model.H"
#include "training-time.H"
#include "training-vector-set.H"

#define DEFAULT_MAX_EPOCHS_IN_DECLINE 5

namespace reranker {

using std::tr1::shared_ptr;
using std::tr1::unordered_set;
using std::vector;

/// \class PerceptronModel
///
/// This class implements a perceptron model reranker.  While this
/// model can consist of arbitrary feature types, there is special
/// handling for n-gram&ndash;based features, to capture the fact that,
/// e.g., a bigram suffix exists whenever a trigram occurs.
class PerceptronModel : public Model {
 public:
  friend class PerceptronModelProtoWriter;
  friend class PerceptronModelProtoReader;

  /// Constructs a new instance with the empty string for its name and
  /// the \link DotProduct \endlink kernel function.
  PerceptronModel() :
      Model("", new DotProduct()),
      models_(),
      best_models_(),
      best_model_epoch_(-1),
      max_epochs_in_decline_(DEFAULT_MAX_EPOCHS_IN_DECLINE),
      num_epochs_in_decline_(0),
      step_size_(1.0) {
    SetDefaultObjects();
  }

  /// Constructs a new perceptron model with a \link DotProduct
  /// \endlink kernel function.
  ///
  /// \param name the unique name of this perceptron model instance
  PerceptronModel(const string &name) :
      Model(name, new DotProduct()),
      models_(),
      best_models_(),
      best_model_epoch_(-1),
      max_epochs_in_decline_(DEFAULT_MAX_EPOCHS_IN_DECLINE),
      num_epochs_in_decline_(0),
      step_size_(1.0) {
    SetDefaultObjects();
  }

  /// Constructs a new perceptron model with the specified kernel function.
  ///
  /// \param name the unique name of this perceptron model instance
  /// \param kernel_fn the kernel function for this model to use when
  ///                  evaluating on training or test instances
  PerceptronModel(const string &name, KernelFunction *kernel_fn) :
      Model(name, kernel_fn),
      models_(),
      best_models_(),
      best_model_epoch_(-1),
      max_epochs_in_decline_(DEFAULT_MAX_EPOCHS_IN_DECLINE),
      num_epochs_in_decline_(0),
      step_size_(1.0) {
    SetDefaultObjects();
  }

  /// Constructs a new perceptron model with the specified kernel function
  /// and symbol table.
  ///
  /// \param name      the unique name of this model instance
  /// \param kernel_fn the kernel function for this model to use when
  ///                  applied to training or test instances
  /// \param symbols   the symbol table for this Model to use; this Model
  ///                  will be responsible for deleting this Symbols object
  PerceptronModel(const string &name, KernelFunction *kernel_fn,
                  Symbols *symbols) :
      Model(name, kernel_fn, symbols),
      models_(),
      best_models_(),
      best_model_epoch_(-1),
      max_epochs_in_decline_(DEFAULT_MAX_EPOCHS_IN_DECLINE),
      num_epochs_in_decline_(0),
      step_size_(1.0) {
    SetDefaultObjects();
  }


  /// Destroys this perceptron model and all its data members.
  virtual ~PerceptronModel() { }

  /// \class DefaultUpdatePredicate
  ///
  /// The default update predicate for perceptron and perceptron-style
  /// models, which indicates to do a model update whenever the
  /// top-scoring candidate hypothesis under the current model differs
  /// from the oracle or &ldquo;gold&rdquo; candidate hypothesis.
  class DefaultUpdatePredicate : public Model::UpdatePredicate {
    /// Returns <tt>true</tt> if the top-scoring candidate hypothesis
    /// under the current model differs from the oracle or
    /// &ldquo;gold&rdquo; candidate hypothesis.  The top-scoring
    /// and gold hypothesis candidates are expected to have already
    /// been found and set inside the specified \link CandidateSet\endlink,
    /// accessed via the \link CandidateSet::GetGold \endlink and
    /// \link CandidateSet::GetBestScoring \endlink methods.
    virtual bool NeedToUpdate(Model *model, CandidateSet &example);
  };

  /// \class DefaultUpdater
  ///
  /// The default update function for perceptron models. The implementation
  /// here does the following operations:
  /// <ol>
  /// <li>Computes which features to update using the \link
  ///     PerceptronModel::ComputeFeaturesToUpdate \endlink method.
  /// <li>Updates the gold and top-scoring candidate feature averages using
  ///     the \link TrainingVectorSet::UpdateGoldAndCandidateFeatureAverages
  ///     \endlink method.
  /// <li>Computes the step size using the \link
  ///     PerceptronModel::ComputeStepSize \endlink method.
  /// <li>Updates weights, biasing the model toward the gold
  ///     candidate&rsquo;s features and away from the current
  ///     top-scoring (but &ldquo;incorrect&rdquo;) candidate&rsquo;s
  ///     features, using the \link TrainingVectorSet::UpdateWeights
  ///     \endlink method.
  /// </ol>
  class DefaultUpdater : public Model::Updater {
    virtual void Update(Model *m, CandidateSet &example);
  };

  /// \copydoc Model::model_spec
  virtual const string &model_spec() const { return model_spec_; }

  /// \copydoc Model::proto_reader_spec
  virtual const string &proto_reader_spec() const {
    return proto_reader_spec_;
  }

  /// \copydoc Model::proto_writer_spec
  virtual const string &proto_writer_spec() const {
    return proto_writer_spec_;
  }

  /// Returns the epoch of the best models seen so far during training.
  /// (Primarily here for the PerceptronModelProtoWriter serializer.)
  virtual int best_model_epoch() const { return best_model_epoch_; }

  /// Registers several variables that may be initialized when this object
  /// is constructed via \link Factory::CreateOrDie\endlink.
  /// <table>
  /// <tr>
  ///   <th>Variable name</th>
  ///   <th>Type</th>
  ///   <th>Required</th>
  ///   <th>Description</th>
  ///   <th>Default value</th>
  /// </tr>
  /// <tr>
  ///   <td><tt>name</tt></td>
  ///   <td><tt>string</tt></td>
  ///   <td>Yes</td>
  ///   <td>The name of this model instance (for human consumption).</td>
  ///   <td>n/a</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>score_comparator</tt></td>
  ///   <td>\link Candidate::Comparator \endlink</td>
  ///   <td>No</td>
  ///   <td>The object by which the scores of two \link Candidate \endlink
  ///       instances are compared.</td>
  ///   <td>\link DefaultScoreComparator \endlink</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>gold_comparator</tt></td>
  ///   <td>\link Candidate::Comparator \endlink</td>
  ///   <td>No</td>
  ///   <td>The object by which two \link Candidate \endlink instances are
  ///       compared when finding the &ldquo;gold&rdquo; candidate.</td>
  ///   <td>\link DefaultGoldComparator \endlink</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>candidate_set_scorer</tt></td>
  ///   <td>\link CandidateSet::Scorer \endlink</td>
  ///   <td>No</td>
  ///   <td>The object to score a \link CandidateSet \endlink instance.</td>
  ///   <td>\link DefaultCandidateSetScorer \endlink</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>update_predicate</tt></td>
  ///   <td>\link Model::UpdatePredicate \endlink</td>
  ///   <td>No</td>
  ///   <td>The object to let the model know if it is time to do an update.</td>
  ///   <td>\link PerceptronModel::DefaultUpdatePredicate
  ///       PerceptronModelDefaultUpdatePredicate \endlink</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>updater</tt></td>
  ///   <td>\link Model::Updater \endlink</td>
  ///   <td>No</td>
  ///   <td>The object to update the model.</td>
  ///   <td>\link PerceptronModel::DefaultUpdater
  ///       PerceptronModelDefaultUpdater \endlink</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>step_size</tt></td>
  ///   <td>double</td>
  ///   <td>No</td>
  ///   <td>The initial value of the step size for parameter updates.</td>
  ///   <td><tt>1.0</tt></td>
  /// </tr>
  /// </table>
  virtual void RegisterInitializers(Initializers &initializers);

  /// Initializes this instance. This method is guaranteed to be
  /// invoked by a \link Factory \endlink just after construction.
  virtual void Init(const string &arg);

  // training methods

  virtual bool NeedToKeepTraining();

  /// Trains this model on a collection of training examples, where each
  /// training example is a set of candidates.
  /// \attention
  /// This method is implemented in terms of the \link TrainOnExample
  /// \endlink method.  Thus, for mistake-driven learning methods
  /// similar to the perceptron, one need only derive a class from
  /// this one and override \link TrainOnExample\endlink.
  /// 
  /// \param examples         the set of training examples on which to train
  ///                         this model
  /// \param development_test the set of held-out examples to use to
  ///                         evaluate the model after each epoch
  virtual void Train(CandidateSetIterator &examples,
                     CandidateSetIterator &development_test);

  virtual void NewEpoch();

  virtual void EndOfEpoch();

  /// Trains this model for one epoch, i.e., a single pass through the specified
  /// set of training examples.  Typically the Train method will be implemented
  /// in terms of this method.
  ///
  /// \param examples the set of training examples on which to train this model
  virtual void TrainOneEpoch(CandidateSetIterator &examples);

  /// Trains this model on the specified training example.
  ///
  /// \param example the example to train on
  virtual void TrainOnExample(CandidateSet &example);

  /// Indicates whether the current model needs to be updated; the
  /// implementation here simply returns true if the best-scoring
  /// candidate is not equal to the gold or reference candidate.
  ///
  /// \param example the current training example
  virtual bool NeedToUpdate(CandidateSet &example);

  /// Updates the current model based on the specified set of
  /// candidates.  \link TrainOnExample \endlink will be implemented
  /// in terms of this method.
  ///
  /// \param example the current training example
  virtual void Update(CandidateSet &example);

  /// \copydoc Model::Evaluate
  virtual double Evaluate(CandidateSetIterator &development_test);

  /// Scores the specified set of candidates according to either the
  /// raw or averaged version of this perceptron model, keeping track
  /// of which candidate has the highest score and which candidate has
  /// the lowest loss with the best score.  The scores of the specified set of
  /// candidates may be modified.  This method is currently entirely
  /// implemented via \link DefaultCandidateSetScorer\endlink.
  ///
  /// \param[in,out] candidates the set of candidates to be scored
  /// \param         training   whether this is being called during
  ///                           training or evaluation of a model
  ///
  /// \see DefaultCandidateSetScorer::Score
  virtual void ScoreCandidates(CandidateSet &candidates, bool training);

  /// Scores a candidate according to either the raw or averaged
  /// version of this perceptron model.  The specified candidate's
  /// score may be modified.
  ///
  /// \param[in,out] candidate the candidate to be scored by this model
  /// \param         training   whether this is being called during
  ///                           training or evaluation of a model
  /// \return the score of the specified candidate according to the specified
  ///         model (also contained in the candidate itself)
  virtual double ScoreCandidate(Candidate &candidate, bool training);

  // mutators

  /// \copydoc Model::CompactifyFeatureUids
  virtual void CompactifyFeatureUids();
  
  /// Sets the maximum number of training epochs to keep training after
  /// the model starts to degrade (i.e., has more errors than the best
  /// model so far).
  void set_max_epochs_in_decline(int max_epochs_in_decline) {
    max_epochs_in_decline_ = max_epochs_in_decline;
  }

  /// Returns the set of models and statistics used by this
  /// PerceptronModel instance.
  virtual const TrainingVectorSet &models() const { return models_; }

 protected:
  void SetDefaultObjects() {
    update_predicate_ =
        GetUpdatePredicate("PerceptronModelDefaultUpdatePredicate()");
    updater_ =
        GetUpdater("PerceptronModelDefaultUpdater()");
  }

  /// Computes the features to be updated for the gold candidate and
  /// the best-scoring candidate.  Let G be gold features and B be
  /// best-scoring features.  For the perceptron, we want to update the
  /// set difference G\\B positively and B\\G negatively.  These
  /// two set difference operations are computed by this method.
  /// \attention
  /// Neither of the two specified sets are cleared by this method.
  ///
  /// \param example the candidate set from which to get the gold feature vector
  ///                and the best-scoring candidate feature vector
  /// \param[out]    gold_features_to_update
  ///                  a set in which to insert the uid's of all features in the
  ///                  gold that are not in the best scoring candidate
  /// \param[out]    best_scoring_features_to_update
  ///                  a set in which to insert the uid's of all features in the
  ///                  best-scoring candidate that are not in the gold candidate
  virtual void ComputeFeaturesToUpdate(const CandidateSet &example,
                                       unordered_set<int> &
                                       gold_features_to_update,
                                       unordered_set<int> &
                                       best_scoring_features_to_update) const;

  /// Computes the step size for the next update, and, as a side effect, caches
  /// this value in step_size_.  In the case of the standard perceptron model
  /// implemented here, the step size does not change, and so this method simply
  /// returns the step size value set at construction time.
  virtual double ComputeStepSize(const unordered_set<int> &gold_features,
                               const unordered_set<int> &best_scoring_features,
                               const CandidateSet &example) {
    return step_size_;
    // For MIRA, simply derive class and override to be
    /*
    int feature_count = gold_features.size() + best_scoring_features.size();
    double loss_diff =
      example.GetBestScoring().loss() - example.GetGold().loss();
    double score_diff =
      example.GetBestScoring().score() - example.GetGold().score();
    double raw_step = (loss_diff + score_diff) / feature_count;
    step_size_ = raw_step > mira_clip_ : mira_clip_ : raw_step;
    return step_size_;
    */
  }

  // data members
  /// The feature vectors representing this model.
  TrainingVectorSet models_;
  /// The best models seen so far during training, according to evaluation on
  /// the held-out development test data.
  TrainingVectorSet best_models_;
  /// The epoch of the best models seen so far during training.
  int best_model_epoch_;
  /// The maximum number of training epochs to keep training after the model
  /// starts to degrade (i.e., has more errors than the best model so far).
  int max_epochs_in_decline_;
  /// The current number of training epochs in which the model has been
  /// degrading in development set performance (i.e., has been having more
  /// errors than best model so far).
  int num_epochs_in_decline_;
  /// The last value computed by the ComputeStepSize method.
  double step_size_;
  string model_spec_;

  /// A string that specifies to construct a \link
  /// PerceptronModelProtoReader \endlink, which is capable of de-serializing
  /// an instance of this class.
  static string proto_reader_spec_;
  /// A string that specifies to construct a \link
  /// PerceptronModelProtoWriter \endlink, which is capable of serializing
  /// an instance of this class.
  static string proto_writer_spec_;
};

}  // namespace reranker

#endif
