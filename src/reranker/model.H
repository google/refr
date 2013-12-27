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
/// Reranker model interface.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_MODEL_H_
#define RERANKER_MODEL_H_

#include <stdexcept>
#include <memory>
#include <vector>

#include "candidate-set.H"
#include "candidate-set-iterator.H"
#include "kernel-function.H"
#include "symbol-table.H"
#include "training-time.H"
#include "factory.H"

/// Provides reranking models for discriminative modeling, with some special
/// handling for discriminative language models.
namespace reranker {

using std::vector;
using std::shared_ptr;

/// The default comparator for comparing two Candidate instances based on
/// their respective scores (i.e., the values returned by invoking
/// their Candidate::score methods).
class DefaultScoreComparator : public Candidate::Comparator {
 public:
  /// Returns 0 if the two candidates&rsquo; scores are equal, less than
  /// zero if the score of <tt>c1</tt> is less than that of <tt>c2</tt>
  /// and more than 0 if the score of <tt>c1</tt> is greater than that
  /// of <tt>c2</tt>.
  virtual int Compare(const Model &model,
                      const Candidate &c1, const Candidate &c2) {
    double score_diff = c1.score() - c2.score();
    return score_diff == 0.0 ? 0 : (score_diff < 0.0 ? -1 : 1);
  }
};

/// The default comparator for comparing two Candidate instances for
/// being the &ldquo;gold&rdquo; candidate.
class DefaultGoldComparator : public Candidate::Comparator {
 public:
  /// This method first compares <tt>c1</tt> to <tt>c2</tt> based on
  /// their respective losses (i.e., the values returned by invoking
  /// their Candidate::loss methods): <tt>c1</tt> is considered to be
  /// <b><i>greater</i></b> than <tt>c2</tt> if it has a <b><i>lower
  /// loss</i></b>.  If the two losses are equal, then the two
  /// specified Candidate instances are compared based on score, as
  /// described by the documentation for the \link
  /// DefaultScoreComparator::Compare \endlink method.
  virtual int Compare(const Model &model,
                      const Candidate &c1, const Candidate &c2) {
    double loss_diff = c1.loss() - c2.loss();
    double score_diff = c1.score() - c2.score();
    // If c1 has "less loss" it is "better", so we return 1.
    if (loss_diff < 0.0) {
      return 1;
    } else if (loss_diff > 0.0) {
      return -1;
    } else {
      // If c1&rsquo;s and c2&rsquo;s losses are equal, we compare
      // them based on score, where a higher score is &ldquo;better&rdquo;
      // (meaning a positive return value).
      return score_diff == 0.0 ? 0 : (score_diff < 0.0 ? -1 : 1);
    }
  }
};

/// The default candidate set scorer scores each candidate using the
/// \link Model::ScoreCandidate \endlink method and then sets the
/// index of the best-scoring candidate based on the results of having
/// applied the \link Model::score_comparator \endlink and sets the
/// index of the gold candidate based on the results of having applied
/// the \link Model::gold_comparator\endlink.
class DefaultCandidateSetScorer : public CandidateSet::Scorer {
 public:
  virtual void Score(Model *model,
                     CandidateSet &candidates, bool training);
};

/// This candidate set scorer picks two candidates at random from the
/// set, scores them and then identifies which has the higher score
/// and the lowest loss, effectively meaning that training proceeds as
/// if those were the only two candidates.
class RandomPairCandidateSetScorer : public CandidateSet::Scorer {
 public:
  /// Initializes the random seed using <tt>srand</tt> and the current time
  /// available by calling <tt>time(NULL)</tt>.
  virtual void Init(const Environment *env, const string &arg);
  /// Picks two candidates at random from the set, scores them and
  /// then identifies which has the higher score and the lowest loss,
  /// effectively meaning that training proceeds as if those were the
  /// only two candidates.
  virtual void Score(Model *model,
                     CandidateSet &candidates, bool training);
 private:
  /// Returns an index in the interval [0,max-1] chosen according to
  /// the discrete reciprocal rank distribution over max items.
  size_t GetRandomIndex(size_t max);
};

/// \class Model
/// Model is an interface for reranking models.
///
/// TODO(dbikel): Add factory for kernel functions.  Remove all constructors
/// but the zero-arg constructor from both Model and PerceptronModel.
class Model : public FactoryConstructible {
 public:
  /// Constructs a new instance with the empty string for its name and
  /// a NULL kernel function.
  Model() : name_(""), time_(), kernel_fn_(NULL),
            symbols_(new LocalSymbolTable()),
            loss_per_epoch_(),
            num_testing_errors_per_epoch_(),
            num_training_errors_per_epoch_(),
            num_training_errors_(0), num_updates_(0),
            min_epochs_(-1), max_epochs_(-1),
            end_of_epoch_hook_(NULL) {  
    SetDefaultObjects();
  }


  /// Constructs a new instance with a NULL kernel function.
  ///
  /// \param name the unique name of this model instance
  Model(const string &name) :
      name_(name), time_(), kernel_fn_(NULL),
      symbols_(new LocalSymbolTable()),
      loss_per_epoch_(),
      num_testing_errors_per_epoch_(),
      num_training_errors_per_epoch_(),
      num_training_errors_(0), num_updates_(0),
      min_epochs_(-1), max_epochs_(-1),
      end_of_epoch_hook_(NULL) {
    SetDefaultObjects();
  }

  /// Constructs a new instance with the specified kernel function.
  ///
  /// \param name      the unique name of this model instance
  /// \param kernel_fn the kernel function for this model to use when
  ///                  applied to training or test instances
  Model(const string &name, KernelFunction *kernel_fn) :
      name_(name), time_(), kernel_fn_(kernel_fn),
      symbols_(new LocalSymbolTable()),
      loss_per_epoch_(),
      num_testing_errors_per_epoch_(),
      num_training_errors_per_epoch_(),
      num_training_errors_(0),
      num_updates_(0),
      min_epochs_(-1), max_epochs_(-1),
      end_of_epoch_hook_(NULL) {
    SetDefaultObjects();
  }

  /// Constructs a new instance with the specified kernel function
  /// and symbol table.
  ///
  /// \param name      the unique name of this model instance
  /// \param kernel_fn the kernel function for this model to use when
  ///                  applied to training or test instances
  /// \param symbols   the symbol table for this Model to use; this Model
  ///                  will be responsible for deleting this Symbols object
  Model(const string &name, KernelFunction *kernel_fn,
        Symbols *symbols) :
      name_(name), time_(), kernel_fn_(kernel_fn),
      symbols_(symbols),
      loss_per_epoch_(),
      num_testing_errors_per_epoch_(),
      num_training_errors_per_epoch_(),
      num_training_errors_(0),
      num_updates_(0),
      min_epochs_(-1), max_epochs_(-1),
      end_of_epoch_hook_(NULL) {
    SetDefaultObjects();
  }

  /// Destroys this model and its associated kernel function.
  virtual ~Model() {
    delete kernel_fn_;
    delete symbols_;
    delete end_of_epoch_hook_;
  }

  // inner interfaces

  /// An interface for specifying a hook to be run by a \link Model \endlink
  /// instance.
  class Hook {
   public:
    virtual ~Hook() { }
    /// The function to be executed by the \link Model \endlink that wraps
    /// this hook.
    ///
    /// \param model the model on which this hook will operate (should be
    ///              the \link Model \endlink instance that wraps this hook)
    virtual void Do(Model *model) = 0;
  };

  /// An inner interface for a predicate that tests whether a \link
  /// Model \endlink needs to be updated based on the current training
  /// example.  Typically, implementations of this interface will be
  /// inner classes of the concrete \link Model \endlink
  /// implementation to which they apply.
  ///
  /// \see Model::NeedToUpdate
  /// \see Model::Init
  /// \see Model::InitHelper
  class UpdatePredicate : public FactoryConstructible {
   public:
    /// Returns whether the specified needs to be updated based on the
    /// specified training example.  The specified \link Model \endlink
    /// is non-<tt>const</tt> because it may be the case that its state
    /// needs modification based on the outcome of this predicate.
    ///
    /// \param model   the model to test
    /// \param example the current training example
    virtual bool NeedToUpdate(Model *model, CandidateSet &example) = 0;
  };

  /// An inner interface specifying an update function for a model.
  /// The update function typically goes hand in hand with the type of
  /// model. For example, perceptron-style models get updated by
  /// adding to the current weight vector a possibly scaled version of
  /// the &ldquo;gold&rdquo; candidate hypothesis and subtracting a
  /// possibly scaled version of the candidate hypothesis that scored
  /// highest under the current model.  Typically, implementations of this
  /// interface will be inner classes of the concrete \link Model \endlink
  /// implementation to which they apply.
  ///
  /// \see Model::Update
  /// \see Model::Init
  /// \see Model::InitHelper
  class Updater : public FactoryConstructible {
   public:
    /// Updates this model based on the specified training example.
    ///
    /// \param model   the model to update
    /// \param example the current training example that caused the model
    ///                to need updating
    virtual void Update(Model *model, CandidateSet &example) = 0;
  };

  // accessors

  /// Returns the unique name for this model instance.
  const string &name() const { return name_; }

  /// Returns the symbol table for this model.
  Symbols *symbols() const { return symbols_; }

  /// Returns the current training time of this model: number of
  /// epochs, number of time steps in the current epoch and total
  /// number of time steps (which is equal to the total number of
  /// training examples seen).
  const Time &time() const { return time_; }

  virtual int best_model_epoch() const = 0;

  /// Returns the number of updates made by this model.  In the case of
  /// the perceptron family of algorithms, this value is typically
  /// equal to the number of training errors.
  int num_updates() const { return num_updates_; }
  /// Returns the number of training errors made for each epoch. The last
  /// element of this vector is the number of training errors made during
  /// the most recent epoch of training.
  const vector<int> &num_training_errors_per_epoch() {
    return num_training_errors_per_epoch_;
  }
  /// Returns the number of training errors made by this model.  In the
  /// case of the perceptron family of algorithms, this value is
  /// typically equal to the number of updates made during training.
  int num_training_errors() const { return num_training_errors_; }

  /// Returns the minimum number of epochs to train.
  int min_epochs() const { return min_epochs_; }

  /// Returns the maximum number of epochs to train.
  int max_epochs() const { return max_epochs_; }

  /// Returns the loss per epoch for epoch of training that was evaluated.
  const vector<double> &loss_per_epoch() { return loss_per_epoch_; }

  /// Returns the spec string for constructing a default instance of this
  /// model so it may be properly de-serialized by its
  /// \link ModelProtoReader \endlink.
  virtual const string &model_spec() const = 0;

  /// Returns the spec string for contructing an instance of a
  /// \link ModelProtoReader \endlink capable of de-serializing this
  /// \link Model \endlink implementation.
  virtual const string &proto_reader_spec() const = 0;

  /// Returns the spec string for contructing an instance of a
  /// \link ModelProtoWriter \endlink capable of serializing this
  /// \link Model \endlink implementation.
  virtual const string &proto_writer_spec() const = 0;

  /// Returns a pointer to the score comparator used by this model.
  virtual shared_ptr<Candidate::Comparator> score_comparator() {
    return score_comparator_;
  }

  /// Returns a pointer to the gold comparator used by this model.
  virtual shared_ptr<Candidate::Comparator> gold_comparator() {
    return gold_comparator_;
  }

  // training methods

  /// Returns whether more training epochs are required for this model.
  /// \par Implementation advice:
  /// Implementations of the \link Train \endlink method are strongly
  /// encouraged to have their main loop test be the return value of this
  /// method.  Also, the return value of this method should respect
  /// the \link min_epochs \endlink and \link max_epochs \endlink
  /// values.
  virtual bool NeedToKeepTraining()  = 0;

  // TODO(dbikel,kbhall): Add Train method that takes MapInput (or
  //                      Hadoop equivalent) for streaming input of
  //                      CandidateSet instances.

  /// Trains this model on a collection of training examples, where
  /// each training example is a set of candidates, possibly with a
  /// distinguished "gold candidate" that is the reference against
  /// which other candidates are scored.  Typically, invoking this
  /// method indicates to train a model until convergence.
  /// 
  /// \param examples         the set of training examples on which to train
  ///                         this model
  /// \param development_test the set of held-out examples to use to evaluate
  ///                         the model after each epoch
  virtual void Train(CandidateSetIterator &examples,
                     CandidateSetIterator &development_test) = 0;

  virtual void NewEpoch() = 0;

  virtual void EndOfEpoch() = 0;

  /// Trains this model for one epoch, i.e., a single pass through the specified
  /// set of training examples.  Typically the Train method will be implemented
  /// in terms of this method.
  ///
  /// \param examples the set of training examples on which to train this model
  virtual void TrainOneEpoch(CandidateSetIterator &examples) = 0;

  /// Trains this model on the specified training example.  Typically, the
  /// TrainOneEpoch method will be implemented in terms of this method.
  ///
  /// \param example the example on which to train
  virtual void TrainOnExample(CandidateSet &example) = 0;

  /// Indicates whether the current model needs to be updated.  For
  /// perceptron-style models, this is usually based on having scored
  /// the candidates and finding that the best-scoring candidate is not
  /// equal to the gold or reference candidate.
  ///
  /// \par Implementation advice:
  /// This method is typically implemented in terms of a \link
  /// UpdatePredicate \endlink instance.
  ///
  /// \param example the current training example
  virtual bool NeedToUpdate(CandidateSet &example) = 0;

  /// Updates the current model based on the specified set of candidates.
  /// TrainOnExample will be implemented in terms of this method.
  ///
  /// \par Implementation advice:
  /// This method is typically implemented in terms of a \link Updater
  /// \endlink instance.
  ///
  /// \param example the current training example
  virtual void Update(CandidateSet &example) = 0;

  /// Evaluates this model on the specified set of held-out development
  /// test data.
  /// \par Side effects:
  /// This method is guaranteed to append a value to the internal
  /// \link loss_per_epoch_ \endlink vector.
  ///
  /// \param development_test a held-out set of examples to use for evaluation
  ///                         of this model (during training, this method is
  ///                         typically invoked after each epoch to determine
  ///                         when to stop)
  /// \return the loss of this model when evaluated on the specified development
  ///         test set
  virtual double Evaluate(CandidateSetIterator &development_test)
  = 0;

  /// Scores the specified set of candidates according to either the
  /// raw or averaged version of this perceptron model.  The scores of
  /// the specified set of candidates may be modified.
  ///
  /// \param[in,out] candidates the set of candidates to be scored
  /// \param         training   whether this is being called during
  ///                           training or evaluation of a model
  virtual void ScoreCandidates(CandidateSet &candidates, bool training) = 0;

  /// Scores a candidate according to either the raw or averaged
  /// version of this perceptron model.  The specified candidate's
  /// score may be modified.
  ///
  /// \param[in,out] candidate the candidate to be scored by this model
  /// \param         training   whether this is being called during
  ///                           training or evaluation of a model
  /// \return the score of the specified candidate according to the specified
  ///         model
  virtual double ScoreCandidate(Candidate &candidate, bool training) = 0;

  // mutators

  /// Sets the minimum number of epochs to train.  This value will only
  /// be respected if greater than 0.
  virtual void set_min_epochs(int min_epochs) { min_epochs_ = min_epochs; }

  /// Sets the maximum number of epochs to train.  This value will only
  /// be respected if greater than 0.
  virtual void set_max_epochs(int max_epochs) { max_epochs_ = max_epochs; }

  /// Renumbers the potentially sparse feature uid&rsquo;s so that
  /// they occupy the interval <tt>[0,n-1]</tt> densely, for <tt>n</tt>
  /// non-zero features in use by this model.  If the internal Symbols instance
  /// is non-<tt>NULL</tt>, then this method also adjusts it to reflect
  /// the new set of feature uid&rsquo;s.
  virtual void CompactifyFeatureUids() = 0;

  virtual void set_end_of_epoch_hook(Hook *end_of_epoch_hook) {
    if (end_of_epoch_hook_ != NULL) {
      delete end_of_epoch_hook_;
    }
    end_of_epoch_hook_ = end_of_epoch_hook;
  }

  virtual bool use_weighted_loss() { return use_weighted_loss_; }

  virtual void set_use_weighted_loss(bool use_weighted_loss) {
    use_weighted_loss_ = use_weighted_loss;
  }

  /// Sets the Symbols instance for this Model to be the specified instance.
  /// This Model will be responsible for destroying the specified Symbols
  /// instance.
  virtual void set_symbols(Symbols *symbols) {
    delete symbols_;
    symbols_ = symbols;
  }

 protected:
  // more mutators (protected, because they should only be used by subclasses)

  /// Sets the name of this \link Model \endlink instance.
  void set_name(const string &name) { name_ = name; }

  /// Sets the kernel function for this model.  The kernel function
  /// instance will be owned by this Model instance.
  void set_kernel_fn(KernelFunction *kernel_fn) {
    delete kernel_fn_;
    kernel_fn_ = kernel_fn;
  }

  void set_score_comparator(shared_ptr<Candidate::Comparator> score_comparator)
  {
    score_comparator_ = score_comparator;
  }

  void set_gold_comparator(shared_ptr<Candidate::Comparator> gold_comparator) {
    gold_comparator_ =  gold_comparator;
  }

  void SetDefaultObjects() {
    SetDefaultComparators();
    SetDefaultCandidateSetScorer();
  }

  void SetDefaultComparators() {
    score_comparator_ = GetComparator("DefaultScoreComparator()");
    gold_comparator_ = GetComparator("DefaultGoldComparator()");
  }

  void SetDefaultCandidateSetScorer() {
    candidate_set_scorer_ =
        GetCandidateSetScorer("DefaultCandidateSetScorer()");
  }

  shared_ptr<Candidate::Comparator> GetComparator(const string &spec) const;

  shared_ptr<CandidateSet::Scorer> GetCandidateSetScorer(const string &spec)
      const;

  shared_ptr<UpdatePredicate> GetUpdatePredicate(const string &spec) const;

  shared_ptr<Updater> GetUpdater(const string &spec) const;

  /// A helper method for implementing the \link Init \endlink method:
  /// throws a std::runtime_error if the number of tokens in the
  /// argument string is not the expected number.
  ///
  /// \param arg                 the arg that was tokenized (for error
  ///                            reporting)
  /// \param tokens              the tokens array to inspect for correct size
  /// \param min_expected_number the minimum expected number of tokens
  /// \param max_expected_number the maximum expected number of tokens
  /// \param class_name          the name of the class invoking this method
  ///                            (for error reporting)
  virtual void CheckNumberOfTokens(const string &arg,
                                   const vector<string> &tokens,
                                   size_t min_expected_number,
                                   size_t max_expected_number,
                                   const string &class_name) const;

  /// This model&rsquo;s unique name.
  string name_;
  /// The tiny object that holds the "training time" for this model
  /// (epoch, index and absolute time index).  Note that the absolute time
  /// index should always equal the total number of training examples seen
  /// during training.
  Time time_;
  /// Yes, this is an interface, but we add the kernel function as a
  /// data member.
  KernelFunction *kernel_fn_;
  /// The symbol table for this model (may be NULL).
  Symbols *symbols_;
  /// A comparator to provide an ordering for candidates based on score
  /// when scoring all candidates in a set.
  shared_ptr<Candidate::Comparator> score_comparator_;
  /// A comparator to provide an ordering for candidates to find the
  /// gold candidate in a set.
  shared_ptr<Candidate::Comparator> gold_comparator_;
  /// A scorer for \link CandidateSet \endlink instances.
  shared_ptr<CandidateSet::Scorer> candidate_set_scorer_;
  /// The update predicate for this model.
  shared_ptr<UpdatePredicate> update_predicate_;
  /// The updater for this model.
  shared_ptr<Updater> updater_;
  /// The average loss per epoch.
  vector<double> loss_per_epoch_;
  /// The number of testing errors made on held-out development test data
  /// for each epoch.
  vector<int> num_testing_errors_per_epoch_;
  /// The number of errors made on training examples during each epoch.
  vector<int> num_training_errors_per_epoch_;
  /// The number of errors made on training examples.  This value may
  /// be identical to num_updates_ in the case of the perceptron family
  /// of algorithms.
  int num_training_errors_;
  /// The number of times an update was performed on this model during
  /// training. This value may be identical to num_training_errors_ in
  /// the case of the perceptron family of algorithms.
  int num_updates_;
  /// The minimum number of training epochs to execute.
  int min_epochs_;
  /// The maximum number of training epochs to execute.
  int max_epochs_;
  /// A hook to be performed at the end of every epoch.
  Hook *end_of_epoch_hook_;
  /// Indicates whether this model should weight each candidate&rsquo;s loss
  /// by the value returned by \link CandidateSet::loss_weight\endlink.
  bool use_weighted_loss_;
};

/// Registers the \link reranker::Model Model \endlink implementation
/// with the specified subtype <tt>TYPE</tt> and <tt>NAME</tt> with
/// the \link reranker::Model Model \endlink \link reranker::Factory
/// Factory\endlink.
#define REGISTER_NAMED_MODEL(TYPE,NAME) REGISTER_NAMED(TYPE,NAME,Model)

/// Registers the \link reranker::Model Model \endlink implementation
/// with the specified subtype <tt>TYPE</tt> with the \link
/// reranker::Model Model \endlink \link reranker::Factory
/// Factory\endlink.
#define REGISTER_MODEL(TYPE) REGISTER_NAMED_MODEL(TYPE,TYPE)

/// Registers the \link reranker::Model::UpdatePredicate
/// Model::UpdatePredicate \endlink implementation with the specified
/// subtype <tt>TYPE</tt> and <tt>NAME</tt> with the \link
/// reranker::Model::UpdatePredicate Model::UpdatePredicate \endlink
/// \link reranker::Factory Factory\endlink.
#define REGISTER_NAMED_MODEL_UPDATE_PREDICATE(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,Model::UpdatePredicate)

/// Registers the \link reranker::Model::UpdatePredicate
/// Model::UpdatePredicate \endlink implementation with the specified
/// subtype <tt>TYPE</tt> with the \link
/// reranker::Model::UpdatePredicate Model::UpdatePredicate \endlink
/// \link reranker::Factory Factory\endlink.
#define REGISTER_MODEL_UPDATE_PREDICATE(TYPE) \
  REGISTER_NAMED_MODEL_UPDATE_PREDICATE(TYPE,TYPE)

/// Registers the \link reranker::Model::Updater
/// Model::Updater \endlink implementation with the specified
/// subtype <tt>TYPE</tt> and <tt>NAME</tt> with the \link
/// reranker::Model::Updater Model::Updater \endlink
/// \link reranker::Factory Factory\endlink.
#define REGISTER_NAMED_MODEL_UPDATER(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,Model::Updater)

/// Registers the \link reranker::Model::Updater Model::Updater
/// \endlink implementation with the specified subtype <tt>TYPE</tt>
/// with the \link reranker::Model::Updater Model::Updater \endlink
/// \link reranker::Factory Factory\endlink.
#define REGISTER_MODEL_UPDATER(TYPE) \
  REGISTER_NAMED_MODEL_UPDATER(TYPE,TYPE)

}  // namespace reranker

#endif
