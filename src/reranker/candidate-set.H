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
/// Class to hold a single training instance for a reranker, which is a
/// set of examples, typically the n-best output of some input process, posibly
/// including a gold-standard feature vector.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_SET_H_
#define RERANKER_CANDIDATE_SET_H_

#include <iostream>
#include <string>
#include <tr1/memory>
#include <vector>

#include "candidate.H"
#include "factory.H"

namespace reranker {

using std::tr1::shared_ptr;
using std::ostream;
using std::string;
using std::vector;

/// \class CandidateSet
///
/// A class to hold a set of candidates, either for training or test.  During
/// training, a candidate set typically consists of a reference instance with
/// one or more candidate instances.  During test, a candidate set is simply
/// a set of candidate instances.
class CandidateSet {
 public:
  /// Constructs a new candidate set with no information set.
  CandidateSet() : compiled_(false) { }
  /// Constructs a candidate set with the specified key.
  ///
  /// \param key the unique key that identifies this candidate set
  CandidateSet(const string &key) : training_key_(key), compiled_(false) { }
  /// Destroys this instance.
  virtual ~CandidateSet() { }

  typedef vector<shared_ptr<Candidate> >::iterator iterator;
  typedef vector<shared_ptr<Candidate> >::const_iterator const_iterator;

  /// \class Scorer
  ///
  /// An inner interface for a model to score a \link CandidateSet\endlink.
  class Scorer : public FactoryConstructable {
   public:
    /// An initialization method, guaranteed to be invoked by \link
    /// Factory \endlink after initial construction.  The default
    /// implementation here does nothing.
    ///
    /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
    virtual void Init(const string &arg) { }
    virtual void Score(Model *model,
                       CandidateSet &candidates, bool training) = 0;
  };

  // accessors
  const_iterator begin() const { return candidates_.begin(); }

  const_iterator end() const { return candidates_.end(); }

  iterator begin() { return candidates_.begin(); }

  iterator end() { return candidates_.end(); }

  size_t size() const { return candidates_.size(); }

  size_t best_scoring_index() const { return best_scoring_index_; }
  size_t gold_index() const { return gold_index_; }

  const string &training_key() const { return training_key_; }

  Candidate &Get(size_t idx) {
    return *(candidates_[idx]);
  }

  const Candidate &GetGold() const {
    return *(candidates_[gold_index_]);
  }
  const Candidate &GetBestScoring() const {
    return *(candidates_[best_scoring_index_]);
  }

  const string &reference_string() const { return reference_string_; }

  int reference_string_token_count() const {
    return reference_string_token_count_;
  }

  /// Returns the weight of the loss for this candidate set&rsquo;s reference.
  /// For now, this method returns the <tt>double</tt> representation of
  /// the value returned by the \link reference_string_token_count \endlink
  /// method.
  double loss_weight() const {
    return reference_string_token_count_;
  }

  /// Returns whether any symbolic features in any of the candidates
  /// in this candidate set were compiled by an invocation of \link
  /// CompileFeatures \endlink or any previous invocations of \link
  /// CompileFeatures \endlink
  bool compiled() const { return compiled_; }

  // mutators
  void AddCandidate(shared_ptr<Candidate> candidate) {
    candidates_.push_back(candidate);
  }

  /// Compiles any symbolic features in this candidate set.  For each
  /// candidate, feature weights for symbolic features will be added
  /// to those for any features already specified with <tt>int</tt>
  /// uid&rsquo;s.
  ///
  /// \param symbols                 the map from symbols (string instances)
  ///                                to unique integer id&rsquo;s
  /// \param clear_features          whether to clear each candidate&rsquo;s
  ///                                &ldquo;normal&rdquo; feature
  ///                                vector <b><i>prior to</i></b>
  ///                                compiling symbolic features
  /// \param clear_symbolic_features whether to clear each
  ///                                candidate&rsquo;s symbolic
  ///                                feature vector
  ///                                <b><i>after</i></b> updating the
  ///                                &ldquo;regular&rdquo; feature
  ///                                vector (to save space)
  /// \param force                   forces feature compilation on each
  ///                                candidate, even if this method has been
  ///                                previously invoked
  /// \return whether any symbolic features in any of the candidates
  ///         in this candidate set were actually compiled by this
  ///         method invocation or any previous invocations of this
  ///         method
  ///
  /// \see Candidate::Compile
  bool CompileFeatures(Symbols *symbols,
                       bool clear_features = false,
                       bool clear_symbolic_features = true,
                       bool force = false) {
    if (!compiled_ || force) {
      for (iterator it = begin(); it != end(); ++it) {
        compiled_ |= (*it)->Compile(symbols, clear_features,
                                    clear_symbolic_features, force);
      }
    }
    return compiled_;
  }

  /// Decompiles any non-symbolic features in the candidates in this
  /// candidate set.  For each candidate, feature weights for
  /// non-symbolic features will be added to those for any features
  /// already specified with <tt>string</tt> uid&rsquo;s.
  ///
  /// \param symbols                 the map from symbols (string instances)
  ///                                to unique integer id&rsquo;s
  /// \param clear_symbolic_features whether to clear each
  ///                                candidate&rsquo;s symbolic
  ///                                feature vector <b><i>prior
  ///                                to</i></b> decompiling features
  /// \param clear_features          whether to clear each candidate&rsquo;s
  ///                                &ldquo;normal&rdquo; feature
  ///                                vector <b><i>after</i></b>
  ///                                updating the symbolic feature
  ///                                vector (to save space)
  /// \param force                   forces feature decompilation on each
  ///                                candidate, even if this method has been
  ///                                previously invoked
  ///
  /// \see Candidate::Decompile
  void DecompileFeatures(Symbols *symbols,
                         bool clear_symbolic_features = false,
                         bool clear_features = true,
                         bool force = false) {
    if (compiled_ || force) {
      for (iterator it = begin(); it != end(); ++it) {
        (*it)->Decompile(symbols, clear_symbolic_features,
                         clear_features, force);
      }
    }
    compiled_ = false;
  }

  /// Clears the raw data for all candidates in this set by setting each
  /// to be the empty string.
  void ClearRawData() {
    for (iterator it = begin(); it != end(); ++it) {
      (*it)->set_raw_data(empty_string);
    }
  }

  void set_best_scoring_index(size_t index) {
    best_scoring_index_ = index;
  }

  void set_gold_index(size_t index) {
    gold_index_ = index;
  }

  void set_training_key(const string &training_key) {
    training_key_ = training_key;
  }

  void set_reference_string(const string &reference_string) {
    reference_string_ = reference_string;
  }

  void set_reference_string_token_count(int reference_string_token_count) {
    reference_string_token_count_ = reference_string_token_count;
  }

  // I/O methods

  friend ostream &operator<<(ostream &os, const CandidateSet &set) {
    os << "Candidate set with key \"" << set.training_key()
       << "\" and reference string\n\t" << set.reference_string()
       << "\nwith " << set.size() << " candidates:\n";
    for (const_iterator it = set.begin(); it != set.end(); ++it) {
      os << "\t" << *(*it) << "\n";
    }
    return os;
  }

 private:
  // data members
  /// The candidates stored by this instance.
  vector<shared_ptr<Candidate> > candidates_;
  /// The unique key of the training example that this candidate belongs to.
  string training_key_;
  /// The index in candidates_ for the gold-standard "candidate".
  size_t gold_index_;
  /// The index in candidates_ for the bet-scoring candidate.
  size_t best_scoring_index_;
  /// The reference string against which all these candidates may be judged.
  string reference_string_;
  /// The number of tokens in the reference string.
  int reference_string_token_count_;
  /// Whether the \link CompileFeatures \endlink method has been invoked.
  bool compiled_;

  static string empty_string;
};

#define REGISTER_NAMED_CANDIDATE_SET_SCORER(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,CandidateSet::Scorer)

#define REGISTER_CANDIDATE_SET_SCORER(TYPE) \
  REGISTER_NAMED_CANDIDATE_SET_SCORER(TYPE,TYPE)

}  // namespace reranker

#endif
