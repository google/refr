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
/// Provides the reranker::Candidate class for representing a candidate
/// hypothesis from an initial model.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_H_
#define RERANKER_CANDIDATE_H_

#include <iostream>
#include <string>
#include <sstream>

#include "../proto/data.pb.h"
#include "factory.H"
#include "feature-vector.H"
#include "symbol-table.H"

namespace reranker {

using std::ostream;
using std::string;

class Model;

/// \class Candidate
///
/// A class to represent a candidate in a set of candidates that constitutes
/// a training instance for a reranker.
class Candidate {
 public:
  friend class FeatureExtractor;
  friend class AbstractFileBackedFeatureExtractor;
  /// Constructor for a candidate without features.
  ///
  /// \param index the index of this candidate within its set
  /// \param loss the loss of this candidate
  /// \param baseline_score the baseline score of this candidate
  /// \param num_words the number of words of this candidate
  /// \param raw_data the original string for this candidate
  Candidate(int index,
            double loss,
            double baseline_score,
            int num_words,
            const string &raw_data) :
      index_(index), num_errors_(0), num_correct_(0),
      loss_(loss), score_(0.0), baseline_score_(baseline_score),
      num_words_(num_words), raw_data_(raw_data), compiled_(false) {
  }

  /// Constructor for a candidate with "compiled" features.
  ///
  /// \param index the index of this candidate within its set
  /// \param loss the loss of this candidate
  /// \param baseline_score the baseline score of this candidate
  /// \param num_words the number of words of this candidate
  /// \param raw_data the original string for this candidate
  /// \param features the features for this candidate
  /// \param symbolic_features the symbolic features for this candidate
  Candidate(int index,
            double loss,
            double baseline_score,
            int num_words,
            const string &raw_data,
            const FeatureVector<int,double> &features,
            const FeatureVector<string,double> &symbolic_features) : 
      index_(index), num_errors_(0), num_correct_(0),
      loss_(loss), score_(0.0), baseline_score_(baseline_score),
      num_words_(num_words), features_(features),
      symbolic_features_(symbolic_features), raw_data_(raw_data),
      compiled_(false) {
  }

  /// Destroys this candidate.
  virtual ~Candidate() { }
  
  /// An inner interface specifying comparison between two Candidate instances.
  class Comparator : public FactoryConstructible {
   public:
    /// An initialization method, guaranteed to be invoked by \link
    /// Factory \endlink after initial construction.  The default
    /// implementation here does nothing.
    ///
    /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
    virtual void Init(const string &arg) { }
    /// Compares the two specified Candidate instances.  A return value of 0
    /// indicates that the two instances are equal, a value of less than 0
    /// indicates that <tt>c1 < c2</tt>, and a value
    /// of more than zero indicates that <tt>c1 > c2</tt>.
    virtual int Compare(const Model &model,
                        const Candidate &c1, const Candidate &c2) = 0;
  };

  // accessors
  /// Returns the index of this candidate relative to the other candidates
  // in its set.
  int index() const { return index_; }
  /// Returns the number of word errors of this candidate relative to a
  /// reference string.
  int num_errors() const { return num_errors_; }
  /// Returns the number of correct words of this candidate relative to a
  /// reference string.
  int num_correct() const { return num_correct_; }
  /// Returns the loss of this candidate.
  double loss() const { return loss_; }
  /// Returns the reranker&rsquo;s score for this candidate.
  double score() const { return score_; }
  /// Returns the baseline model score for this candidate.
  double baseline_score() const { return baseline_score_; }
  /// Returns the number of words in this candidate.
  int num_words() const { return num_words_; }
  /// Returns the feature vector for this candidate.
  const FeatureVector<int,double> &features() const { return features_;  }
  /// Returns the symbolic feature vector for this candidate.
  const FeatureVector<string,double> &symbolic_features() const {
    return symbolic_features_;
  }
  /// Returns the raw data (typically the sentence) for this candidate.
  const string &raw_data() const { return raw_data_; }
  /// Returns whether this candidate&rsquo;s symbolic features have been
  /// compiled.
  /// \see Compile
  bool compiled() const { return compiled_; }

  // mutators
  /// Sets the raw data (typically the sentence) for this candidate).
  void set_raw_data(const string &raw_data) { raw_data_ = raw_data; }

  /// Sets the score of this candidate.
  ///
  /// \param score the new score of this candidate
  void set_score(double score) { score_ = score; }

  /// Sets the score of this candidate.
  ///
  /// \param baseline_score the new baseline_score of this candidate
  void set_baseline_score(double baseline_score) {
    baseline_score_ = baseline_score;
  }

  /// Sets the loss of this candidate.
  ///
  /// \param loss the new loss of this candidate.
  void set_loss(double loss) { loss_ = loss; }

  /// Compiles any symbolic features in this candidate.  Feature weights
  /// for symbolic features will be added to those for any features already
  /// specified with <tt>int</tt> uid&rsquo;s.
  ///
  /// \param symbols                 the map from symbols (string instances)
  ///                                to unique integer id&rsquo;s
  /// \param clear_features          whether to clear the &ldquo;normal&rdquo;
  ///                                feature vector <b><i>prior
  ///                                to</i></b> compiling
  ///                                symbolic features
  /// \param clear_symbolic_features whether to clear the symbolic
  ///                                feature vector
  ///                                <b><i>after</i></b> updating the
  ///                                &ldquo;regular&rdquo; feature
  ///                                vector (to save space)
  /// \param force                   forces feature compilation, even if this
  ///                                method has been previously invoked
  /// \return whether any symbolic features were actually compiled by this
  ///         method invocation
  bool Compile(Symbols *symbols,
               bool clear_features = false,
               bool clear_symbolic_features = true,
               bool force = false);

  /// Decompiles any non-symbolic features in this candidate.  Feature weights
  /// for non-symbolic features will be added to those for any features already
  /// specified with <tt>string</tt> uid&rsquo;s.
  ///
  /// \param symbols                 the map from symbols (string instances)
  ///                                to unique integer id&rsquo;s
  /// \param clear_symbolic_features whether to clear the symbolic
  ///                                feature vector <b><i>prior to</i></b>
  ///                                decompiling features
  /// \param clear_features          whether to clear the &ldquo;normal&rdquo;
  ///                                feature vector <b><i>after</i></b>
  ///                                updating the symbolic feature vector (to
  ///                                save space)
  /// \param force                   forces feature decompilation, even if this
  ///                                method has been previously invoked
  void Decompile(Symbols *symbols,
                 bool clear_symbolic_features = false,
                 bool clear_features = true,
                 bool force = false);

  /// Outputs a human-readable string version of this Candidate instance
  /// to the specified ostream.
  ///
  /// \param os the output stream to which to output a human-readable
  ///           version of this candidate
  /// \param c  the candidate to be output to the specified output stream
  friend ostream &operator<<(ostream &os, const Candidate &c) {
    os << "{index:" << c.index() << "; loss:" << c.loss()
       << "; score:" << c.score() << "; baseline_score:" << c.baseline_score()
       << "; features:" << c.features()
       << "; symbolic_features:" << c.symbolic_features()
       << "; raw_data:\"" << c.raw_data() << "\""
       << "; compiled: " << (c.compiled() ? "true" : "false")
       << "}";
    return os;
  }

  /// Returns a the string that would be output by operator<<.
  string to_string() {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

 private:
  // Methods for friend class FeatureExtractor.
  FeatureVector<int,double> &mutable_features() { return features_; }
  FeatureVector<string,double> &mutable_symbolic_features() {
    return symbolic_features_;
  }

  // data members
  // The index of this candidate in its ordered list of candidates
  // (its candidate set).
  // TODO(dbikel,kbhall): Is this really necessary?  The CandidateSet
  //                      will contain this index.  Maybe this will be
  //                      necessary if we want to pass a single
  //                      Candidate to a feature function that needs
  //                      its "rank" in the CandidateSet.
  int index_;
  // TODO(dbikel,kbhall): The following three data members seem overly
  //                      specific to WER (and therefore to speech
  //                      evaluation).  The number of word errors in
  //                      this candidate.
  int num_errors_;
  // The number of correct words in this candidate.a
  int num_correct_;
  // The loss of this candidate relative to the reference.  For speech, this
  // value should be (num_errors_ / num_correct_).
  double loss_;
  // The score of this candidate according to the reranker.
  double score_;
  // The baseline model score for this candidate.
  double baseline_score_;
  // The number of words in this candidate.  (This value might be hard
  // or impossible to determine simply from features.)
  int num_words_;
  // The features for this candidate.
  FeatureVector<int,double> features_;
  // The symbolic features for this candidate.
  FeatureVector<string,double> symbolic_features_;
  // The raw data (string) corresponding to this candidate.
  string raw_data_;
  // Whether this candidate's symbolic features have been "compiled".
  bool compiled_;
};

#define REGISTER_NAMED_CANDIDATE_COMPARATOR(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,Candidate::Comparator)

#define REGISTER_CANDIDATE_COMPARATOR(TYPE) \
  REGISTER_NAMED_CANDIDATE_COMPARATOR(TYPE,TYPE)

}  // namespace reranker

#endif
