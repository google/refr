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
/// Provides the reranker::TrainingVectorSet class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_TRAINING_VECTOR_SET_H_
#define RERANKER_TRAINING_VECTOR_SET_H_

#include <iostream>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include "training-time.H"
#include "feature-vector.H"

namespace reranker {

using std::cerr;
using std::endl;
using std::tr1::unordered_map;
using std::tr1::unordered_set;

/// \class TrainingVectorSet
///
/// A class to hold the several feature vectors needed during training
/// (especially for the perceptron family of algorithms), as well as
/// for performing the updates to those feature vectors (again, with
/// the perceptron family of algorithms in mind).
class TrainingVectorSet {
 public:
  friend class PerceptronModelProtoReader;
  /// Constructs a new set of feature vectors (models) for use during training.
  TrainingVectorSet() { }
  /// Destroys this instance.
  virtual ~TrainingVectorSet() { }

  // accessors

  /// Returns the "raw" feature weights computed during training.  This
  /// is essentially the "most recent" perceptron created during
  /// training.
  const FeatureVector<int,double> &weights() const { return weights_; }
  /// Returns the feature vector corresponding to the averaged perceptron.
  const FeatureVector<int,double> &average_weights() const {
    return average_weights_;
  }

  /// Returns either the raw or averaged feature vector, depending on the
  /// argument.
  ///
  /// \param raw if true, return the raw model; otherwise, return the
  ///            averaged perceptron model
  const FeatureVector<int,double> &GetModel(bool raw) const {
    return raw ? weights() : average_weights();
  }

  // mutators

  /// Increments the weights for the specified collection of features.
  /// Technically, this method adds a scaled version of the specified vector
  /// projected into the subspace specified by the collection of feature uid's.
  ///
  /// \param time           the current training time
  /// \param feature_uids   the set of features to be updated
  ///                       (essentially a subspace into which to
  ///                       project <tt>feature_vector</tt>)
  /// \param feature_vector the feature vector to be projected into the
  ///                       specified subspace, scaled and then added to the
  ///                       current set of model weights
  /// \param scalar         the amount by which to scale the specified subvector
  ///
  /// \see FeatureVector::AddScaledSubvector
  template <typename Collection>
  void UpdateWeights(const Time &time,
                     const Collection &feature_uids,
                     const FeatureVector<int,double> &feature_vector,
                     double scalar) {
    weights_.AddScaledSubvector(feature_uids, feature_vector, scalar);
  }

  /// Updates the feature averages the specified pair of feature uid
  /// collections, one for a gold reference training instance and the
  /// other for a one-best hypothesis.
  template <typename Collection>
  void UpdateGoldAndCandidateFeatureAverages(const Time &time,
                                             const Collection &
                                               gold_feature_uids,
                                             const Collection &
                                               candidate_feature_uids) {
    UpdateFeatureAverages(time,
                          gold_feature_uids.begin(),
                          gold_feature_uids.end());
    UpdateFeatureAverages(time,
                          candidate_feature_uids.begin(),
                          candidate_feature_uids.end());
  }

  void UpdateAllFeatureAverages(const Time &time) {
    // Get set union of non-zero feature weights and average feature weights.
    unordered_set<int> uids;
    weights_.GetNonZeroFeatures(uids);
    average_weights_.GetNonZeroFeatures(uids);
    UpdateFeatureAverages(time, uids.begin(), uids.end());
  }

  void RemapFeatureUids(const unordered_map<int, int> &old_to_new_uids) {
    RemapFeatureUids(old_to_new_uids, weights_);
    RemapFeatureUids(old_to_new_uids, average_weights_);
    RemapFeatureUids(old_to_new_uids, weight_sums_);
    RemapFeatureUids(old_to_new_uids, last_update_indices_);
  }

  // I/O methods
  friend ostream &operator<<(ostream &os, const TrainingVectorSet &tvs) {
    os << "weights: " << tvs.weights_ << "\n"
       << "average weights: " << tvs.average_weights_ << "\n"
       << "weight sums: " << tvs.weight_sums_ << "\n"
       << "last update indices: " << tvs.last_update_indices_ << "\n";
    return os;
  }

 private:

  template <typename V>
  void RemapFeatureUids(const unordered_map<int, int> &old_to_new_uids,
                        FeatureVector<int, V> &vector) {
    FeatureVector<int, V> old_vector = vector;
    vector.clear();
    typedef typename FeatureVector<int, V>::const_iterator fv_const_iterator;
    for (fv_const_iterator old_vector_it = old_vector.begin();
         old_vector_it != old_vector.end();
         ++old_vector_it) {
      int old_uid = old_vector_it->first;
      unordered_map<int, int>::const_iterator old_to_new_uid_it =
          old_to_new_uids.find(old_uid);
      if (old_to_new_uid_it != old_to_new_uids.end()) {
        int new_uid = old_to_new_uid_it->second;
        V old_value = old_vector_it->second;
        vector.SetWeight(new_uid, old_value);
      }
    }
  }

  /// Updates the average value for the feature with the specified uid.
  /// \param time the current training time
  /// \param uid the uid of the feature whose average is to be updated
  void UpdateAverage(const Time &time, int uid) {
    int iterations_since_update =
        time.absolute_index() - last_update_indices_.GetValue(uid);
    if (iterations_since_update <= 0) {
      return;
    }
    // Add in perceptron values to weight sum.
    double add_to_sum = iterations_since_update * weights_.GetWeight(uid);
    double new_weight_sum = weight_sums_.IncrementWeight(uid, add_to_sum);
    average_weights_.SetWeight(uid, new_weight_sum / time.absolute_index());
    last_update_indices_.SetValue(uid, time.absolute_index());
  }

  /// Updates the feature averages for the specified collection of features.
  template <typename Iterator>
  void UpdateFeatureAverages(const Time &time,
                             Iterator feature_uids_begin_it,
                             Iterator feature_uids_end_it) {
    for (Iterator it = feature_uids_begin_it; it != feature_uids_end_it; ++it) {
      UpdateAverage(time, *it);
    }
  }

  // data members
  FeatureVector<int,double> weights_;
  FeatureVector<int,double> average_weights_;
  FeatureVector<int,double> weight_sums_;
  FeatureVector<int,int> last_update_indices_;
};

}  // namespace reranker

#endif
