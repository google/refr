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
/// \file feature-vector.H
/// Defines the reranker::FeatureVector class, which, as it happens,
/// is used to store feature vectors.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_FEATURE_VECTOR_H_
#define RERANKER_FEATURE_VECTOR_H_

#include <iostream>
#include <map>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <vector>

#include "string-canonicalizer.H"

namespace reranker {

using std::cerr;
using std::endl;
using std::map;
using std::ostream;
using std::string;
using std::tr1::unordered_map;
using std::tr1::unordered_set;

/// \class UidGetter
///
/// A simple class that provides a layer of abstraction when retrieving
/// objects to represent unique identifiers for features.
///
/// \tparam T the type of object representing a feature&rsquo;s uid
template<typename T>
struct UidGetter {
  /// Gets the canonical uid object for the specified uid object.
  static const T &Get(const T &uid) { return uid; }
};

/// A specialization for when feature uid&rsquo;s are <tt>string</tt>
/// objects, where \link StringCanonicalizer::Get \endlink is used to
/// provide a canonical string instance.
template<>
struct UidGetter<string> {
  /// Gets the canonical uid string for the specified uid string.
  static const string &Get(const string &uid) {
    return StringCanonicalizer::Get(uid);
  }
};

/// \class FeatureVector
///
/// A class to represent a feature vector, where features are
/// represented by unique identifiers, and feature values are
/// represented by the template type.
///
/// \tparam K the type to represent unique identifiers for each feature
/// \tparam V the value or weight of a feature in this vector
template <typename K, typename V, typename Map = map<K, V> >
class FeatureVector {
 public:
  /// Create an empty feature vector.
  FeatureVector() { }

  /// Copy features from the type of map used internally.
  ///
  /// \param features the map of features with which to initialize this
  ///                 feature vector
  FeatureVector(const Map &features) : features_(features) { }

  /// Copy features from any map, or any collection of (feature,value) pairs,
  /// for that matter.
  ///
  /// \tparam MapType the type of map from which to copy features into this
  ///                 feature vector
  /// \param features the map or collection of (feature,value) pairs with which
  ///                 to initialize this feature vector
  template <typename MapType>
  FeatureVector(const MapType &features) {
    for (typename MapType::const_iterator it = features.begin();
         it != features.end(); ++it) {
      features_.insert(*it);
    }
  }

  virtual ~FeatureVector() {}

  // Forward type declarations that mirror what STL maps do.
  /// The type of vector component/feature uid's in this vector.
  typedef K key_type;
  /// The type of values/feature weights in this vector.
  typedef V mapped_type;

  /// The underlying type that this class stores the mapping of
  /// feature uid's to their weights.
  typedef Map FeatureMap;
  /// The type of const iterator for the feature-weight pairs in this vector.
  typedef typename FeatureMap::const_iterator const_iterator;

  // accessors

  /// Returns a const iterator pointing to the first of the
  /// feature-value pairs of this feature vector.
  const_iterator begin() const {
    return features_.begin();
  }

  /// Returns a const iterator pointing to the end of the feature-value
  /// pairs of this feature vector.
  const_iterator end() const {
    return features_.end();
  }

  /// Inserts the uid's of features with non-zero weights into the
  /// specified set.
  /// \attention
  /// The specified is <i>not</i> cleared at any time by this method.
  ///
  /// \param[out] set the set in which to insert uid's of all features with
  ///             non-zero weight
  /// \return the specified set, having been modified by this method
  unordered_set<K> &GetNonZeroFeatures(unordered_set<K> &set) const {
    for (const_iterator it = features_.begin(); it != features_.end(); ++it) {
      set.insert(it->first);
    }
    return set;
  }

  /// Removes the uid's of features with non-zero weights from the
  /// specified set.
  /// \attention
  /// The specified is <i>not</i> cleared at any time by this method.
  ///
  /// \param[out] set the set from which to remove uid's of all features with
  ///             non-zero weight
  /// \return the specified set having been modified
  unordered_set<K> &RemoveNonZeroFeatures(unordered_set<K> &set) const {
    for (const_iterator it = features_.begin(); it != features_.end(); ++it) {
      set.erase(it->first);
    }
    return set;
  }

  /// Removes from the specified set the uid's of feature with weights
  /// equal in this vector to their weights in the specified vector.
  ///
  /// \param[in]  other the vector whose feature weights are to be
  ///                   compared to those in this vector
  /// \param[out] set   the set to be modified by this method
  /// \return the specified set having been modified
  unordered_set<K> &RemoveEqualFeatures(const FeatureVector<K,V> &other,
                                        unordered_set<K> &set) const {
    for (const_iterator it = features_.begin(); it != features_.end(); ++it) {
      const K &uid = it->first;
      if (set.find(uid) != set.end() &&
          GetWeight(uid) == other.GetWeight(uid)) {
        set.erase(uid);
      }
    }
    return set;
  }

  /// Returns the weight of the feature with the specified uid, where crucially
  /// features not "present" in this vector implicitly have a weight of 0.0
  /// (or whatever the default constructor of the value type is).
  ///
  /// \param uid the uid of the feature whose weight is to be retrieved
  V GetWeight(const K &uid) const {
    const_iterator feature_it = features_.find(uid);
    return feature_it == features_.end() ? V() : feature_it->second;
  }

  /// Synonymous with \link GetWeight \endlink.
  ///
  /// \param uid the uid of the feature whose value is to be retrieved
  V GetValue(const K &uid) const {
    return GetWeight(uid);
  }

  /// Returns the number of non-zero feature components of this feature vector.
  size_t size() const { return features_.size(); }

  bool empty() const { return features_.empty(); }

  /// Computes the dot product of this feature vector with the
  /// specified FeatureVector.  This primitive operation is one out of
  /// which a multitude of kernel functions may be constructed.
  V Dot(const FeatureVector<K,V> &other) const {
    bool this_is_smaller = size() < other.size();
    const FeatureVector &small = this_is_smaller ? *this : other;
    const FeatureVector &large = this_is_smaller ? other : *this;
    V dot_product = V();
    for (const_iterator it = small.features_.begin();
         it != small.features_.end();
         ++it) {
      dot_product += it->second * large.GetWeight(it->first);
    }
    return dot_product;
  }

  // mutators

  /// Increments the weight of the specified feature by the
  /// specified amount.  This method is synonymous with
  /// \link IncrementValue \endlink.
  ///
  /// \param uid the uid of the feature whose weight is to be incremented
  /// \param by the amount by which to increment the specified feature's value
  /// \return the new value for the specified feature
  V IncrementWeight(const K &uid, V by) {
    const K &canonical_uid = UidGetter<K>::Get(uid);
    V default_val = V();  // zero
    // Short circuit: if we're trying to increase a feature's weight by
    // zero, bail out early.
    bool no_increase = by == default_val;
    iterator feature_it = features_.find(canonical_uid);
    if (feature_it == features_.end()) {
      if (no_increase) {
        return default_val;
      }
      features_[canonical_uid] = by;
      return by;
    } else {
      if (no_increase) {
        return feature_it->second;
      }
      V new_weight = feature_it->second + by;
      if (new_weight == default_val) {
        features_.erase(canonical_uid);
      } else {
        feature_it->second = new_weight;
      }
      return new_weight;
    }
  }

  /// Increments the value of the specified vector component by the
  /// specified amount.  (Synonym for \link IncrementWeight \endlink.)
  ///
  /// \param uid the uid of the vector component whose value is to be
  ///            incremented
  /// \param by  the amount by which to increment the specified
  ///            component's value
  /// \return the new value for the specified vector component
  V IncrementValue(const K &uid, V by) {
    return IncrementWeight(uid, by);
  }

  /// Sets the weight of the specified feature to the specified value.
  /// \param uid        the uid of the feature whose weight is to be set
  /// \param new_weight the weight to which to set the specified feature
  /// \return the old weight for the specified feature
  V SetWeight(const K &uid, V new_weight) {
    V old_weight = GetWeight(uid);
    if (new_weight == V()) {
      features_.erase(uid);
    } else {
      features_[uid] = new_weight;
    }
    return old_weight;
  }

  /// Synonym for \link SetWeight \endlink.
  V SetValue(const K &uid, V new_value) {
    return SetWeight(uid, new_value);
  }

  /// Multiplies this vector, in situ, by the specified scalar.
  ///
  /// \param scalar the amount by which to scale this vector
  /// \return this vector, having been modified by this method
  FeatureVector<K,V> &Scale(V scalar) {
    for (iterator it = features_.begin(); it != features_.end(); ++it) {
      it->second = it->second * scalar;
    }
    return *this;
  }

  /// Modifies this vector so that it equals this vector plus the
  /// scaled specified vector.  In notation, if this vector is v1 and
  /// the specified vector is v2 and the scalar is a, then after invoking
  /// this method this vector would equal v1 + a*v2.
  ///
  /// \param other  the vector the scaled version of which is to be
  ///               added to this vector
  /// \param scalar the amount by which to scale the specified vector before
  ///               adding the result to this vector
  /// \return this vector, having been modified by this method
  FeatureVector<K,V> &AddScaledVector(const FeatureVector<K,V> &other,
                                      V scalar) {
    for (const_iterator it = other.begin(); it != other.end(); ++it) {
      IncrementWeight(it->first, scalar * it->second);
    }
    return *this;
  }

  /// Modifies this vector so that it equals this vector plus the
  /// scaled specified subvector.  The specified collection of feature
  /// uid's&mdash;the <tt>feature_uids</tt> parameter&mdash;specifies the
  /// subspace into which to project the specified feature vector&mdash;the
  /// <tt>feature_vector</tt> parameter.  This subvector is then
  /// scaled by the specified scalar&mdash;the <tt>scalar</tt>
  /// parameter&mdash;and added to this vector in situ.
  ///
  /// \param feature_uids   the subspace into which to project the specified
  ///                       feature vector before scaling and adding to this
  ///                       vector
  /// \param feature_vector the feature vector to be projected into a subspace,
  ///                       scaled and then added to this vector
  /// \param scalar         the amount by which to scale the specified subvector
  ///                       prior to adding it to this vector
  /// \return this vector, having been modified by this method
  template <typename Collection>
  FeatureVector<K,V> &AddScaledSubvector(const Collection &feature_uids,
                                         const FeatureVector<K,V> &
                                         feature_vector,
                                         V scalar) {
    for (typename Collection::const_iterator it = feature_uids.begin();
         it != feature_uids.end();
         ++it) {
      IncrementWeight(*it, feature_vector.GetWeight(*it) * scalar);
    }
    return *this;
  }

  /// Increments the weights for the specified collection of features.
  ///
  /// \param feature_uids a collection of feature uid's whose weights are to be
  ///                     incremented by the specified amount
  /// \param increment    the amount by which to increment the weight of the
  ///                     specified set of features
  /// \return this vector, having been modified by this method
  template <typename Collection>
  FeatureVector<K,V> &IncrementWeights(const Collection &feature_uids,
                                       V increment) {
    for (typename Collection::const_iterator it = feature_uids.begin();
         it != feature_uids.end();
         ++it) {
      IncrementWeight(*it, increment);
    }
    return *this;
  }

  /// Sets all feature weights to zero and, because this is a sparse vector,
  /// clears all storage.
  void clear() {
    features_.clear();
  }

  // I/O methods

  // TODO(dbikel): Add methods to serialize and de-serialize to/from protobuf's.

  friend ostream &operator<<(ostream &os, const FeatureVector<K,V> &fv) {
    os << "[";
    const_iterator it = fv.features_.begin();
    if (it != fv.features_.end()) {
      os << it->first << "=" << it->second;
      ++it;
    }
    for ( ; it != fv.features_.end(); ++it) {
      os << " " << it->first << "=" << it->second;
    }
    os << "]";
    return os;
  }

 private:
  // a helpful internal typedef for mutating methods
  typedef typename FeatureMap::iterator iterator;

  // data members
  /// Sparse vector of features stores non-zero components, as map from unique
  /// feature indices to Feature objects.
  FeatureMap features_;
};

}  // namespace reranker

#endif
