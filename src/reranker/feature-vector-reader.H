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
/// \file feature-vector-reader.H
/// De-serializer for reranker::FeatureVector instances from FeatureVecMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_FEATURE_VECTOR_READER_H_
#define RERANKER_FEATURE_VECTOR_READER_H_

#include <cmath>
#include <string>

#include "../proto/model.pb.h"
#include "feature-vector.H"
#include "symbol-table.H"

namespace reranker {

using confusion_learning::FeatureMessage;
using confusion_learning::FeatureVecMessage;

using std::string;

/// \class FeatureVectorReader
///
/// A class to de-serialize FeatureVector instances from
/// FeatureVecMessage instances.
///
/// \tparam the type of FeatureVector to de-serialize
template <typename FV>
class FeatureVectorReader {
 public:
  /// Serializes the specified feature vector to the specified
  /// FeatureVecMessage protocol buffer message.
  ///
  /// \param[in]     fv_message the FeatureVecMessage instance from which to
  ///                           read the serialized version of a feature vector
  /// \param[out]    features   the feature vector to be de-serialized from the
  ///                           specified FeatureVecMessage
  /// \param[in,out] symbols    the symbol table to get an <tt>int</tt> feature
  ///                           uid when a <tt>FeatureMessage</tt> has its
  ///                           <tt>name</tt> field set
  void Read(const FeatureVecMessage &fv_message,
            FV &features,
            Symbols *symbols) const {
    for (int i = 0; i < fv_message.feature_size(); ++i) {
      const FeatureMessage &feature_msg = fv_message.feature(i);
      int uid = feature_msg.id();
      if (symbols != NULL &&
          feature_msg.has_name() && ! feature_msg.name().empty()) {
        uid = symbols->GetIndex(feature_msg.name());
      }
      double value = feature_msg.value();
      if (std::isnan(value)) {
        cerr << "FeatureVectorReader: WARNING: feature " << uid
             << " has value that is NaN" << endl;
      } else {
        features.IncrementWeight(uid, value);
      }
    }
  }
};

/// Partial specialization of the FeatureVectorReader class for feature vectors
/// whose unique identifiers for features are string objects.
template <typename V>
class FeatureVectorReader<FeatureVector<string,V> >  {
 public:
  void Read(const FeatureVecMessage &fv_message,
            FeatureVector<string,V> &features,
            Symbols *symbols) const {
    for (int i = 0; i < fv_message.feature_size(); ++i) {
      const FeatureMessage &feature_msg = fv_message.feature(i);
      double value = feature_msg.value();
      if (std::isnan(value)) {
        cerr << "FeatureVectorReader: WARNING: feature " << feature_msg.name()
             << " has value that is NaN" << endl;
      } else {
        features.IncrementWeight(feature_msg.name(), value);
      }
    }
  }
};

}  // namespace reranker

#endif

