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
/// \file feature-vector-writer.H
/// Serializer for reranker::FeatureVector instances to FeatureVecMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_FEATURE_VECTOR_WRITER_H_
#define RERANKER_FEATURE_VECTOR_WRITER_H_

#include <string>

#include "../proto/model.pb.h"
#include "feature-vector.H"
#include "symbol-table.H"

namespace reranker {

using confusion_learning::FeatureMessage;
using confusion_learning::FeatureVecMessage;
using confusion_learning::FeatureMessage_FeatureType;

using std::string;

/// \class FeatureVectorWriter
///
/// A class to serialize FeatureVector instances to FeatureVecMessage instances.
///
/// \tparam the type of FeatureVector to write
template <typename FV>
class FeatureVectorWriter {
 public:
  /// Serializes a single feature-value pair to a <tt>FeatureMessage</tt>.
  ///
  /// \param[in]  uid             the uid of this feature-value pair
  /// \param[in]  value           the value of this feature-value pair
  /// \param[in] feature_type     the type of features contained in the
  ///                             specified feature vector to be
  ///                             serialized (specified in
  ///                             model.proto)
  /// \param[out] feature_message the <tt>FeatureMessage</tt> instance
  ///                             to fill in
  /// \param[in]  symbols         a pointer to an optional Symbols instance
  ///                             to be used when determining the symbol of a
  ///                             feature&rsquo;s uid
  void SerializeFeature(const typename FV::key_type &uid,
                        const typename FV::mapped_type &value,
                        FeatureMessage_FeatureType feature_type,
                        FeatureMessage *feature_message,
                        Symbols *symbols = NULL) const {
    feature_message->set_id(uid);
    feature_message->set_type(feature_type);
    feature_message->set_value(value);
    if (symbols != NULL) {
      const string &name = symbols->GetSymbol(uid);
      if (name != "") {
        feature_message->set_name(name);
      }
    }
  }

  /// Serializes the specified feature vector to the specified
  /// <tt>FeatureVecMessage</tt> protocol buffer message.
  ///
  /// \param[in]  features     the feature vector to serialize
  /// \param[in]  feature_type the type of features contained in the specified
  ///                          feature vector to be serialized (specified in
  ///                          model.proto)
  /// \param[out] fv_message   the FeatureVecMessage instance to fill in with
  ///                          the serialized version of the specified feature
  ///                          vector
  /// \param[in]  symbols      a pointer to an optional Symbols instance
  ///                          to be used when determining the symbol of a
  ///                          feature&rsquo;s uid
  void Write(const FV &features,
             FeatureMessage_FeatureType feature_type,
             FeatureVecMessage *fv_message,
             Symbols *symbols = NULL) const {
    for (typename FV::const_iterator feature_it = features.begin();
         feature_it != features.end();
         ++feature_it) {
      FeatureMessage *feature_message = fv_message->add_feature();
      SerializeFeature(feature_it->first,
                       feature_it->second,
                       feature_type,
                       feature_message,
                       symbols);
    }
  }
};

/// Partial specialization of the FeatureVectorWriter class for feature vectors
/// whose unique identifiers for features are string objects.
template <typename V>
class FeatureVectorWriter<FeatureVector<string,V> >  {
 public:
  void Write(const FeatureVector<string,V> &features,
             confusion_learning::FeatureMessage_FeatureType feature_type,
             FeatureVecMessage *fv_message) const {
    typedef typename FeatureVector<string,V>::const_iterator const_iterator;
    for (const_iterator feature_it = features.begin();
         feature_it != features.end();
         ++feature_it) {
      FeatureMessage *feature_message = fv_message->add_feature();
      // Note that, crucially, we use FeatureMessage::set_name and not
      // FeatureMessage::set_id, as above.
      feature_message->set_name(feature_it->first);
      feature_message->set_type(feature_type);
      feature_message->set_value(feature_it->second);
    }
  }
};

}  // namespace reranker

#endif

