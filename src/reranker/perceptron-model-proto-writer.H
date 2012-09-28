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
/// Serializer for reranker::PerceptronModel instances to ModelMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_PERCEPTRON_MODEL_PROTO_WRITER_H_
#define RERANKER_PERCEPTRON_MODEL_PROTO_WRITER_H_

#include "../proto/model.pb.h"
#include "feature-vector-writer.H"
#include "perceptron-model.H"
#include "model-proto-writer.H"

namespace reranker {

using confusion_learning::ModelMessage;

/// \class PerceptronModelProtoWriter
///
/// A class to construct a <tt>ModelMessage</tt> from a \link
/// PerceptronModel \endlink instance.
///
/// TODO(dbikel): Need some kind of a factory for model writers, so the
///               proper ModelProtoWriter gets instantiated given a
///               particular Model subclass.
///
class PerceptronModelProtoWriter : public ModelProtoWriter {
 public:
  /// Constructs a new instance that can serialize Model instances to
  /// ModelMessage protocol buffer messages.
  PerceptronModelProtoWriter() { }
  /// Destroys this writer.
  virtual ~PerceptronModelProtoWriter() { }

  /// Serializes a PerceptronModel instance to a ModelMessage.  When
  /// the <tt>write_features</tt> parameter is <tt>true</tt>, the
  /// feature vectors written out will be those of the best model
  /// epoch (typically the epoch with the lowest average devtest loss).
  ///
  /// \param model                 the \link PerceptronModel \endlink to be
  ///                              serialized to the specified
  ///                              <tt>ModelMessage</tt>
  /// \param model_message         the ModelMessage to be filled in
  ///                              by this method with the serialized version
  ///                              of the specified Model
  /// \param write_features        whether the features of the specified
  ///                              \link Model \endlink should be written out
  virtual void Write(const Model *model, ModelMessage *model_message,
                     bool write_features) const;

  /// Writes out the features of this model to a series of
  /// <tt>FeatureMessage</tt> instances using the specified
  /// <tt>ConfusionProtoIO</tt> instance.  This method writes out both the raw
  /// feature values as well as the averaged values; however, the averaged
  /// values are stored in the avg_value field of the FeatureMessage object.
  /// In contrast to the behavior of the \link Write \endlink method, 
  /// the features written out by this method will be those of the 
  /// current model, as opposed to those of the best model.
  ///
  /// \param model             the \link PerceptronModel \endlink to be
  ///                          serialized to the specified <tt>ModelMessage</tt>
  /// \param os                the output stream to which to write features
  /// \param output_best_epoch output the weights of the best epoch of
  ///                          training, as opposed to the most recent epoch
  /// \param weight            the weight by which all feature weights should be
  ///                          multiplied before being output
  /// \param output_key        output the key (feature name) and separator for
  ///                          each feature
  /// \param separator         the separator string to output between each
  ///                          feature&rsquo;s key and its base64-encoded
  ///                          <tt>FeatureMessage</tt>
  ///                          (only used if output_key == True)
  virtual void WriteFeatures(const Model *model,
                             ostream &os,
                             bool output_best_epoch,
                             double weight,
                             bool output_key,
                             const string separator) const;
 private:
  FeatureVectorWriter<FeatureVector<int,double> > fv_writer_;
};

}  // namespace reranker

#endif
