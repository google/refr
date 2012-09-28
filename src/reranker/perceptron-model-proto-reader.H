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
/// \file perceptron-model-proto-reader.H
/// De-serializer for reranker::PerceptronModel instances from ModelMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_PERCEPTRON_MODEL_PROTO_READER_H_
#define RERANKER_PERCEPTRON_MODEL_PROTO_READER_H_

#include "../proto/model.pb.h"
#include "feature-vector-reader.H"
#include "perceptron-model.H"
#include "model-proto-reader.H"

namespace reranker {

using confusion_learning::ModelMessage;

/// \class PerceptronModelProtoReader
///
/// A class to construct a \link PerceptronModel \endlink from a
/// <tt>ModelMessage</tt> instance.
///
/// TODO(dbikel): Need some kind of a factory for model readers, so the
///               proper ModelProtoReader gets instantiated given a
///               particular Model subclass.
class PerceptronModelProtoReader : public ModelProtoReader {
 public:
  /// Constructs a new instance that can serialize Model instances to
  /// ModelMessage protocol buffer messages.
  PerceptronModelProtoReader() : smart_copy_(true) { }
  /// Destroys this reader.
  virtual ~PerceptronModelProtoReader() { }

  virtual void RegisterInitializers(Initializers &initializers) {
    initializers.Add("smart_copy", &smart_copy_);
  }

  /// Initializes this instance with the contents of the argument string,
  /// which should either be the empty string, or else be the string
  /// <tt>&quot;true&quot;</tt> or <tt>&quot;false&quot;</tt>, indicating
  /// whether to use &ldquo;smart copying&rdquo;.  &ldquo;Smart copying&rdquo;
  /// is when only one of the parameter vectors of a model is present
  /// in a message, and so this reader automatically copies the non-empty
  /// vector&rsquo;s contents to the empty vector.
  virtual void Init(const string &arg) {
    smart_copy_ = arg != "false";
  }

  /// De-serializes a PerceptronModel instance from a ModelMessage.
  ///
  /// \param[in]  model_message the ModelMessage from which to de-serialize
  ///                           a PerceptronModel instance
  /// \param[out] model         the PerceptronModel to be de-serialized from the
  ///                           specified ModelMessage
  virtual void Read(const ModelMessage &model_message, Model *model) const;

  /// De-serializes Features from an instance.
  ///
  /// \param[in]     is       input stream from which to read data
  /// \param[in,out] model    model to add features to
  /// \param[in]     skip_key assumes there is a key before the separator
  /// \param[in]     separator the string separator to use
  virtual void ReadFeatures(istream& is,
                            Model *model,
                            bool skip_key,
                            const string& separator) const;
 private:
  FeatureVectorReader<FeatureVector<int,double> > fv_reader_;
  bool smart_copy_;
};

}  // namespace reranker

#endif
