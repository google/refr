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
/// Interface for de-serializer for reranker::Model instances from ModelMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_MODEL_PROTO_READER_H_
#define RERANKER_MODEL_PROTO_READER_H_

#include <iostream>
#include "../proto/model.pb.h"
#include "factory.H"
#include "model.H"

namespace reranker {

using confusion_learning::FeatureVecMessage;
using confusion_learning::ModelMessage;

/// \class ModelProtoReader
///
/// A class to de-serialize a Model instance from a ModelMessage instance.
class ModelProtoReader : public FactoryConstructable {
 public:
  /// Constructs a new instance that can de-serialize Model instances from
  /// ModelMessage protocol buffer messages.
  ModelProtoReader() { }
  /// Destroys this reader.
  virtual ~ModelProtoReader() { }

  /// Initializes this instance with the specified argument string
  /// The default implementation here does nothing.
  /// This method is guaranteed to be invoked after construction via
  /// a \link Factory \endlink instance.
  ///
  /// \param arg the string with which to initialize this instance.
  ///
  /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
  virtual void Init(const string &arg) { }

  /// De-serializes a Model instance from a ModelMessage.
  ///
  /// \param[in]  model_message the ModelMessage from which to de-serialize
  ///                           a Model instance
  /// \param[out] model         the Model to be de-serialized from the
  ///                           specified ModelMessage
  virtual void Read(const ModelMessage &model_message, Model *model) const = 0;

  /// De-serializes <tt>FeatureMessage</tt> instances from the specified
  /// input stream.
  ///
  /// \param[in]     is        input stream from which to read data
  /// \param[in,out] model     model to add features to
  /// \param[in]     skip_key  assumes there is a key before the separator
  /// \param[in]     separator the string separator to use
  virtual void ReadFeatures(istream& is,
                            Model *model,
                            bool skip_key = false,
                            const string& separator = "\t") const = 0;
};

/// Registers the \link reranker::ModelProtoReader ModelProtoReader
/// \endlink implementation with the specified subtype <tt>TYPE</tt>
/// and <tt>NAME</tt> with the \link reranker::ModelProtoReader ModelProtoReader
/// \endlink \link reranker::Factory Factory\endlink.
#define REGISTER_NAMED_MODEL_PROTO_READER(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,ModelProtoReader)

/// Registers the \link reranker::ModelProtoReader ModelProtoReader
/// \endlink implementation with the specified subtype <tt>TYPE</tt>
/// with the \link reranker::ModelProtoReader ModelProtoReader
/// \endlink \link reranker::Factory Factory\endlink.
#define REGISTER_MODEL_PROTO_READER(TYPE) \
  REGISTER_NAMED_MODEL_PROTO_READER(TYPE,TYPE)


}  // namespace reranker

#endif
