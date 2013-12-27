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
/// Interface for serializer for reranker::Model instances to ModelMessage
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_MODEL_PROTO_WRITER_H_
#define RERANKER_MODEL_PROTO_WRITER_H_

#include <cstdio>
#include <iostream>
#include <memory>

#include "../proto/model.pb.h"
#include "../proto/dataio.h"
#include "factory.H"
#include "feature-vector-writer.H"
#include "model.H"

namespace reranker {

using confusion_learning::FeatureVecMessage;
using confusion_learning::ModelMessage;
using std::cerr;
using std::endl;
using std::shared_ptr;

/// \class ModelProtoWriter
///
/// A class to construct a ModelMessage from a Model instance.
class ModelProtoWriter : public FactoryConstructible {
 public:
  /// Constructs a new instance that can serialize Model instances to
  /// ModelMessage protocol buffer messages.
  ModelProtoWriter() { }
  /// Destroys this writer.
  virtual ~ModelProtoWriter() { }

  /// Serializes a Model instance to a ModelMessage.
  ///
  /// \param model                 the \link Model \endlink to be serialized
  ///                              to the specified ModelMessage
  /// \param model_message         the ModelMessage to be filled in
  ///                              by this method with the serialized version
  ///                              of the specified Model
  /// \param write_features        whether the features of the specified
  ///                              \link Model \endlink should be written out
  virtual void Write(const Model *model, ModelMessage *model_message,
                     bool write_features = true) const = 0;

  /// Writes out the features of this model to a series of
  /// <tt>FeatureMessage</tt> instances using the specified
  /// <tt>ConfusionProtoIO</tt> instance.
  ///
  /// \param model             the \link Model \endlink to be serialized to the
  ///                          specified <tt>ModelMessage</tt>
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
                             bool output_best_epoch = false,
                             double weight = 1.0,
                             bool output_key = false,
                             const string separator = "\t") const = 0;
};

/// Registers the \link reranker::ModelProtoWriter ModelProtoWriter
/// \endlink implementation with the specified subtype <tt>TYPE</tt>
/// and <tt>NAME</tt> with the \link reranker::ModelProtoWriter
/// ModelProtoWriter \endlink \link reranker::Factory Factory\endlink.
#define REGISTER_NAMED_MODEL_PROTO_WRITER(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,ModelProtoWriter)

/// Registers the \link reranker::ModelProtoWriter ModelProtoWriter
/// \endlink implementation with the specified subtype <tt>TYPE</tt>
/// with the ModelProtoWriter \link reranker::Factory Factory\endlink.
#define REGISTER_MODEL_PROTO_WRITER(TYPE) \
  REGISTER_NAMED_MODEL_PROTO_WRITER(TYPE,TYPE)

/// \class EndOfEpochModelWriter
///
/// An end-of-epoch hook for writing out the best model so far to file
/// after each epoch (if the best model changes from the last time it
/// was written out).
///
/// \see Model::set_end_of_epoch_hook
class EndOfEpochModelWriter : public Model::Hook {
 public:
  /// Constructs a new instance to write out the best model to the specified
  /// file.  A model will only be serialized if the best model epoch so far is
  /// different from that of the last model written out by this hook.
  ///
  /// \param model_file the name of the file to which to serialize a
  ///                   \link Model \endlink
  /// \param writer     the serializer for serializing a \link Model \endlink
  ///                   to a <tt>ModelMessage</tt>
  /// \param compressed whether to use compression when serializing a
  ///                   \link Model \endlink
  /// \param use_base64 whether to serialize using base64 encoding
  /// \param verbose    whether this end-of-epoch hook should be verbose
  ///
  /// \see Model::set_end_of_epoch_hook
  EndOfEpochModelWriter(const string &model_file,
                        shared_ptr<ModelProtoWriter> writer,
                        bool compressed,
                        bool use_base64,
                        bool verbose = true) :
      writer_(writer), model_file_(model_file),
      compressed_(compressed),
      use_base64_(use_base64), verbose_(verbose), prev_best_epoch_index_(-1) { }

  /// Executes this end-of-epoch hook, which writes out the best model
  /// so far if its epoch differs from that of the last model written
  /// out by this hook.  For safety, the model will be first serialized
  /// to a temporary file and then moved into place.
  ///
  /// \param model the model that will execute this hook in its \link
  ///              Model::EndOfEpoch \endlink method
  ///
  /// \see Model::set_end_of_epoch_hook
  virtual void Do(Model *model) {
    if (model->best_model_epoch() == prev_best_epoch_index_) {
      return;
    }
    if (verbose_) {
      cerr << "Writing out model from epoch " << model->best_model_epoch()
           << " to file \"" << model_file_ << "\"...";
      cerr.flush();
    }
    // First, serialize Model to a ModelMessage.
    confusion_learning::ModelMessage model_message;
    writer_->Write(model, &model_message);
    // Next, write out ModelMessage to file.
    string tmp_file = model_file_ + ".tmp";
    ConfusionProtoIO proto_writer(tmp_file, ConfusionProtoIO::WRITE,
                                  compressed_, use_base64_);
    proto_writer.Write(model_message);
    proto_writer.Close();
    rename(tmp_file.c_str(), model_file_.c_str());
    // Record which model epoch we've written out.
    prev_best_epoch_index_ = model->best_model_epoch();
    if (verbose_) {
      cerr << " done." << endl;
    }
  }
 private:
  shared_ptr<ModelProtoWriter> writer_;
  string model_file_;
  bool compressed_;
  bool use_base64_;
  bool verbose_;
  int prev_best_epoch_index_;
};

}  // namespace reranker

#endif
