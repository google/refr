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
/// Provides the \link reranker::ModelReader ModelReader \endlink class, which
/// can create \link reranker::Model Model \endlink instances from a file.
/// \author dbikel@google.com (Dan Bikel)

#include <string>

#include "../proto/model.pb.h"
#include "factory.H"
#include "model-proto-reader.H"

#ifndef RERANKER_MODEL_READER_H_
#define RERANKER_MODEL_READER_H_

namespace reranker {

using confusion_learning::ModelMessage;
using std::string;

/// \class ModelReader
///
/// Knows how to create \link Model \endlink instances that have been serialized
/// to a file.
class ModelReader {
 public:
  ModelReader(int verbosity = 0) : verbosity_(verbosity) { }

  shared_ptr<Model> Read(const string &filename,
                         bool compressed, bool use_base64) {
    ConfusionProtoIO proto_reader(filename, ConfusionProtoIO::READ,
                                  compressed, use_base64);
    ModelMessage model_message;
    if (verbosity_ >= 1) {
      cerr << "ModelReader: reading serialized model from \""
           << filename << "\"...";
      cerr.flush();
    }
    if (!proto_reader.Read(&model_message)) {
      cerr << "ModelReader: unable to read serialized model from \""
           << filename << "\"." << endl;
      return shared_ptr<Model>();
    }
    shared_ptr<ModelProtoReader> model_reader =
        GetModelProtoReader(model_message);
    shared_ptr<Model> model =
        model_factory_.CreateOrDie(model_message.model_spec(),"model spec");
    if (model.get() == NULL) {
      return model;
    }
    model_reader->ReadFeatures(*(proto_reader.inputstream()), model.get());

    if (verbosity_ >= 1) {
      cerr << "done." << endl
           << "Loaded model \"" << model->name() << "\"." << endl;
    }

    return model;
  }

  shared_ptr<Model> Read(const ModelMessage &model_message) {
    shared_ptr<ModelProtoReader> model_proto_reader =
        GetModelProtoReader(model_message);
    shared_ptr<Model> model =
        model_factory_.CreateOrDie(model_message.model_spec(), "model spec");
    model_proto_reader->Read(model_message, model.get());
    return model;
  }
 private:
  shared_ptr<ModelProtoReader>
  GetModelProtoReader(const ModelMessage &model_message) {
    if (!model_message.has_reader_spec()) {
      cerr << "ModelReader: no reader_spec in ModelMessage" << endl;
      return shared_ptr<ModelProtoReader>();
    }
    string model_reader_spec = model_message.reader_spec();
    Factory<ModelProtoReader> model_proto_reader_factory;
    return
        model_proto_reader_factory_.CreateOrDie(model_reader_spec,
                                                "model proto reader spec");
  }

  Factory<Model> model_factory_;
  Factory<ModelProtoReader> model_proto_reader_factory_;
  int verbosity_;
};

}  // namespace reranker

#endif
