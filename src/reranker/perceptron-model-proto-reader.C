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
/// Implementation of de-serializer for reranker::PerceptronModel
/// instances from ModelMessage instances.
/// \author dbikel@google.com (Dan Bikel)

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include "../proto/model.pb.h"
#include "training-vector-set.H"
#include "perceptron-model-proto-reader.H"

namespace reranker {

REGISTER_MODEL_PROTO_READER(PerceptronModelProtoReader)

using confusion_learning::FeatureMessage;
using confusion_learning::SymbolTableMessage;
using confusion_learning::SymbolMessage;

void
PerceptronModelProtoReader::Read(const ModelMessage &model_message,
                                 Model *model) const {
  PerceptronModel *perceptron_model = static_cast<PerceptronModel *>(model);
  perceptron_model->name_ = model_message.identifier();
  perceptron_model->best_model_epoch_ = model_message.num_iterations();
  perceptron_model->time_ = Time(perceptron_model->best_model_epoch_, -1, -1);
  // TODO(dbikel): Emit warning if model_message.has_symbols() returns true
  //               when perceptron_model->symbols_ is NULL?
  if (perceptron_model->symbols_ != NULL && model_message.has_symbols()) {
    const SymbolTableMessage &symbol_table_message = model_message.symbols();
    for (int i = 0; i < symbol_table_message.symbol_size(); ++i) {
      const SymbolMessage &symbol_message = symbol_table_message.symbol(i);
      perceptron_model->symbols_->SetIndex(symbol_message.symbol(),
                                           symbol_message.index());
    }
  }
  // TODO(dbikel): De-serialize model loss.
  if (model_message.has_raw_parameters()) {
    fv_reader_.Read(model_message.raw_parameters(),
                    perceptron_model->best_models_.weights_,
                    perceptron_model->symbols());
  }
  if (model_message.has_avg_parameters()) {
    fv_reader_.Read(model_message.avg_parameters(),
                    perceptron_model->best_models_.average_weights_,
                    perceptron_model->symbols());
  }
  // Do "smart copying".
  if (smart_copy_) {
    if (perceptron_model->best_models_.weights_.size() == 0 &&
        perceptron_model->best_models_.average_weights_.size() > 0) {
      perceptron_model->best_models_.weights_ =
          perceptron_model->best_models_.average_weights_;
    } else if (perceptron_model->best_models_.average_weights_.size() == 0 &&
               perceptron_model->best_models_.weights_.size() > 0) {
      perceptron_model->best_models_.average_weights_ = 
          perceptron_model->best_models_.weights_;
    }
  }

  // Finally, make sure best_models_ is copied to models_.
  perceptron_model->models_ = perceptron_model->best_models_;
}

void PerceptronModelProtoReader::ReadFeatures(istream& is,
                                              Model *model,
                                              bool skip_key,
                                              const string& separator) const {
  PerceptronModel *perceptron_model = dynamic_cast<PerceptronModel *>(model);
  TrainingVectorSet &features = perceptron_model->best_models_;
  Symbols *symbols = perceptron_model->symbols();
  ConfusionProtoIO proto_reader;
  string buffer;
  while (is && is.good()) {
    getline(is, buffer);
    if (buffer.empty()) {
      break;
    }
    if (skip_key) {
      size_t seppos = buffer.find(separator);
      if (seppos != string::npos) {
        buffer.erase(0, seppos+1);
      }
    }
    FeatureMessage feature_msg;
    if (!proto_reader.DecodeBase64(buffer, &feature_msg)) {
      cerr << "Error decoding: " << feature_msg.Utf8DebugString() << endl;
      continue;
    }
    int uid = feature_msg.id();
    if (symbols != NULL &&
        feature_msg.has_name() && !feature_msg.name().empty()) {
      uid = symbols->GetIndex(feature_msg.name());
    }
    double value = feature_msg.value();
    if (isnan(value)) {
        cerr << "PerceptronModelProtoReader: WARNING: feature "
             << uid << " has value that is NaN" << endl;
    } else {
      features.weights_.IncrementWeight(uid, value);
    }
    if (feature_msg.has_avg_value()) {
      double avg_value = feature_msg.avg_value();
      if (isnan(avg_value)) {
        cerr << "PerceptronModelProtoReader: WARNING: feature "
             << uid << " has avg_value that is NaN" << endl;
      } else {
        features.average_weights_.IncrementWeight(uid, avg_value);
      }
    }
  }
  // Do "smart copying".
  if (smart_copy_) {
    if (features.weights_.size() == 0 && features.average_weights_.size() > 0) {
      features.weights_ = features.average_weights_;
    } else if (features.average_weights_.size() == 0 &&
               features.weights_.size() > 0) {
      features.average_weights_ = features.weights_;
    }
  }
  // Make sure to copy latest model to models_.
  perceptron_model->models_ = features;
}

}  // namespace reranker
