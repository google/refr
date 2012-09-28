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
/// \file perceptron-model-proto-writer.C
/// Implementation for serializer for reranker::PerceptronModel instances to
/// ModelMessage instances.
/// \author dbikel@google.com (Dan Bikel)

#include <sstream>

#include "../proto/dataio.h"
#include "perceptron-model-proto-writer.H"

namespace reranker {

REGISTER_MODEL_PROTO_WRITER(PerceptronModelProtoWriter)

using confusion_learning::FeatureMessage;
using confusion_learning::FeatureVecMessage;
using confusion_learning::SymbolTableMessage;
using confusion_learning::SymbolMessage;

void
PerceptronModelProtoWriter::Write(const Model *model,
                                  ModelMessage *model_message,
                                  bool write_features) const {
  const PerceptronModel *perceptron_model =
      dynamic_cast<const PerceptronModel *>(model);
  model_message->set_identifier(perceptron_model->name());
  model_message->set_reader_spec(perceptron_model->proto_reader_spec());
  model_message->set_num_iterations(perceptron_model->best_model_epoch());
  model_message->set_training_errors(perceptron_model->num_training_errors());
  model_message->set_model_spec(perceptron_model->model_spec());


  if (write_features) {
    // TODO(dbikel): Figure out exactly what quantity to serialize as
    // model loss.
    FeatureVecMessage *raw_feature_vector_message =
        model_message->mutable_raw_parameters();
    fv_writer_.Write(perceptron_model->best_models_.weights(),
                     FeatureMessage::BASIC,
                     raw_feature_vector_message);
    FeatureVecMessage *avg_feature_vector_message =
        model_message->mutable_avg_parameters();
    fv_writer_.Write(perceptron_model->best_models_.average_weights(),
                     FeatureMessage::BASIC,
                     avg_feature_vector_message);

    // Don't write the symbol table if we are not writing the features.
    if (perceptron_model->symbols_ != NULL) {
      SymbolTableMessage *symbol_table_message =
          model_message->mutable_symbols();
      Symbols *symbols = perceptron_model->symbols_;
      for (Symbols::const_iterator it = symbols->begin();
           it != symbols->end();
           ++it) {
        SymbolMessage *symbol_message = symbol_table_message->add_symbol();
        symbol_message->set_symbol(it->first);
        symbol_message->set_index(it->second);
      }
    }
  }
}

void
PerceptronModelProtoWriter::WriteFeatures(const Model *model,
                                          ostream &os,
                                          bool output_best_epoch,
                                          double weight,
                                          bool output_key,
                                          const string separator)
    const {
  ConfusionProtoIO proto_writer;
  const PerceptronModel *perceptron_model =
      dynamic_cast<const PerceptronModel *>(model);
  const FeatureVector<int, double> &avg_weights =
      output_best_epoch ?
      perceptron_model->best_models_.average_weights() :
      perceptron_model->models_.average_weights();
  const FeatureVector<int, double> &raw_weights =
      perceptron_model->models_.weights();
  FeatureMessage feature_message;
  for (FeatureVector<int, double>::const_iterator it = raw_weights.begin();
       it != raw_weights.end();
       ++it) {
    feature_message.Clear();
    Symbols *symbols = perceptron_model->symbols();
    fv_writer_.SerializeFeature(it->first, weight * it->second,
                                FeatureMessage::BASIC,
                                &feature_message,
                                symbols);
    // Lookup the average weight.
    if (!avg_weights.empty()) {
      feature_message.set_avg_value(weight * avg_weights.GetValue(it->first));
    }
    string encoded_message;
    proto_writer.EncodeBase64(feature_message, &encoded_message);
    if (output_key) {
      const string &feat_name = symbols->GetSymbol(it->first);
      if (feat_name == "") {
        stringstream ss;
        ss << it->first;
        os << ss.str() << separator;
      } else {
        os << feat_name << separator;
      }
    }
    os << encoded_message;
  }
  os.flush();
}

}  // namespace reranker
