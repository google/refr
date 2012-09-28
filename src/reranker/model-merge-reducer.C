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
/// MapReduce Reducer classes for model merging.
///
/// \author kbhall@google.com (Keith Hall)

#include <iostream>
#include <string>
#include <stdlib.h>
#include "../proto/dataio.h"
#include "../utils/kdebug.h"
#include "../proto/model.pb.h"
#include "model-merge-reducer.H"

namespace reranker {

using namespace std;
using confusion_learning::FeatureMessage;
using confusion_learning::ModelMessage;

const char* ModelInfoReducer::kModelMessageFeatureName = "__MODEL_INFO_FIELD__";

FeatureReducer::FeatureReducer(bool uniform_mix, double mix_denominator)
   : num_merged_(0), uniform_mix_(uniform_mix),
     mix_denominator_(mix_denominator) {
}

int FeatureReducer::Reduce(const string& feat_id, const string& value) {
  // Decode the value as a FeatureMessage protocol buffer.
  FeatureMessage new_message;
  if (!messageio_.DecodeBase64(value, &new_message)) {
    cerr << "Error decoding message: " << value.c_str() << endl;
  }
  int num_output = 0;
  if (feat_id.compare(prev_feat_) != 0) {
    // If this is a new feature (new key), then output the previous features.
    if (!prev_feat_.empty()) {
      // Do we want to normalize the mixture with the number of mappers which
      // found this feature ?
      double normalizer = uniform_mix_ ? static_cast<double>(num_merged_) :
                          mix_denominator_;
      if (normalizer != 1.0) {
        cur_message_.set_value(cur_message_.value() / normalizer);
        cur_message_.set_avg_value(cur_message_.avg_value() / normalizer);
      }
      // Encode message and output to stdout.
      string encoded_msg;
      messageio_.EncodeBase64(cur_message_, &encoded_msg);
      cout << prev_feat_.c_str() << "\t" << encoded_msg.c_str();
    }
    // Record the new key and clear the state.
    prev_feat_ = feat_id;
    cur_message_.CopyFrom(new_message);
    num_merged_ = 1;
    num_output = 1;
  } else {
    cur_message_.set_value(cur_message_.value() + new_message.value());
    cur_message_.set_avg_value(cur_message_.avg_value() + new_message.avg_value());
    cur_message_.set_count(cur_message_.count() + new_message.count());
    num_merged_++;
  }
  // Update state.
  return num_output;
}

int FeatureReducer::Flush(void) {
  if (!prev_feat_.empty()) {
      double normalizer = uniform_mix_ ? static_cast<double>(num_merged_) :
                          mix_denominator_;
      if (normalizer != 1.0) {
        cur_message_.set_value(cur_message_.value() / normalizer);
        cur_message_.set_avg_value(cur_message_.avg_value() / normalizer);
      }
    string encoded_msg;
    messageio_.EncodeBase64(cur_message_, &encoded_msg);
    cout << prev_feat_.c_str() << "\t" << encoded_msg.c_str();
    prev_feat_.clear();
    cur_message_.Clear();
    num_merged_ = 0;
    return 1;
  }
  return 0;
}

int ModelInfoReducer::Reduce(const string& key, const string& value) {
  ModelMessage new_message;
  if (!messageio_.DecodeBase64(value, &new_message)) {
    cerr << "Error decoding message: " << value.c_str() << endl;
  }
  if (new_model_message_) {
    model_message_.CopyFrom(new_message);
    new_model_message_ = false;
    if (model_message_.has_symbols()) {
      model_message_.clear_symbols();
    }
  } else {
    model_message_.set_loss(model_message_.loss() + new_message.loss());
    model_message_.set_training_errors(
        model_message_.training_errors() + new_message.training_errors());
    if (model_message_.reader_spec().compare(new_message.reader_spec()) != 0) {
      cerr << "Combining messages with different reader_spec fields.";
      return -1;
    }
    // Check that the models being merged have the same specs.
    if (model_message_.model_spec().compare(new_message.model_spec()) != 0) {
      cerr << "Combining messages with different model_spec fields.";
      return -1;
    }
    if (model_message_.identifier().compare(new_message.identifier()) != 0) {
      cerr << "Combining messages with different identifier fields.";
      return -1;
    }
    if (model_message_.num_iterations() != new_message.num_iterations()) {
      cerr << "Combining messages with different num_iterations fields.";
      return -1;
    }
    if (model_message_.has_symbols()) {
      if (new_message.has_symbols()) {
        // Do something sensible to merge symbols ???
        // Or assume they are the same.
        // TODO(kbhall): resolve this symbols problem.
      }
    }
  }
  return 0;
}

int ModelInfoReducer::Flush(void) {
  if (new_model_message_) {
    return 0;
  }
  if (!model_message_.has_num_iterations()) {
    cerr << "No model information";
    return -1;
  }
  model_message_.set_num_iterations(model_message_.num_iterations() + 1);
  string encoded_msg;
  messageio_.EncodeBase64(model_message_, &encoded_msg);
  cout << ModelInfoReducer::kModelMessageFeatureName;
  cout << "\t" << encoded_msg.c_str();
  model_message_.Clear();
  return 1;
}

int SymbolReducer::Reduce(const string& key, const string& value) {
  if (key.compare(prev_sym_) != 0) {
    cout << key.c_str() << endl;
    prev_sym_ = key;
    return 1;
  }
  return 0;
}

}  // End of namespace.
