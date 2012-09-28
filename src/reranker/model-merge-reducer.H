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
/// Reducer classes for trainer.
///
/// \author kbhall@google.com (Keith Hall)

#ifndef RERANKER_MODEL_MERGE_REDUCER_H_
#define RERANKER_MODEL_MERGE_REDUCER_H
#include <string>
#include "../proto/dataio.h"
#include "../proto/model.pb.h"

namespace reranker {

using confusion_learning::FeatureMessage;
using confusion_learning::ModelMessage;


/// \class Reducer
///
/// Abstract base-class for a streaming reducer.
class Reducer {
 public:
   virtual int Reduce(const string& key, const string& value) = 0;
   virtual int Flush(void) = 0;
};


/// \class FeatureReducer
///
/// A reducer class which processes FeatureMessage proto buffers.
/// All features with the same feature ID are merged into a single output
/// FeatureMessage object.  This object is output to stdout.
/// Flush() must be called to output the final combined feature message.
class FeatureReducer : public Reducer {
 public:
  FeatureReducer(bool uniform_mix, double mix_denominator);
  virtual int Reduce(const string& feat_id, const string& value);
  virtual int Flush(void);

 private:
  string prev_feat_;
  FeatureMessage cur_message_;
  int num_merged_;
  ConfusionProtoIO messageio_;
  int uniform_mix_;
  int mix_denominator_;
};


/// \class ModelInfoReducer
///
/// A reducer class which processes ModelMessage protocol messages and merges
/// them into a single message.
/// Flush() must be called to output the final combined message.
class ModelInfoReducer : public Reducer {
 public:
  ModelInfoReducer()
    : new_model_message_(true) {
  }
  virtual int Reduce(const string& key, const string& value);
  virtual int Flush(void);

  static const char* kModelMessageFeatureName;

 private:
  bool new_model_message_;
  ConfusionProtoIO messageio_;
  ModelMessage model_message_;
};

/// \class SymbolReducer
///
/// A reducer class which processes SymbolMessage messages and returns a set of
/// unique 
/// them into a single message.
/// Flush() must be called to output the final combined message.
class SymbolReducer : public Reducer {
 public:
  SymbolReducer() { }
  virtual int Reduce(const string& key, const string& value);
  virtual int Flush(void) {
    return 0;
  };

 private:
  string prev_sym_;
};

}

#endif
