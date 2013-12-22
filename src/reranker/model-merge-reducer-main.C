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
/// MapReduce Reducer to merge the features from multiple models.
/// - Input: Sorted list of FeatureMessage protocol buffer message.
///          <FeatureId, encoded FeatureMessage>
/// - Output: Mixed features.
///          <FeatureId, encoded FeatureMessage>
/// \author kbhall@google.com (Keith Hall)

#include <cstdio>
#include <iostream>
#include <string>
#include <unistd.h>
#include "../utils/kdebug.h"
#include "model-merge-reducer.H"

using namespace std;
using reranker::Reducer;
using reranker::FeatureReducer;
using reranker::ModelInfoReducer;
using reranker::SymbolReducer;

int main(int argc, char* argv[]) {
  int option_char;
  bool uniform_mix = false;
  int mix_denominator = 1.0;
  bool reduce_symbols = false;

  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "ud:S")) != EOF) {
    switch (option_char) {  
    case 'S':
      reduce_symbols = true;
    case 'u':
      uniform_mix = true;
      break;
    case 'd':
      mix_denominator = atoi(optarg);
      break;
    case '?':
      cerr << "usage: " << argv[0] << "[-K] [-u] [-d denom]" << endl;
      cerr << "-u - mix the features uniformly (overrides -d)" << endl;
      cerr << "-d - normalize mixture with this value" << endl;
      cerr << "-S - Run this in symbol reducer mode (unique)" << endl;
      return -1;
      break;
    }
  }

  FeatureReducer feat_reducer(uniform_mix, mix_denominator);
  ModelInfoReducer model_reducer;
  SymbolReducer sym_reducer;
  string empty_string;

  // Process each of the input records.  This reducer assumes that the input is
  // <FeatureId, encoded FeatureMessage> pair per line in the following format:
  // FeatureString \t EncodedMsg \n
  while (cin) {
    // Process input.
    string input_data;
    getline(cin, input_data);
    if (input_data.empty()) {
      break;
    }
    if (reduce_symbols) {
      sym_reducer.Reduce(input_data, empty_string);
    } else {
      int delim_pos = input_data.find('\t');
      string feat_id = input_data.substr(0, delim_pos);
      string value = input_data.substr(delim_pos + 1);

      Reducer* reducer;
      if (feat_id.compare(ModelInfoReducer::kModelMessageFeatureName) == 0) {
        reducer = &model_reducer;
      } else {
        reducer = &feat_reducer;
      }
      reducer->Reduce(feat_id, value);
    }
  }
  if (!reduce_symbols) {
    feat_reducer.Flush();
    model_reducer.Flush();
  } else {
    sym_reducer.Flush();
  }
  return 0;
}
