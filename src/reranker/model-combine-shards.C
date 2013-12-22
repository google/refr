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
/// Combine the model shards output from the Model Merging Reducer.
/// \author kbhall@google.com (Keith Hall)

#include <cstdio>
#include <iostream>
#include <string>
#include <memory>
#include <math.h>
#include <unistd.h>
#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "../proto/model.pb.h"
#include "../utils/kdebug.h"
#include "candidate.H"
#include "candidate-set.H"
#include "candidate-set-iterator.H"
#include "candidate-set-reader.H"
#include "factory.H"
#include "model-merge-reducer.H"
#include "model-reader.H"

#define DEFAULT_MODEL_PROTO_READER_SPEC "PerceptronModelProtoReader()"

using namespace std;
using namespace reranker;
using confusion_learning::FeatureMessage;
using confusion_learning::ModelMessage;

int main(int argc, char* argv[]) {
  int option_char;
  bool use_integer_feats = false;
  string output_name;
  string devtest_filename;
  int max_examples_to_read = -1;

  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "Io:d:n:")) != EOF) {
    switch (option_char) {  
    case 'I':
      use_integer_feats = true;
      break;
    case 'o':
      output_name = optarg;
      break;
    case 'd':
      devtest_filename = optarg;
      break;
    case 'n':
      max_examples_to_read = atoi(optarg);
      break;
    case '?':
      cerr << "usage: " << argv[0]
           << " [-E] [-I] [-d <devtest examples>] [-o <output file>]"
           << endl;
      cerr << "-E - normalize with the total number of errors" << endl;
      cerr << "-I - use integer feature id's from proto" << endl;
      return -1;
      break;
    }
  }

  ModelMessage model_with_feats;
  // Process each of the input records.  This reducer assumes that the input is
  // <FeatureId, encoded FeatureMessage> pair per line in the following format:
  // FeatureString | EncodedMsg \n
  ConfusionProtoIO reader;
  ConfusionProtoIO* writer;
  if (output_name.empty()) {
    writer = new ConfusionProtoIO("", ConfusionProtoIO::WRITESTD, false, true);
  } else {
    writer = new ConfusionProtoIO(output_name, ConfusionProtoIO::WRITE, true,
                                  true);
  }
  while (cin) {
    // Process input.
    string input_data;
    getline(cin, input_data);
    if (input_data.empty()) {
      break;
    }
    int delim_pos = input_data.find('\t');
    string feat_id = input_data.substr(0, delim_pos);
    string value = input_data.substr(delim_pos + 1);

    if (feat_id.compare(ModelInfoReducer::kModelMessageFeatureName) == 0) {
      if (model_with_feats.num_iterations() > 0) {
        cerr << "Merging in more than one model message." << endl;
        return -1;
      }
      ModelMessage new_model;
      if (!reader.DecodeBase64(value, &new_model)) {
        cerr << "Error decoding message: " << value.c_str() << endl;
      }
      // Output model message
      model_with_feats.MergeFrom(new_model);
      writer->Write(new_model);
    } else {
      FeatureMessage* feat =
          model_with_feats.mutable_raw_parameters()->add_feature();
      if (!reader.DecodeBase64(value, feat)) {
        cerr << "Error decoding message: " << value.c_str() << endl;
      }
    }
  }
  if (model_with_feats.raw_parameters().feature_size() == 0) {
    cerr << "Empty model, nothing to output." << endl;
    return -1;
  }
  // Normalize the feature values.
  for (int fix = 0; fix < model_with_feats.raw_parameters().feature_size();
       ++fix) {
    FeatureMessage* feat =
      model_with_feats.mutable_raw_parameters()->mutable_feature(fix);
    if (!isfinite(feat->value()) || !isfinite(feat->avg_value())) {
      cerr << "WARNING: feature " << feat->name() << " (ID:"
           << feat->id() << ") has non-finite value." << endl;
    } else {
      if (model_with_feats.training_errors() > 0) {
        feat->set_value(feat->value() / model_with_feats.training_errors());
        feat->set_avg_value(feat->avg_value() / model_with_feats.training_errors());
        if (!isfinite(feat->value()) || !isfinite(feat->avg_value())) {
          cerr << "WARNING: after error normalization, feature "
               << feat->name() << " (ID:" << feat->id()
               << ") has non-finite value." << endl;
        }
      }
    }
    writer->Write(*feat);
  }
  delete writer;

  double loss = 0.0;
  if (! devtest_filename.empty()) {
    // Evaluate model.
    ModelReader model_reader(1);
    shared_ptr<Model> model = model_reader.Read(model_with_feats);
    
    vector<shared_ptr<CandidateSet> > devtest_examples;
    CandidateSetReader csr(max_examples_to_read, -1, 1000);
    csr.set_verbosity(1);
    csr.Read(devtest_filename, true, true, true, devtest_examples);
    typedef CollectionCandidateSetIterator<vector<shared_ptr<CandidateSet> > >
        CandidateSetVectorIt;
    CandidateSetIterator *devtest_it =
        new CandidateSetVectorIt(devtest_examples);
    model->NewEpoch(); // sets epoch to 0
    model->Evaluate(*devtest_it);
    loss = model->loss_per_epoch().back();
    delete devtest_it;
  }
  cout << loss << endl;

  return 0;
}
