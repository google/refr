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
/// \file perceptron-model-test.C
/// Test driver for training a reranker::PerceptronModel.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <tr1/memory>

#include "candidate.H"
#include "candidate-set.H"
#include "candidate-set-iterator.H"
#include "candidate-set-reader.H"
#include "model.H"
#include "perceptron-model.H"
#include "model-proto-writer.H"
#include "perceptron-model-proto-writer.H"
#include "../proto/data.pb.h"
#include "../proto/dataio.h"

#define DEBUG 1
#define REPORTING_INTERVAL 100

#define MAX_NUM_EXAMPLES 1000
#define MAX_NUM_CANDIDATES 1000000


using namespace reranker;
using namespace std;
using namespace std::tr1;

int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "usage: <training data>+ <devtest data> <model output file>"
         << endl;
    return -1;
  }
  vector<string> training_files;
  int i = 1;
  for ( ; i < argc - 2; ++i) {
    training_files.push_back(argv[i]);
  }
  const string devtest_file = argv[i++];
  const string model_file = argv[i++];

  CandidateSetReader csr(MAX_NUM_EXAMPLES, MAX_NUM_CANDIDATES,
                         REPORTING_INTERVAL);
  csr.set_verbosity(DEBUG);

  vector<shared_ptr<CandidateSet> > training_examples;
  bool compressed = true;
  bool use_base64 = true;
  bool reset_counters = true;
  for (vector<string>::const_iterator it = training_files.begin();
       it != training_files.end();
       ++it) {
    csr.Read(*it, compressed, use_base64, reset_counters, training_examples);
  }
  if (DEBUG) {
    cout << "Read " << training_examples.size() << " training examples."
         << endl;
  }

  vector<shared_ptr<CandidateSet> > devtest_examples;
  csr.Read(devtest_file, compressed, use_base64, reset_counters,
           devtest_examples);
  if (DEBUG) {
    cout << "Read " << devtest_examples.size() << " devtest examples."
         << endl;
  }

  shared_ptr<Model> model(new PerceptronModel("My Test Model"));
  typedef CollectionCandidateSetIterator<vector<shared_ptr<CandidateSet> > >
      CandidateSetVectorIt;
  CandidateSetVectorIt training_examples_it(training_examples);
  CandidateSetVectorIt devtest_examples_it(devtest_examples);
  model->Train(training_examples_it, devtest_examples_it);

  model->CompactifyFeatureUids();

  // TODO(dbikel): Need some kind of a factory for model writers, so the
  //               proper ModelProtoWriter gets instantiated given a
  //               particular Model subclass.
  shared_ptr<ModelProtoWriter> model_writer(new PerceptronModelProtoWriter());

  confusion_learning::ModelMessage model_message;
  model_writer->Write(model.get(), &model_message);

  // Write out serialized model.
  shared_ptr<ConfusionProtoIO> proto_writer(
      new ConfusionProtoIO(model_file, ConfusionProtoIO::WRITE,
                           compressed, use_base64));
  proto_writer->Write(model_message);

  cout << "Have a nice day!" << endl;
}
