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
/// \file feature-extractor-test.C
/// Test driver for the reranker::FeatureExtractor and
/// reranker::FeatureExtractorFactory classes.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <memory>

#include "candidate.H"
#include "candidate-set.H"
#include "feature-extractor.H"
#include "example-feature-extractor.H"

using namespace std;
using namespace reranker;

int
main(int argc, char **argv) {
  Factory<FeatureExtractor> factory;
  shared_ptr<FeatureExtractor> example_feature_extractor1 =
      factory.CreateOrDie("ExampleFeatureExtractor(arg(\"my_feats:\"))", "");

  shared_ptr<FeatureExtractor> example_feature_extractor2 =
      factory.CreateOrDie("ExampleFeatureExtractor(b(true), "
                                                  "arg(\"your_feats:\"))", "");

  shared_ptr<FeatureExtractor> example_feature_extractor3 =
      factory.CreateOrDie("ExampleFeatureExtractor(arg(\"whose_feats:\"), "
                          "strvec({\"foo\", \"bar\", \"baz\"}))", "");

  shared_ptr<FeatureExtractor> example_feature_extractor4 =
      factory.CreateOrDie("ExampleFeatureExtractor(b(true))", "");

  CandidateSet candidate_set("test candidate set");
  candidate_set.set_reference_string("This is a reference string.");
  shared_ptr<Candidate> c1(new Candidate(0, 0.1, 0.7, 5,
                                         "This is a silly string."));
  candidate_set.AddCandidate(c1);
  shared_ptr<Candidate> c2(new Candidate(1, 0.2, 0.8, 5,
                                         "This is a sillier string."));
  candidate_set.AddCandidate(c2);

  example_feature_extractor1->Extract(candidate_set);
  example_feature_extractor2->Extract(candidate_set);
  example_feature_extractor3->Extract(candidate_set);

  cout << candidate_set << endl;
}
