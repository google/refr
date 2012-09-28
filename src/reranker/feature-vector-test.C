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
/// \file feature-vector-test.C
/// Test for the reranker::FeatureVector class.
/// \author dbikel@google.com (Dan Bikel)

#include <map>

#include "feature-vector.H"

using reranker::FeatureVector;
using std::map;
using std::cout;
using std::endl;

int
main(int argc, char **argv) {
  map<int, double> features1;
  features1[0] = 2.0;
  features1[1] = 4.0;
  FeatureVector<int,double> feature_vector1(features1);

  map<int, double> features2;
  features2[0] = 4.0;
  features2[1] = 8.0;
  features2[2] = 45.0;
  FeatureVector<int,double> feature_vector2(features2);

  cout << "Feature vector 1:" << feature_vector1 << endl
       << "Feature vector 2:" << feature_vector2 << endl
       << "Dot product: " << feature_vector1.Dot(feature_vector2) << endl;

  cout << "Weight in feature vector 1 for feature 2: "
       << feature_vector1.GetWeight(2) << endl;
  cout << "Weight in feature vector 2 for feature 2: "
       << feature_vector2.GetWeight(2) << endl;

  cout << "Feature vector 1 size: " << feature_vector1.size() << endl
       << "Setting feature 0 to 0..." << endl;
  feature_vector1.SetWeight(0, 0.0);
  cout << "New feature vector 1 size: " << feature_vector1.size() << endl;

  cout << "Printing out feature vector 1 using its iterators:" << endl;
  for (FeatureVector<int,double>::const_iterator it = feature_vector1.begin();
       it != feature_vector1.end();
       ++it) {
    cout << "\t" << it->first << "=" << it->second << "\n";
  }
  cout << "Done." << endl;

  cout << "Printing out feature vector 2 using its iterators:" << endl;
  for (FeatureVector<int,double>::const_iterator it = feature_vector2.begin();
       it != feature_vector2.end();
       ++it) {
    cout << "\t" << it->first << "=" << it->second << "\n";
  }
  cout << "Done." << endl;

  cout << "Scaling feature vector 2 by 2.0:" << endl;
  feature_vector2.Scale(2.0);
  cout << "\t" << feature_vector2 << endl;

  cout << "Adding feature vector 2 scaled by 2.0 to feature vector 1:" << endl;
  feature_vector1.AddScaledVector(feature_vector2, 2.0);
  cout << "\t" << feature_vector1 << endl;

  cout << "Have a nice day!" << endl;
}
