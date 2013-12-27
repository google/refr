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
/// Provides the reranker::ExampleFeatureExtractor class, which illustrates
/// how to write a reranker::FeatureExtractor implementation.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_EXAMPLE_FEATURE_EXTRACTOR_H_
#define RERANKER_EXAMPLE_FEATURE_EXTRACTOR_H_

#include <iostream>
#include <string>
#include <vector>

#include "feature-extractor.H"

namespace reranker {

using std::cout;
using std::string;
using std::vector;

/// \class ExampleFeatureExtractor
///
/// This class illustrates how to write a FeatureExtractor implementation.
/// \par Implementation advice:
/// Most implementations are likely only to have to worry about
/// implementing the \link ExtractSymbolic \endlink method,
/// implementing the \link Extract \endlink method to do nothing.
class ExampleFeatureExtractor : public FeatureExtractor {
 public:
  /// Constructs an instance, emitting that fact to <tt>cout</tt>.
  ExampleFeatureExtractor() : FeatureExtractor() {
    cout << "ExampleFeatureExtractor: creating instance!" << endl;
  }
  /// Destroys this instance.
  virtual ~ExampleFeatureExtractor() { }

  virtual void RegisterInitializers(Initializers &initializers) {
    initializers.Add("arg", &arg_);
    initializers.Add("strvec", &strvec_);
    initializers.Add("b", &b_);
  }

  /// Initializes this instance with the specified argument string.  This
  /// method is guaranteed to be invoked by a \link Factory \endlink
  /// just after construction.
  ///
  /// \param env the environment used by the \link Factory \endlink when
  ///            this object was constructed
  /// \param arg the argument string with which to initialize this instance
  ///
  /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
  virtual void Init(const Environment *env, const string &arg) {
    cout << "ExampleFeatureExtractor: set arg to \"" << arg_ << "\"" << endl;
    cout << "ExampleFeatureExtractor: set strvec to {";
    for (vector<string>::const_iterator it = strvec_.begin();
         it != strvec_.end();
         ++it) {
      cout << "\"" << *it << "\", ";
    }
    cout << "}" << endl;
    cout << "ExampleFeatureExtractor: set b to " << (b_ ? "true" : "false")
         << endl;

    cout << "Environment during construction: " << endl;
    env->Print(cout);
  }

  /// Extracts &ldquo;compiled&rdquo; features.
  /// \par Implementation advice:
  /// Most implementations of the FeatureExtractor interface will
  /// likely implement <i>either</i> this method <i>or</i> the \link
  /// ExtractSymbolic \endlink method, but not both.  In fact, most
  /// implementations will likely <b><i>only</i></b> implement the
  /// \link ExtractSymbolic \endlink method.
  ///
  /// \param[in]  candidate the candidate for which to extract features
  /// \param[out] features  the features extracted for the specified candidate
  virtual void Extract(Candidate &candidate,
                       FeatureVector<int,double> &features) {
    features.IncrementWeight(0, 5.0);
    features.IncrementWeight(27, 4.2);
  }

  /// Extracts symbolic (or <i>string</i>) features.
  ///
  /// \param[in]  candidate         the candidate for which to extract features
  /// \param[out] symbolic_features the features extracted for the specified
  ///                               candidate
  virtual void ExtractSymbolic(Candidate &candidate,
                               FeatureVector<string,double> &symbolic_features)
  {
    symbolic_features.IncrementWeight(arg_ + "foo", 3.2);
    symbolic_features.IncrementWeight(arg_ + "bar", 6734.3);
    const string &candidate_sentence = candidate.raw_data();
    size_t first_space_idx = candidate_sentence.find_first_of(" ");
    const string feature = candidate_sentence.substr(0, first_space_idx);
    symbolic_features.IncrementWeight(arg_ + feature, 34.2);
  }

 private:
  // data members
  string arg_;
  vector<string> strvec_;
  bool b_;
};

REGISTER_FEATURE_EXTRACTOR(ExampleFeatureExtractor)

}  // namespace reranker

#endif
