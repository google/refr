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
/// Provides the reranker::FeatureExtractor interface.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_FEATURE_EXTRACTOR_H_
#define RERANKER_FEATURE_EXTRACTOR_H_

#include <string>
#include <tr1/memory>
#include <tr1/unordered_map>

#include "feature-vector.H"
#include "candidate.H"
#include "candidate-set.H"
#include "factory.H"

namespace reranker {

using std::string;
using std::tr1::shared_ptr;
using std::tr1::unordered_map;

/// A free-floating function (within the \link reranker \endlink namespace)
/// that frees statically allocated objects.
void TearDown();

/// \class FeatureExtractor
///
/// An abstract base class/interface for all feature extractors.
/// Concrete subclasses need to register themselves with the \link
/// FeatureExtractor \endlink \link Factory \endlink via the \link
/// REGISTER_FEATURE_EXTRACTOR(MyDerivedClassName) \endlink macro.
///
/// Example:
/// \code
/// my-feature-extractor.H:
///
/// class MyFeatureExtractor : public FeatureExtractor {
///  public:
///   ...
/// };
///
/// my-feature-extractor.C:
///
/// REGISTER_FEATURE_EXTRACTOR(MyFeatureExtractor)
/// \endcode
///
/// \see ExampleFeatureExtractor for a fully functional, if not quite useful,
///      example FeatureExtractor implementation
/// \see feature-extractor-test.C for a working example employing a
///      FeatureExtractor
/// \see ExecutiveFeatureExtractor for a class that can &ldquo;execute&rdquo;
///      a suite of \link FeatureExtractor \endlink instances on a
///      \link CandidateSet \endlink
class FeatureExtractor : public FactoryConstructible {
 public:
  /// Constructs an empty feature vector
  FeatureExtractor() {}
  /// Destroys this vector.
  virtual ~FeatureExtractor() {}

  /// Initializes this feature extractor.  This method is guaranteed
  /// to be invoked by a \link Factory \endlink after construction.
  ///
  /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
  virtual void Init(const string &arg) = 0;

  /// Extracts features for the specified candidate, where each
  /// feature is represented by an int for its unique identifier.
  /// \par Implementation advice:
  /// Please play nice with others!  For efficiency, the feature
  /// vector you are modifying is the actual feature vector inside the
  /// specified Candidate.  Please use FeatureVector::IncrementWeight
  /// wherever possible; only use FeatureVector::SetWeight if you
  /// truly want to blow away any existing weight for a feature.
  ///
  /// \param[in]  candidate the candidate for which to extract features
  /// \param[out] features  the features extracted for the specified candidate
  virtual void Extract(Candidate &candidate,
                       FeatureVector<int,double> &features) = 0;

  /// Extracts symbolic features for the specified candidate, where
  /// each feature is represented by a string for its unique identifier.
  /// \par Implementation advice:
  /// Please play nice with others!  For efficiency, the feature
  /// vector you are modifying is the actual feature vector inside the
  /// specified Candidate.  Please use FeatureVector::IncrementWeight
  /// wherever possible; only use FeatureVector::SetWeight if you
  /// truly want to blow away any existing weight for a feature.
  ///
  /// \param[in]  candidate         the candidate for which to extract features
  /// \param[out] symbolic_features the features extracted for the specified
  ///                               candidate
  virtual void ExtractSymbolic(Candidate &candidate,
                               FeatureVector<string,double> &symbolic_features)
  = 0;

  /// Indicates to this instance that iteration over candidate sets on
  /// which features are being extracted has been reset.  Concrete
  /// subclasses should override this method (which by default does
  /// nothing) if they need to do something special at the beginning
  /// of each iteration.
  virtual void Reset() {
  }

  /// Extracts features for all the candidates in the specified CandidateSet.
  /// The default implementation here uses the concrete implementations
  /// of the
  /// \link Extract(Candidate&,FeatureVector<int,double>&) \endlink
  /// and
  /// \link ExtractSymbolic \endlink methods.
  ///
  /// \param candidate_set the set of candidates for which to extract features.
  virtual void Extract(CandidateSet &candidate_set) {
    for (CandidateSet::iterator it = candidate_set.begin();
         it != candidate_set.end();
         ++it) {
      Candidate &candidate = *(*it);
      Extract(candidate, candidate.mutable_features());
      ExtractSymbolic(candidate, candidate.mutable_symbolic_features());
    }
  }
};

/// Registers the \link reranker::FeatureExtractor FeatureExtractor
/// \endlink with the specified subtype <tt>TYPE</tt> and
/// <tt>NAME</tt> with the \link reranker::FeatureExtractor
/// FeatureExtractor \endlink \link reranker::Factory Factory\endlink.
#define REGISTER_NAMED_FEATURE_EXTRACTOR(TYPE,NAME) \
  REGISTER_NAMED(TYPE,NAME,FeatureExtractor)

/// Registers the \link reranker::FeatureExtractor FeatureExtractor
/// \endlink with the specified subtype <tt>TYPE</tt> with the \link
/// reranker::FeatureExtractor FeatureExtractor \endlink \link
/// reranker::Factory Factory\endlink.
#define REGISTER_FEATURE_EXTRACTOR(TYPE) \
  REGISTER_NAMED_FEATURE_EXTRACTOR(TYPE,TYPE)

}  // namespace reranker

#endif
