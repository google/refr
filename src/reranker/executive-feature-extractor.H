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
/// \file executive-feature-extractor.H
/// Provides the reranker::ExecutiveFeatureExtractor class.  This class
/// executes the extraction methods of a suite of reranker::FeatureExtractor
/// instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_EXECUTIVE_FEATURE_EXTRACTOR_H_
#define RERANKER_EXECUTIVE_FEATURE_EXTRACTOR_H_

#include <iostream>
#include <string>
#include <tr1/memory>
#include <vector>

#include "candidate.H"
#include "candidate-set.H"
#include "feature-extractor.H"
#include "tokenizer.H"

namespace reranker {

using std::istream;
using std::string;
using std::tr1::shared_ptr;
using std::vector;

/// \class ExecutiveFeatureExtractor
///
/// This class is like a regular FeatureExtractor, but has been promoted
/// to the executive level and thus wears fancypants.  More specifically,
/// this class executes the extraction of a suite of feature extractors,
/// specified at run-time via a set of string specifications, which are
/// typically read from a file.
///
/// \see FeatureExtractorFactory
class ExecutiveFeatureExtractor {
 public:
  /// Constructs a new intance.
  ExecutiveFeatureExtractor() { }
  /// Destroys this instance.
  virtual ~ExecutiveFeatureExtractor() { }

  /// Initializes this instance from the feature extractor
  /// specifications in the specified file.  A feature extractor specification
  /// is a line that is of the form<br>
  /// \code FeatureExtractorClassName(init_arg) \endcode
  /// <p>
  /// More formally, each line must have the form
  /// \code
  /// <feature extractor classname> <lparen> <init str> <rparen>
  /// \endcode
  /// where
  /// <table border=0>
  /// <tr>
  ///   <td><tt>\<feature extractor classname\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>the string passed to the \link REGISTER_NAMED \endlink macro
  ///       by a \link FeatureExtractor \endlink implementation</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<lparen\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>a left parenthesis character, <tt>'('</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<init str\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>an initialization string to be passed to the newly-constructed
  ///       \link FeatureExtractor \endlink&rsquo;s
  ///       \link FeatureExtractor::Init \endlink method
  ///   </td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<rparen\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>a right parenthesis character, <tt>')'</tt></td>
  /// </tr>
  /// </table>
  ///
  /// \param filename the name of a file containing a set of feature extractor
  ///                 specifications, one per line
  ///
  /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
  void Init(const string &filename);

  /// Initializes this instance from the feature extractor
  /// specifications in the specified input stream.  A feature extractor
  /// specification is a line that is of the form<br>
  /// \code FeatureExtractorClassName(init_arg) \endcode
  /// Please see the documentation for the \link Init(const string&) \endlink
  /// method for the formal specification.
  ///
  /// \param is an input stream containing a set of feature extractor
  ///           specifications, one per line
  void Init(istream &is);

  /// Indictes to this instance that iteration has been reset.  This
  /// is especially important for file-backed feature extractors.
  ///
  /// \see FeatureExtractor::Reset
  void Reset() const;

  /// Extracts features for the specified CandidateSet using the suite
  /// of FeatureExtractor instances specified at construction time.
  ///
  /// \param candidate_set the CandidateSet for which to extract features
  void Extract(CandidateSet &candidate_set) const;
 private:
  // data members
  /// The feature extractor factory so this executive feature extractor
  /// can construct individual FeatureExtractor instances from their
  /// specification strings.
  Factory<FeatureExtractor> factory_;
  /// The set of feature extractors used by this executive feature extractor.
  vector<shared_ptr<FeatureExtractor> > extractors_;
};

}  // namespace reranker

#endif
