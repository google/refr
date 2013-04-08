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
/// Provides the reranker::FileBackedNgramFeatureExtractor class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_FILE_BACKED_NGRAM_FEATURE_EXTRACTOR_H_
#define RERANKER_FILE_BACKED_NGRAM_FEATURE_EXTRACTOR_H_

#include "basic-file-backed-feature-extractor.H"
#include "ngram-feature-extractor.H"

namespace reranker {

/// \class FileBackedNgramFeatureExtractor
///
/// A class to read one line at a time from  backing file, and tokenize
/// that line based on whitespace, and then extract n-gram features
/// from that token stream.

class FileBackedNgramFeatureExtractor :
      public BasicFileBackedFeatureExtractor {
 public:
  /// Constructs an instance.
  FileBackedNgramFeatureExtractor() : BasicFileBackedFeatureExtractor() {
  }
  /// Destroys this instance.
  ~FileBackedNgramFeatureExtractor() { }


  /// Registers three variables that may be initialized when this object
  /// is constructed via \link Factory::CreateOrDie\endlink.
  /// <table>
  /// <tr>
  ///   <th>Variable name</th>
  ///   <th>Type</th>
  ///   <th>Required</th>
  ///   <th>Description</th>
  ///   <th>Default value</th>
  /// </tr>
  /// <tr>
  ///   <td><tt>filename</tt></td>
  ///   <td><tt>string</tt></td>
  ///   <td>Yes</td>
  ///   <td>The name of the file backing this feature extractor.</td>
  ///   <td>n/a</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>n</tt></td>
  ///   <td><tt>int</tt></td>
  ///   <td>Yes</td>
  ///   <td>The n-gram order for this feature extractor to extract.</td>
  ///   <td>n/a</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>prefix</tt></td>
  ///   <td><tt>string</tt></td>
  ///   <td>No</td>
  ///   <td>A prefix string to be prepended to each feature name produced.</td>
  ///   <td><i><b>n</b></i> + <tt>&quot;g_ng&quot;</tt> where
  ///       <i><b>n</b></i> is the string representation of the n-gram
  ///       order</td>
  /// </tr>
  /// </table>
  virtual void RegisterInitializers(Initializers &initializers) {
    bool required = true;
    initializers.Add("filename", &filename_, required);
    initializers.Add("n", &n_, required);
    initializers.Add("prefix", &prefix_);
  }

  /// Does nothing.
  virtual void Extract(Candidate &candidate,
                       FeatureVector<int,double> &features) { }

  /// Extracts symbolic features based on the most recent line read from
  /// the file (or stream) that backs this feature extractor.  The line
  /// is tokenized based on whitespace, and each token is transformed into
  /// a feature-value pair via the \link ExtractFeatureValuePair \endlink
  /// method.
  virtual void ExtractSymbolic(Candidate &candidate,
                               FeatureVector<string,double> &symbolic_features);
 protected:
  // data members
  int n_;
  string prefix_;
  NgramExtractor ngram_extractor_;
};

}  // namespace reranker

#endif
