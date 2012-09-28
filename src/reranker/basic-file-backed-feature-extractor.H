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
/// Provides the reranker::BasedFileBackedFeatureExtractor class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_BASIC_FILE_BACKED_FEATURE_EXTRACTOR_H_
#define RERANKER_BASIC_FILE_BACKED_FEATURE_EXTRACTOR_H_

#include <cstdlib>
#include <string>
#include <vector>

#include "abstract-file-backed-feature-extractor.H"

namespace reranker {

using std::string;
using std::vector;

/// \class BasicFileBackedFeatureExtractor
///
/// A class to read one line at a time from a backing file, and tokenize
/// that line based on whitespace, and then interpret each token as a
/// feature-value pair via a protected method \link
/// ExtractFeatureValuePair \endlink (see the documentation for that method
/// for details on how tokens are interpreted as feature-value pairs).
class BasicFileBackedFeatureExtractor :
      public AbstractFileBackedFeatureExtractor {
 public:
  /// Constructs an instance.
  BasicFileBackedFeatureExtractor() : AbstractFileBackedFeatureExtractor() {
  }
  /// Destroys this instance.
  virtual ~BasicFileBackedFeatureExtractor() { }

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
  /// Extracts a symbolic feature and value from the specified string.  The
  /// specified string may have one of three forms:
  /// <table border=0>
  /// <tr><th>String</th><th>Feature name</th><th>Feature weight</th></tr>
  /// <tr><td><tt>foo</tt></td><td><tt>foo</tt></td><td><tt>1.0</tt></td></tr>
  /// <tr><td><tt>foo=</tt></td><td><tt>foo</tt></td><td><tt>1.0</tt></td></tr>
  /// <tr><td><tt>foo=value</tt></td>
  ///     <td><tt>foo</tt></td>
  ///     <td><tt>value</tt></td>
  /// </tr>
  /// </table>
  /// where <tt>value</tt> is the (string representation of) a
  /// floating-point value.
  ///
  /// \param[in]  s       the string from which to extract a feature-value pair
  /// \param[out] feature the symbolic feature name extracted from the
  ///                     specified string <tt>s</tt>
  /// \param[out] value   the symbolic feature&rsquo;s value extracted from
  ///                     the specified string <tt>s</tt>
  virtual void ExtractFeatureValuePair(const string &s,
                                       string &feature, double &value) {
    size_t equals_pos = s.find_last_of("=");
    bool found_equals = equals_pos != string::npos;
    value =
        found_equals && equals_pos < s.length() - 1 ?
        atof(s.substr(equals_pos + 1, s.length() - (equals_pos + 1)).c_str()) :
        1.0;
    feature = found_equals ? s.substr(0, equals_pos) : s;
  }
};

}  // namespace reranker

#endif
