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
/// Provides the abstract reranker::AbstractFileBackedFeatureExtractor class,
/// which makes it easy for concrete subclasses to extract features
/// based on lines of a file.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_ABSTRACT_FILE_BACKED_FEATURE_EXTRACTOR_H_
#define RERANKER_ABSTRACT_FILE_BACKED_FEATURE_EXTRACTOR_H_

#include <iostream>
#include <fstream>

#include "tokenizer.H"
#include "feature-extractor.H"

namespace reranker {

using std::cout;
using std::istream;
using std::ifstream;

/// \class AbstractFileBackedFeatureExtractor
///
/// This class makes it easy for concrete subclasses to extract features
/// based on input from a file.
class AbstractFileBackedFeatureExtractor : public FeatureExtractor {
 public:
  /// Constructs an instance.
  AbstractFileBackedFeatureExtractor() : FeatureExtractor(), is_(NULL) {
  }
  /// Destroys this instance.
  virtual ~AbstractFileBackedFeatureExtractor() {
    delete is_;
  }

  /// Registers the data member to be initialized when an instance of
  /// this class is constructed by a \link Factory\endlink.
  ///
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
  /// </table>
  virtual void RegisterInitializers(Initializers &initializers) {
    bool required = true;
    initializers.Add("filename", &filename_, required);
  }

  /// This method is guaranteed to be inokved by a \link Factory
  /// \endlink after invoking the \link RegisterInitializers \endlink
  /// method just after construction
  ///
  /// \param arg the string parsed by the \link Factory \endlink
  ///
  /// \see Factory::CreateOrDie
  virtual void Init(const string &arg) {
    delete is_;
    is_ = new ifstream(filename_.c_str());    
  }

  /// \copydoc FeatureExtractor::Extract
  virtual void Extract(Candidate &candidate,
                       FeatureVector<int,double> &features) = 0;

  /// \copydoc FeatureExtractor::ExtractSymbolic
  virtual void ExtractSymbolic(Candidate &candidate,
                               FeatureVector<string,double> &symbolic_features)
  = 0;

  virtual void Reset() {
    delete is_;
    is_ = new ifstream(filename_.c_str());    
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
      ReadFromStream();
      Extract(candidate, candidate.mutable_features());
      ExtractSymbolic(candidate, candidate.mutable_symbolic_features());
    }
  }

  /// Returns the current number of lines read by this feature extractor
  /// from the underlying stream.
  virtual long line_number() const { return line_number_; }

 protected:
  /// Reads from the stream.  The default implementation here reads a single
  /// line and puts it into the protected <tt>line_</tt> member.
  virtual void ReadFromStream() {
    // TODO(dbikel): Do error checking here for is_->good().
    //               Emit massive warnings when something goes wrong.
    // TODO(dbikel): Keep track of number of lines read.
    if (is_->good()) {
      getline(*is_, line_);
      ++line_number_;
    } else {
      cerr << "AbstractFileBackedFeatureExtractor: error: no more input\n"
           << "\t(number of lines read so far: " << line_number_ << ")" << endl;
    }
  }

  // data members
  /// The name of the file backing this feature extractor.
  string filename_;
  /// The stream created from the file backing this feature extractor.
  istream *is_;
  /// The last line read by this feature extractor.
  string line_;
  /// The number of lines read so far by this feature extractor.
  long line_number_;
  /// A simple whitespace tokenizer for use by concrete subclasses.
  Tokenizer tokenizer_;
};

}  // namespace reranker

#endif
