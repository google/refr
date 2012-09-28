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
/// \file file-backed-loss-setter.H
/// Provides the reranker::FileBackedLossSetter class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_FILE_BACKED_LOSS_SETTER_H_
#define RERANKER_FILE_BACKED_LOSS_SETTER_H_

#include <cstdlib>
#include <vector>

#include "abstract-file-backed-feature-extractor.H"

namespace reranker {

using std::vector;

/// \class FileBackedLossSetter
///
/// A &ldquo;feature extractor&rdquo; that reads lines from a backing file,
/// setting each candidate&rsquo;s loss via its \link Candidate::set_loss
/// \endlink method.
class FileBackedLossSetter : public AbstractFileBackedFeatureExtractor {
 public:
  /// Constructs a new instance.
  FileBackedLossSetter() : AbstractFileBackedFeatureExtractor() {
  }
  /// Destroys this instance.
  virtual ~FileBackedLossSetter() { }

  virtual void RegisterInitializers(Initializers &initializers) {
    token_idx_ = 0;
    initializers.Add("filename", &filename_);
    initializers.Add("token_idx", (int *)&token_idx_);
  }

  /// Initializes this instance with the specified argument string,
  /// which is expected to be just a filename, or else a filename
  /// followed by a tab character followed by an integer specifying
  /// which token index of each line contains the loss value.
  /// If the initialization string consists solely of a filename (no
  /// tab character), then each line is assumed to contain nothing
  /// but a floating-point loss value.
  ///
  /// This method is guaranteed to be invoked by a \link Factory \endlink
  /// just after construction.
  ///
  /// \param arg either a just the name of the file that backs
  ///            this feature extractor, or else that filename
  ///            followed by a tab character followed by an
  ///            integer specifying which token index of each
  ///            line contains the loss
  ///
  /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
  virtual void Init(const string &arg) {
    filename_ = arg;
    token_idx_ = 0;
    size_t tab_pos = arg.find_first_of("\t");
    if (tab_pos != string::npos) {
      filename_ = arg.substr(tab_pos);
      if (tab_pos < arg.length()) {
        token_idx_ = atoi(arg.substr(tab_pos + 1).c_str());
      }
    }
    is_ = new ifstream(filename_.c_str());
  }

  /// Sets the loss for the specified candidate based on what was read from
  /// the last line in the stream backing this FeatureExtractor instance.
  virtual void Extract(Candidate &candidate,
                       FeatureVector<int,double> &features);

  /// Overridden to do nothing.
  virtual void ExtractSymbolic(Candidate &candidate,
                               FeatureVector<string,double> &symbolic_features)
  {
  }
 private:
  // data members
  /// The index of the token containing the loss in each line read by this
  /// file-backed &ldquo;feature extractor&rdquo;.
  size_t token_idx_;
  /// The tokens obtained from the last line.
  vector<string> tokens_;
};

}  // namespace reranker

#endif
