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
/// Provides the reranker::StringCanonicalizer class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_STRING_CANONICALIZER_H_
#define RERANKER_STRING_CANONICALIZER_H_

#include <iostream>
#include <string>
#include <tr1/unordered_set>

#define STR_CANON_DEBUG 1

namespace reranker {

using std::cerr;
using std::endl;
using std::string;
using std::tr1::unordered_set;

/// \class StringCanonicalizer
///
/// A class that stores a canonical version of <tt>string</tt> objects
/// in a static data structure.
class StringCanonicalizer {
 public:
  /// Returns the canonical string for the specified string.
  static const string &Get(const string &s) {
    unordered_set<string>::const_iterator it = canonical_.find(s);
    if (it == canonical_.end()) {
      canonical_.insert(s);
      return s;
    } else {
      return *it;
    }
  }

  /// Clears the strings from the internal data structure.
  static void Clear() {
    if (STR_CANON_DEBUG) {
      cerr << "StringCanonicalizer: clearing out canonical map of size "
           << canonical_.size() << endl;
    }
    canonical_.clear();
  }
 private:
  // data members
  static unordered_set<string> canonical_;
};

}  // namespace reranker

#endif
