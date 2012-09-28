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
/// \file
/// Provides the \link reranker::Tokenizer Tokenizer \endlink class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_TOKENIZER_H_
#define RERANKER_TOKENIZER_H_

#include <string>
#include <vector>

namespace reranker {

#define SPACE_CHARS " \t"

using std::string;
using std::vector;

/// A very simple tokenizer class.
class Tokenizer {
 public:
  /// Tokenizes the specified string, depositing the results into the specified
  /// vector. The default delimiters are space and tab characters (so that
  /// this method performs whitespace tokenization).
  ///
  /// \param[in]  s         the string to tokenize
  /// \param[out] toks the vector to which to append the results of
  ///                       tokenizing the specified string; this
  ///                       vector <b><i>is not cleared</i></b> by
  ///                       this method
  /// \param[in] delimiters the set of delimiter characters to use when
  ///                       tokenizing (defaults to space and tab characters)
  void Tokenize(const string &s, vector<string> &toks,
                const char *delimiters = " \t") const {
    size_t end_pos = 0;
    size_t begin_pos = 0;
    while (begin_pos != string::npos) {
      begin_pos = s.find_first_not_of(delimiters, end_pos);
      end_pos = s.find_first_of(delimiters, begin_pos);
      if (end_pos == string::npos) {
        end_pos = s.length();
      }
      if (begin_pos != string::npos) {
        toks.push_back(s.substr(begin_pos, end_pos - begin_pos));
        begin_pos = end_pos;
      }
    }
  }

  /// Parses a specification string of the form "ClassName(init_string)",
  /// depositing the results into the specified string reference parameters.
  /// This method will return whether <tt>spec</tt> was successfully parsed.
  /// Note that this method can return false but set <tt>error</tt> to
  /// <tt>false</tt> if <tt>spec</tt> is, <i>e.g.</i>, all whitespace
  /// (which cannot be parsed, but is not an error).
  ///
  /// \param[in]  spec        the specifcation to be parsed
  /// \param[out] class_name  the class name in the specification string
  /// \param[out] init_string the initialization string obtained from
  ///                         the specification string <tt>spec</tt>
  /// \param[out] error       whether there was an error parsing the
  ///                         specification string <tt>spec</tt>
  /// \return if the specification string <tt>spec</tt> was successfully
  ///         parsed by this method with the results placed into
  ///         <tt>class_name</tt> and <tt>init_string</tt>
  ///
  /// \see Factory::Create(const string&,string&,string&,bool&,bool&)
  bool ParseSpecString(const string &spec,
                       string &class_name,
                       string &init_string,
                       bool &error) {
    error = false;
    size_t first_non_ws_idx = spec.find_first_not_of(SPACE_CHARS);
    if (first_non_ws_idx == string::npos) {
      // line is entirely ws
      return false;
    }

    size_t first_paren_idx = spec.find_first_of("(", first_non_ws_idx + 1);
    size_t last_paren_idx = spec.find_last_of(")");

    if (first_paren_idx == string::npos ||
        last_paren_idx == string::npos ||
        last_paren_idx < first_paren_idx) {
      error = true;
      return false;
    }
    class_name = spec.substr(first_non_ws_idx,
                                    first_paren_idx - first_non_ws_idx);
    size_t init_string_start_idx = first_paren_idx + 1;
    size_t init_string_len = last_paren_idx - init_string_start_idx;
    init_string = spec.substr(init_string_start_idx, init_string_len);    
    return true;
  }
};

}  // namespace reranker

#endif

