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
/// Provides the \link reranker::StreamTokenizer StreamTokenizer \endlink class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_STREAM_TOKENIZER_H_
#define RERANKER_STREAM_TOKENIZER_H_

#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>

namespace reranker {

using std::istream;
using std::istringstream;
using std::ostringstream;
using std::streampos;
using std::string;
using std::vector;
using std::cerr;
using std::endl;

/// \class StreamTokenizer
///
/// A simple class for tokenizing a stream of tokens for the informally
/// specified, ad-hoc language used to construct objects for the Reranker
/// framework.
class StreamTokenizer {
 public:
  enum TokenType {
    EOF_TYPE,
    RESERVED,
    STRING,
    NUMBER,
    IDENTIFIER
  };
  static const char *TypeName(TokenType token_type) {
    static const char *names[] = {
      "EOF", "RESERVED", "STRING", "NUMBER", "IDENTIFIER"
    };
    return names[token_type];
  }

  /// Constructs a new instance around the specified byte stream.
  ///
  /// \param is       the input byte stream for this stream tokenizer to use
  /// \param reserved the set of single characters serving as
  ///                 &ldquo;reserved words&rdquo;
  StreamTokenizer(istream &is, const char *reserved = "(),") :
      is_(is), curr_pos_(0), num_read_(0), has_next_(false) {
    Init(reserved);
  }
  StreamTokenizer(const string &s, const char *reserved = "(),") :
      sstream_(s), is_(sstream_), curr_pos_(0), num_read_(0), has_next_(false) {
    Init(reserved);
  }
  
  /// Destroys this instance.
  virtual ~StreamTokenizer() {
    delete[] reserved_;
  }

  string str() { return oss_.str(); }

  /// Returns the number of bytes read from the underlying byte
  /// stream.
  size_t tellg() { return curr_pos_; }
  /// Returns whether there is another token in the token stream.
  bool HasNext() { return has_next_; }
  /// Returns the next token in the token stream.
  string Next() {
    string curr_tok = next_tok_;
    curr_pos_ = num_read_;
    GetNext();
    return curr_tok;
  }
  size_t PeekTokenStart() {
    return has_next_ ? next_tok_start_ : tellg();
  }
  TokenType PeekTokenType() { return has_next_ ? next_tok_type_ : EOF_TYPE; }
  /// Returns the next token that would be returned by the \link Next
  /// \endlink method.  The return value of this method is only valid
  /// when \link HasNext \endlink returns <tt>true</tt>.
  string Peek() { return has_next_ ? next_tok_ : ""; }
 private:
  void Init(const char *reserved) {
    num_reserved_ = strlen(reserved);
    reserved_ = new char[num_reserved_];
    strcpy(reserved_, reserved);
    GetNext();
  }

  /// Retrieves the next token from the <tt>istream</tt> wrapped by this
  /// stream tokenizer.
  void GetNext();

  /// Returns whether the specified character represents a
  /// &ldquo;reserved word&rdquo;.
  bool Reserved(char c) {
    for (size_t i = 0; i < num_reserved_; ++i) {
      if (c == reserved_[i]) {
        return true;
      }
    }
    return false;
  }
  
  // data members
  istringstream sstream_;
  istream &is_;
  char *reserved_;
  size_t num_reserved_;
  string next_tok_;
  TokenType next_tok_type_;
  size_t next_tok_start_;
  size_t curr_pos_;
  size_t num_read_;
  ostringstream oss_;
  bool has_next_;
};

}  // namespace reranker

#endif
