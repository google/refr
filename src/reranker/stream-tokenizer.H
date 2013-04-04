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
#include <set>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>

namespace reranker {

using std::istream;
using std::istringstream;
using std::ostringstream;
using std::set;
using std::streampos;
using std::string;
using std::vector;
using std::cerr;
using std::endl;

/// Default set of reserved words for the StreamTokenizer class.
/// Use the \link reranker::StreamTokenizer::set_reserved_words
/// StreamTokenizer::set_reserved_words \endlink
/// to customize this set.
static const char *default_reserved_words[] = {
  "false",
  "true",
  "bool",
  "int",
  "double",
  "string",
  "bool[]",
  "int[]",
  "double[]",
  "string[]",
};

/// Default set of reserved characters for the StreamTokenizer class.
static const char *default_reserved_chars = "(){},=;";

/// \class StreamTokenizer
///
/// A simple class for tokenizing a stream of tokens for the formally
/// specified language used to construct objects for the Reranker
/// framework.
class StreamTokenizer {
 public:
  /// The set of types of tokens read by this stream tokenizer.
  ///
  /// \see PeekTokenType
  enum TokenType {
    EOF_TYPE,
    RESERVED_CHAR,
    RESERVED_WORD,
    STRING,
    NUMBER,
    IDENTIFIER
  };

  /// Returns a string type name for the specified TokenType constant.
  static const char *TypeName(TokenType token_type) {
    static const char *names[] = {
      "EOF", "RESERVED_CHAR", "RESERVED_WORD", "STRING", "NUMBER", "IDENTIFIER"
    };
    return names[token_type];
  }

  /// Constructs a new instance around the specified byte stream.
  ///
  /// \param is             the input byte stream for this stream tokenizer to
  ///                       use
  /// \param reserved_chars the set of single characters serving as
  ///                       &ldquo;reserved characters&rdquo;
  StreamTokenizer(istream &is,
                  const char *reserved_chars = default_reserved_chars) :
      is_(is), curr_pos_(0), num_read_(0), has_next_(false), has_prev_(false) {
    Init(reserved_chars);
  }

  /// Constructs a new instance around the specified string.
  ///
  /// \param s              the string providing the stream of characters
  ///                       for this stream tokenizer to use
  /// \param reserved_chars the set of single characters serving as
  ///                       &ldquo;reserved characters&rdquo;
  StreamTokenizer(const string &s,
                  const char *reserved_chars = default_reserved_chars) :
      sstream_(s), is_(sstream_), curr_pos_(0), num_read_(0), has_next_(false),
      has_prev_(false) {
    Init(reserved_chars);
  }

  /// Sets the set of &ldquo;reserved words&rdquo; used by this stream
  /// tokenizer.  Should be invoked just after construction time.
  void set_reserved_words(set<string> &reserved_words) {
    reserved_words_ = reserved_words;
  }

  /// Destroys this instance.
  virtual ~StreamTokenizer() {
    delete[] reserved_chars_;
  }

  /// Returns the entire sequence of characters read so far by this
  /// stream tokenizer as a newly constructed string object.
  string str() { return oss_.str(); }

  /// Returns the number of bytes read from the underlying byte
  /// stream.
  size_t tellg() const { return curr_pos_; }

  /// Returns whether there is another token in the token stream.
  bool HasNext() const { return has_next_; }

  bool HasPrev() const { return has_prev_; }

  string PeekPrev() const { return has_prev_ ? prev_tok_ : ""; }

  size_t PeekPrevTokenStart() const {
    return has_prev_ ? prev_tok_start_ : 0;
  }

  TokenType PeekPrevTokenType() const {
    return has_prev_ ? prev_tok_type_ : EOF_TYPE;
  }

  /// Returns the next token in the token stream.
  string Next() {
    if (!HasNext()) {
      // Error.
    }

    // First, preserve the next token as the previous token.
    prev_tok_ = next_tok_;
    prev_tok_type_ = next_tok_type_;
    prev_tok_start_ = next_tok_start_;
    has_prev_ = true;

    curr_pos_ = num_read_;
    GetNext();
    return prev_tok_;
  }

  /// Returns the next token&rsquo;s start position.
  size_t PeekTokenStart() const {
    return has_next_ ? next_tok_start_ : tellg();
  }

  /// Returns the type of the next token.
  TokenType PeekTokenType() const {
    return has_next_ ? next_tok_type_ : EOF_TYPE;
  }

  /// Returns the next token that would be returned by the \link Next
  /// \endlink method.  The return value of this method is only valid
  /// when \link HasNext \endlink returns <tt>true</tt>.
  string Peek() const { return has_next_ ? next_tok_ : ""; }

 private:
  void Init(const char *reserved_chars) {
    num_reserved_chars_ = strlen(reserved_chars);
    reserved_chars_ = new char[num_reserved_chars_];
    strcpy(reserved_chars_, reserved_chars);
    int num_reserved_words = sizeof(default_reserved_words)/sizeof(const char*);
    for (int i = 0; i < num_reserved_words; ++i) {
      reserved_words_.insert(string(default_reserved_words[i]));
    }
    GetNext();
  }

  /// Retrieves the next token from the <tt>istream</tt> wrapped by this
  /// stream tokenizer.
  void GetNext();

  /// Returns whether the specified character represents a
  /// &ldquo;reserved character&rdquo;.
  bool ReservedChar(char c) const {
    for (size_t i = 0; i < num_reserved_chars_; ++i) {
      if (c == reserved_chars_[i]) {
        return true;
      }
    }
    return false;
  }

  // data members

  // The stream itself.
  /// This data member is for use when we need to construct the is_ data member
  /// from a string at construction time.
  istringstream sstream_;
  /// The underlying byte stream of this token stream.
  istream &is_;

  // Information about special tokens.
  char *reserved_chars_;
  size_t num_reserved_chars_;
  set<string> reserved_words_;

  // Information about the current state of this stream.
  size_t curr_pos_;
  size_t num_read_;
  ostringstream oss_;

  // Information about the next token in this stream.
  string next_tok_;
  TokenType next_tok_type_;
  size_t next_tok_start_;
  bool has_next_;

  // Information about the previous token in this stream.
  string prev_tok_;
  TokenType prev_tok_type_;
  size_t prev_tok_start_;
  bool has_prev_;
};

}  // namespace reranker

#endif
