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
/// Implementation of the \link reranker::StreamTokenizer StreamTokenizer
/// \endlink class.
/// \author dbikel@google.com (Dan Bikel)

#include <ctype.h>
#include <stdexcept>

#include "stream-tokenizer.H"

namespace reranker {

void
StreamTokenizer::GetNext() {
  if (!is_.good()) {
    has_next_ = false;
    return;
  }
  // Get first character of next token.
  char c;
  bool is_whitespace = true;
  while (is_whitespace) {
    c = is_.get();
    if (!is_.good()) {
      has_next_ = false;
      return;
    }
    oss_ << c;
    ++num_read_;
    is_whitespace = isspace(c);
  }

  has_next_ = true;
  next_tok_start_ = num_read_ - 1;
  bool next_tok_complete = false;
  next_tok_.clear();
  if (Reserved(c)) {
    next_tok_ += c;
    next_tok_complete = true;
    next_tok_type_ = RESERVED;
  } else if (c == '"') {
    // We've got a string literal, so keep reading characters,
    // until hitting a non-escaped double quote.
    streampos string_literal_start_pos = num_read_ - 1;
    bool found_closing_quote = false;
    while (is_.good()) {
      c = is_.get();
      if (is_.good()) {
        oss_ << c;
        ++num_read_;
        if (c == '"') {
          found_closing_quote = true;
          break;
        } else if (c == '\\') {
          c = is_.get();
          if (is_.good()) {
            oss_ << c;
            ++num_read_;
          }
        }
      }
      if (is_.good()) {
        next_tok_ += c;
      }
    }
    if (!found_closing_quote) {
      cerr << "reranker::StreamTokenizer: error: could not find closing "
           << "double quote for string literal beginning at stream index "
           << string_literal_start_pos
           << "; partial string literal read: \""
           << next_tok_ << endl;
      throw std::runtime_error("unclosed string literal");
    }
    next_tok_complete = true;
    next_tok_type_ = STRING;
  } else {
    // This is a number or C++ identifier token, so add first character;
    // the remainder of the token will be handled after this switch statement.
    next_tok_ += c;
    next_tok_type_ = (c >= '0' && c <= '9') ? NUMBER : IDENTIFIER;
  }
  if (!next_tok_complete) {
    // The current token is a number or C++ identifier, so we keep
    // reading characters until hitting a "reserved word" character,
    // a whitespace character or EOF.
    bool done = false;
    while (!done && is_.good()) {
      c = is_.get();
      if (is_.good()) {
        if (Reserved(c) || c == '"' || isspace(c)) {
          is_.putback(c);
          done = true;
        } else {
          ++num_read_;
          oss_ << c;
          next_tok_ += c;
        }
      }
    }
  }
}

}  // namespace reranker
