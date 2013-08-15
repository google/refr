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
/// Interpreter implementation.
/// Author: dbikel@google.com (Dan Bikel)

#include <sstream>

#include "interpreter.H"

using namespace std;

namespace reranker {

using std::ostringstream;

/// TODO(dbikel): Support explicitly declared type specifies instead of
///               always trying to infer the type.  I.e., allow
///                 int f = 1;
///               in addition to
///                 f = 1;
///               Crucially, this will allow you to declare the type
///               of empty vectors, such as
///                 FeatureExtractor[] extractors = {};
///               Currently, empty vectors are only possible when
///               initializing a vector member of a Factory-constructible
///               object (since the type is known beforehand).
void
Interpreter::Eval(StreamTokenizer &st) {
  // Keeps reading assignment statements until there are no more tokens.
  while (st.PeekTokenType() != StreamTokenizer::EOF_TYPE) {
    // Read variable name.
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::IDENTIFIER) {
      WrongTokenTypeError(st.PeekTokenStart(), StreamTokenizer::IDENTIFIER,
                          token_type, st.Peek());
    }
    string varname = st.Next();

    // Next, read equals sign.
    token_type = st.PeekTokenType();
    if (st.Peek() != "=") {
      WrongTokenError(st.PeekTokenStart(), "=", st.Peek(), st.PeekTokenType());
    }

    // Consume equals sign.
    st.Next();

    if (st.PeekTokenType() == StreamTokenizer::EOF_TYPE) {
      ostringstream err_ss;
      err_ss << "Interpreter:" << filename_
             << ": error: unexpected EOF at stream position "
             << st.tellg();
      throw std::runtime_error(err_ss.str());
    }

    // Consume and set the value for this variable in the environment.
    env_->ReadAndSet(varname, st);

    token_type = st.PeekTokenType();
    if (st.Peek() != ";") {
      WrongTokenError(st.PeekTokenStart(), ";", st.Peek(), st.PeekTokenType());
    }
    // Consume semicolon.
    st.Next();
  }
}

void
Interpreter::WrongTokenError(size_t pos,
                             const string &expected,
                             const string &found,
                             StreamTokenizer::TokenType found_type) const {
  ostringstream err_ss;
  err_ss << "Interpreter:" << filename_ << ": at stream pos " << pos
         << " expected token \"" << expected << "\" but found \"" << found
         << "\" (token type: " << StreamTokenizer::TypeName(found_type)
         << ")";
  throw std::runtime_error(err_ss.str());
}

void
Interpreter::WrongTokenTypeError(size_t pos,
                                 StreamTokenizer::TokenType expected,
                                 StreamTokenizer::TokenType found,
                                 const string &token) const {
  ostringstream err_ss;
  err_ss << "Interpreter:" << filename_ << ": at stream pos " << pos
         << " expected token type " << StreamTokenizer::TypeName(expected)
         << " but found " << StreamTokenizer::TypeName(found)
         << "; token=\"" << token << "\"";
  throw std::runtime_error(err_ss.str());
}

}  // namespace reranker
