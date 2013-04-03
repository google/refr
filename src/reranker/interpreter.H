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
/// Provides an interpreter for assigning primitives and Factory-constructible
/// objects to named variables, as well as vectors thereof.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_INTERPRETER_H_
#define RERANKER_INTERPRETER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include "environment.H"

namespace reranker {

using std::iostream;
using std::ifstream;

class Interpreter {
 public:
  Interpreter() { }
  virtual ~Interpreter() { }

  /// Evaluates the statements in the specified text file.
  void Eval(const string &filename) {
    filename_ = filename;
    ifstream file(filename_.c_str());
    Eval(file);
  }

  void EvalString(const string& input) {
    StreamTokenizer st(input);
    Eval(st);
  }

  /// Evaluates the statements in the specified stream.
  void Eval(istream &is) {
    StreamTokenizer st(is);
    Eval(st);
  }

  Environment &env() { return env_; }

 private:
  void Eval(StreamTokenizer &st);

  void WrongTokenError(size_t pos,
                       const string &expected, const string &found) const;

  void WrongTokenTypeError(size_t pos,
                           StreamTokenizer::TokenType expected,
                           StreamTokenizer::TokenType found,
                           const string &token) const;
  /// The environment of this interpreter.
  ///
  /// TODO(dbikel): Possibly turn this into a stack.
  Environment env_;

  string filename_;
};

}  // namespace reranker

#endif
