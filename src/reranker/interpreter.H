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

#include "environment-impl.H"

namespace reranker {

using std::iostream;
using std::ifstream;

class EnvironmentImpl;

/// Provides an interpreter for assigning primitives and
/// Factory-constructible objects to named variables, as well as
/// vectors thereof.  The interpreter maintains an internal
/// environment whereby previously defined variables may be used in
/// the definition of subsequent ones.  The syntax of this language
/// extends the syntax of the \link reranker::Factory Factory \endlink
/// class, described in the documentation of the \link
/// reranker::Factory::CreateOrDie Factory::CreateOrDie \endlink
/// method.
///
/// Statements in this language look like the following:
/// \code
/// // This is a comment.
/// b = true;  // assigns the value true to the boolean variable named "b"
/// f = 1;     // assigns the int value 1 to the variable "f"
/// g = 2.4;   // assigns the double value 2.4 to the variable "g"
/// n = "foo"; // assigns the string value "foo" to the variable "n"
/// b_vec = {true, false, true};  // assigns a vector of bool to "b_vec"
///
/// // Constructs an object of abstract type Model and assigns it to "m1"
/// m1 = PerceptronModel(name("foo"));
///
/// // Constructs an object of type Model and assigns it to "m2", crucially
/// // using the previously defined string variable "n" as the value for
/// // the PerceptronModel's name parameter.
/// m2 = PerceptronModel(name(n));
///
/// // Constructs a vector of Model objects and assigns it to "m_vec".
/// m_vec = {m2, PerceptronModel(name("bar"))};
/// \endcode
///
/// More formally, a statement in this language must conform to the
/// following grammar, defined on top of the BNF syntax in the
/// documentation of the \link reranker::Factory::CreateOrDie
/// Factory::CreateOrDie \endlink method:
/// <table border=0>
/// <tr>
///   <td><tt>\<statement_list\></tt></td>
///   <td><tt>::=</tt></td>
///   <td><tt>[ \<statement\> ]*</tt></td>
/// </tr>
/// <tr>
///   <td><tt>\<statement\></tt></td>
///   <td><tt>::=</tt></td>
///   <td><tt>\<variable_name\> '=' \<value\> ';' </tt></td>
/// </tr>
/// <tr>
///   <td><tt>\<variable_name\></tt></td>
///   <td><tt>::=</tt></td>
///   <td>any valid C++ identifier</td>
/// </tr>
/// <tr>
///   <td valign=top><tt>\<value\></tt></td>
///   <td valign=top><tt>::=</tt></td>
///   <td valign=top><tt>\<literal\> | '{' \<literal_list\> '}' |<br>
///                      \<spec\> | '{' \<spec_list\> '}'</tt>
///   </td>
/// </tr>
/// </table>
///
/// The above grammar doesn&rsquo;t contain rules covering C++ style
/// line comments, but they have the same behavior in this language as
/// they do in C++, i.e., everything after the <tt>//</tt> to the end
/// of the current line is treated as a comment and ignored.  There
/// are no C-style comments in this language.
class Interpreter {
 public:
  /// Constructs a new instance with the specified debug level.  The
  /// wrapped \link reranker::Environment Environment \endlink will
  /// also have the specified debug level.
  Interpreter(int debug = 0) : debug_(debug) {
    env_ = new EnvironmentImpl(debug);
  }

  /// Destroys this interpreter.
  virtual ~Interpreter() {
    delete env_;
  }

  /// Evaluates the statements in the specified text file.
  void Eval(const string &filename) {
    filename_ = filename;
    ifstream file(filename_.c_str());
    Eval(file);
  }

  /// Evaluates the statements in the specified string.
  void EvalString(const string& input) {
    StreamTokenizer st(input);
    Eval(st);
  }

  /// Evaluates the statements in the specified stream.
  void Eval(istream &is) {
    StreamTokenizer st(is);
    Eval(st);
  }


  void PrintEnv(ostream &os) const {
    env_->Print(os);
  }

  /// Retrieves the value of the specified variable.  It is an error
  /// if the type of the specified pointer to a value object is different
  /// from the specified variable in this interpreter&rsquo;s environment.
  ///
  /// \tparam the type of value object being set by this method
  ///
  /// \param varname the name of the variable for which to retrieve the value
  /// \param value   a pointer to the object whose value to be set by this
  ///                method
  template<typename T>
  bool Get(const string &varname, T *value) const {
    return env_->Get(varname, value);
  }

  /// Returns a pointer to the environment of this interpreter.
  /// Crucially, this method returns a pointer to the Environment
  /// implementation class, \link reranker::EnvironmentImpl
  /// EnvironmentImpl\endlink, so that its templated \link
  /// reranker::EnvironmentImpl::Get EnvironmentImpl::Get \endlink
  /// method may be invoked.
  EnvironmentImpl *env() { return env_; }

 private:
  void Eval(StreamTokenizer &st);

  void WrongTokenError(size_t pos,
                       const string &expected,
                       const string &found,
                       StreamTokenizer::TokenType found_type) const;

  void WrongTokenTypeError(size_t pos,
                           StreamTokenizer::TokenType expected,
                           StreamTokenizer::TokenType found,
                           const string &token) const;

  /// The environment of this interpreter.
  EnvironmentImpl *env_;

  /// The name of the file being interpreted, or the empty string if there
  /// is no file associated with the stream being interpreted.
  string filename_;

  /// The debug level of this interpreter and its associated environment.
  int debug_;
};

}  // namespace reranker

#endif
