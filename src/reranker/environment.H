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
/// Provides an environment for variables and their values, either primitive
/// or Factory-constructible objects.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_ENVIRONMENT_H_
#define RERANKER_ENVIRONMENT_H_

#include <tr1/unordered_map>

#include "factory.H"

namespace reranker {

using std::tr1::unordered_map;

/// Provides a set of named variables and their types, as well as the values
/// for those variable.
///
/// N.B.: Primitive values are stored as strings, and are converted on demand
/// to a concrete primitive type, one of <tt>{bool,int,double,string}</tt>.
/// Because of this, this class only considers variables to have one of
/// four types (even if their concrete types are more specific):
/// <ul>
/// <li> primitive
/// <li> vector of a primitive type
/// <li> \link reranker::Factory Factory\endlink-constructible type
/// <li> vector of a \link reranker::Factory Factory\endlink-constructible type
/// </ul>
class Environment {
 public:
  enum VarType {
    PRIMITIVE,
    PRIMITIVE_VECTOR,
    OBJECT,
    OBJECT_VECTOR,
  };

  Environment() { }
 private:
  /// A map from primitive variable names to their values, encoded as strings.
  unordered_map<string, string> primitives_;
  /// A map from vector-of-primitive variable names to their values,
  /// encoded as <tt>vector&lt;string&gt;</tt> objects.
  unordered_map<string, vector<string> > primitive_vectors_;
  /// A map from variable names to the \link reranker::EnvFactoryBase
  /// EnvFactoryBase \endlink instances that contain their values,
  /// whether they are object or vector-of-object variables (the \link
  /// \link reranker::EnvFactory EnvFactory \endlink class holds
  /// both).
  unordered_map<string, shared_ptr<EnvFactoryBase> > factory_vars_;
};

}  // namespace reranker

#endif
