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
/// Provides a map from variables of a specific type (primitives,
/// \link reranker::Factory Factory\endlink-constructible objects, or
/// vectors thereof) to their values.  These maps exist primarily to
/// support the \link reranker::Environment Environment \endlink class.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_VAR_MAP_H_
#define RERANKER_VAR_MAP_H_

#define VAR_MAP_DEBUG 1

#include <sstream>
#include <vector>

#include "member-init.H"
#include "stream-tokenizer.H"

namespace reranker {

using std::cerr;
using std::endl;
using std::ostream;
using std::ostringstream;
using std::tr1::shared_ptr;
using std::tr1::unordered_map;
using std::tr1::unordered_set;
using std::vector;

/// A template class that helps print out values with ostream& operator
/// support and vectors of those values.
///
/// \tparam T the type to print to an ostream
template <typename T>
class ValueString {
 public:
  string ToString(const T &value) {
    ostringstream oss;
    oss << value;
    return oss.str();
  }
};

/// A specialization of the ValueString class to support printing of
/// vectors of values.
///
/// \tparam T the element type for a vector to be printed out to an ostream
template <typename T>
class ValueString<vector<T> > {
 public:
  string ToString(const vector<T> &value) {
    ostringstream oss;
    oss << "{";
    typename vector<T>::const_iterator it = value.begin();
    if (it != value.end()) {
      oss << *it;
      ++it;
    }
    for (; it != value.end(); ++it) {
      oss << ", " << *it;
    }
    oss << "}";
    return oss.str();
  }
};

/// A base class for a mapping from variables of a specific type to their
/// values.
class VarMapBase {
 public:
  VarMapBase(const string &name, bool is_primitive) :
      name_(name), is_primitive_(is_primitive) { }

  virtual bool IsPrimitive() const { return is_primitive_; }
  virtual const string &Name() const { return name_; }

  virtual bool Defined(const string &varname) const = 0;
  virtual void ReadAndSet(const string &varname, StreamTokenizer &st) = 0;
 protected:
  /// The type name of this VarMap.
  string name_;
  /// Whether this VarMap instance holds variables of primitive type
  /// or vector of primitives.
  bool is_primitive_;
};

/// A container to hold the mapping between named variables of a specific
/// type and their values.
///
/// \tparam T the type of variables held by this variable map
template <typename T>
class VarMap : public VarMapBase {
 public:
  VarMap(const string &name, bool is_primitive = true) :
      VarMapBase(name, is_primitive) { }

  // Accessor methods
  bool Get(const string &varname, T *value) const {
    typename unordered_map<string, T>::const_iterator it = vars_.find(varname);
    if (it == vars_.end()) {
      return false;
    } else {
      *value = it->second;
      return true;
    }
  }

  virtual bool Defined(const string &varname) const {
    return vars_.find(varname) != vars_.end();
  }

  void Set(const string &varname, T value) {
    vars_[varname] = value;
  }

  /// Reads the next primitive value from the specified stream
  /// tokenizer and sets the specified variable to that value.
  void ReadAndSet(const string &varname, StreamTokenizer &st) {
    bool required = true;

    if (VAR_MAP_DEBUG >= 1) {
      cerr << "VarMap<" << Name() << ">::ReadAndSet: about to set varname "
           << varname << " of type " << typeid(T).name() << endl;
    }

    T value;
    Initializer<T> initializer(varname, &value, required);
    initializer.Init(st);
    Set(varname, value);

    if (VAR_MAP_DEBUG >= 1) {
      cerr << "VarMap<" << Name() << ">::ReadAndSet: set varname "
           << varname;
      if (IsPrimitive()) {
        ValueString<T> value_string;
        cerr << " to value " << value_string.ToString(value);
      }
      cerr << endl;
    }
  }
 private:
  unordered_map<string, T> vars_;
};


}  // namespace reranker

#endif
