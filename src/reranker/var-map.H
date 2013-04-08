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

class Environment;

/// A base class for a mapping from variables of a specific type to their
/// values.
class VarMapBase {
 public:
  /// Constructs a base class for a concrete implementation providing
  /// a mapping from variables of a particular type to their values.
  ///
  /// \param name         the type name of the variables in this instance
  /// \param is_primitive whether this instance contains primitive or primitive
  ///                     vector variables
  VarMapBase(const string &name, Environment *env, bool is_primitive) :
      name_(name), env_(env), is_primitive_(is_primitive) { }

  /// Returns whether this instance contains primitive or primtive
  /// vector variables.
  virtual bool IsPrimitive() const { return is_primitive_; }
  /// Returns the type name of the variables of this instance.
  virtual const string &Name() const { return name_; }

  /// Returns whether the specified variable has a definition in this
  /// environment.
  virtual bool Defined(const string &varname) const = 0;

  /// Reads the next value (primitive or spec for constructing a \link
  /// reranker::Factory Factory\endlink-constructible object) from the
  /// specified stream tokenizer and sets the specified variable to
  /// that value.
  virtual void ReadAndSet(const string &varname, StreamTokenizer &st) = 0;
 protected:
  /// The type name of this VarMap.
  string name_;
  /// The Environment that holds this VarMap instance.
  Environment *env_;
  /// Whether this VarMap instance holds variables of primitive type
  /// or vector of primitives.
  bool is_primitive_;
};

/// An interface for an environment in which variables of various
/// types are mapped to their values.
class Environment {
 public:
  /// Returns whether the specified variable has been defined in this
  /// environment.
  virtual bool Defined(const string &varname) const = 0;

  /// Sets the specified variable to the value obtained from the following
  /// tokens available from the specified token stream.
  virtual void ReadAndSet(const string &varname, StreamTokenizer &st) = 0;

  /// Retrieves the type name of the specified variable.
  virtual const string &GetType(const string &varname) const = 0;

  /// Retrieves the VarMap instance for the specified variable.
  virtual VarMapBase *GetVarMap(const string &varname) = 0;

  /// Retrieves the VarMap instance for the specified type.
  virtual VarMapBase *GetVarMapForType(const string &type) = 0;
};

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

/// A container to hold the mapping between named variables of a specific
/// type and their values.
///
/// \tparam T the type of variables held by this variable map
template <typename T>
class VarMap : public VarMapBase {
 public:
  /// Constructs a mapping from variables of a particular type to their values.
  ///
  /// \param name         the type name of the variables in this instance
  /// \param is_primitive whether this instance contains primitive or primitive
  ///                     vector variables
  VarMap(const string &name, Environment *env, bool is_primitive = true) :
      VarMapBase(name, env, is_primitive) { }

  // Accessor methods

  /// Assigns the value of the specified variable to the object pointed to
  /// by the <tt>value</tt> parameter.
  ///
  /// \return whether the specified variable exists and the assignment
  ///         was successful
  bool Get(const string &varname, T *value) const {
    typename unordered_map<string, T>::const_iterator it = vars_.find(varname);
    if (it == vars_.end()) {
      return false;
    } else {
      *value = it->second;
      return true;
    }
  }

  /// \copydoc VarMapBase::Defined
  virtual bool Defined(const string &varname) const {
    return vars_.find(varname) != vars_.end();
  }

  /// Sets the specified variable to the specified value.
  void Set(const string &varname, T value) {
    vars_[varname] = value;
  }

  /// \copydoc VarMapBase::ReadAndSet
  void ReadAndSet(const string &varname, StreamTokenizer &st) {
    if (VAR_MAP_DEBUG >= 1) {
      cerr << "VarMap<" << Name() << ">::ReadAndSet: about to set varname "
           << varname << " of type " << typeid(T).name() << endl;
    }

    // First check if next token is an identifier and is a variable in
    // the environment, set varname to its value.
    if (env_->Defined(st.Peek())) {
      VarMapBase *var_map = env_->GetVarMap(st.Peek());
      VarMap<T> *typed_var_map = dynamic_cast<VarMap<T> *>(var_map);
      if (typed_var_map != NULL) {
        // Finally consume variable.
        string rhs_variable = st.Next();
        T value = T();
        // Retrieve rhs variable's value.
        bool success = typed_var_map->Get(rhs_variable, &value);
        // Set varname to the same value.
        if (VAR_MAP_DEBUG >= 1) {
          cerr << "VarMap<" << Name() << ">::ReadAndSet: setting variable "
               << varname << " to same value as rhs variable " << rhs_variable
               << endl;
        }
        if (success) {
          Set(varname, value);
        } else {
          // Error: we couldn't find the varname in this VarMap.
        }
      } else {
        // Error: inferred (or, eventually, declared) type of varname is
        // different from the type of the rhs variable.
      }
    } else {
      T value;
      Initializer<T> initializer(&value);
      initializer.Init(st, env_);
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
  }
 private:
  unordered_map<string, T> vars_;
};

}  // namespace reranker

#endif
