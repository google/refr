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

#include <sstream>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include "factory.H"

namespace reranker {

using std::ostringstream;
using std::string;
using std::tr1::unordered_map;
using std::tr1::unordered_set;

template <typename T>
class ValueString {
 public:
  string ToString(const T &value) {
    ostringstream oss;
    oss << value;
    return oss.str();
  }
};

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

class VarMapBase {
 public:
  virtual bool Defined(const string &varname) const = 0;
  virtual void ReadAndSet(const string &varname, StreamTokenizer &st) = 0;
};

template <typename T>
class VarMap : public VarMapBase {
 public:
  VarMap(const string &name) : name_(name) { }

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
    T value;
    Initializer<T> initializer(varname, &value, required);
    initializer.Init(st);
    Set(varname, value);
    ValueString<T> value_string;
    cerr << "VarMap<" << name_ << ">::ReadAndSet: set varname "
         << varname << " to value " << value_string.ToString(value) << endl;
  }
 private:
  unordered_map<string, T> vars_;
  /// The name of this VarMap.
  string name_;
};

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
    BOOL,
    INT,
    DOUBLE,
    STRING,
    OBJECT,
    BOOL_VECTOR,
    INT_VECTOR,
    DOUBLE_VECTOR,
    STRING_VECTOR,
    OBJECT_VECTOR,

    NUM_TYPES,

    NOT_A_TYPE
  };

  static const string TypeName(VarType type) {
    switch (type) {
      case BOOL:
        return "bool";
      case INT:
        return "int";
      case DOUBLE:
        return "double";
      case STRING:
        return "string";
      case OBJECT:
        return "Object";
      case BOOL_VECTOR:
        return "bool[]";
      case INT_VECTOR:
        return "int[]";
      case DOUBLE_VECTOR:
        return "double[]";
      case STRING_VECTOR:
        return "string[]";
      case OBJECT_VECTOR:
        return "Object[]";
      case NUM_TYPES:
        throw std::runtime_error("Environment::TypeName invoked on bad enum "
                                 " constant NUM_TYPES");
      case NOT_A_TYPE:
        throw std::runtime_error("Environment::TypeName invoked on bad enum "
                                 " constant NOT_A_TYPE");
    }
    return "ERROR";
  }

  Environment(bool debug = false) {
    for (FactoryContainer::iterator factory_it = FactoryContainer::begin();
         factory_it != FactoryContainer::end(); ++factory_it) {
      unordered_set<string> registered;
      (*factory_it)->CollectRegistered(registered);
      string base_name = (*factory_it)->BaseName();
      shared_ptr<EnvFactoryBase> env_factory =
          (*factory_it)->CreateEnvFactory();
      for (unordered_set<string>::const_iterator it = registered.begin();
           it != registered.end(); ++it) {
        factories_[*it] = env_factory;

        if (debug) {
          cerr << "Environment: associating concrete typename " << *it
               << " with factory for " << (*factory_it)->BaseName() << endl;
        }
      }
    }

    var_map_.resize(NUM_TYPES);
    for (int type = BOOL; type < NUM_TYPES; ++type) {
      string type_name = TypeName((VarType)type);
      switch (type) {
        case BOOL:
          var_map_[type] = new VarMap<bool>(type_name);
          break;
        case INT:
          var_map_[type] = new VarMap<int>(type_name);
          break;
        case DOUBLE:
          var_map_[type] = new VarMap<double>(type_name);
          break;
        case STRING:
          var_map_[type] = new VarMap<string>(type_name);
          break;
        case BOOL_VECTOR:
          var_map_[type] = new VarMap<vector<bool> >(type_name);
          break;
        case INT_VECTOR:
          var_map_[type] = new VarMap<vector<int> >(type_name);
          break;
        case DOUBLE_VECTOR:
          var_map_[type] = new VarMap<vector<double> >(type_name);
          break;
        case STRING_VECTOR:
          var_map_[type] = new VarMap<vector<string> >(type_name);
          break;
      }
    }
  }

  ~Environment() {
    for (int type = BOOL; type < NUM_TYPES; ++type) {
      delete var_map_[type];
    }
  }

  void ReadAndSet(const string &varname, StreamTokenizer &st) {
    bool is_vector =
        st.PeekTokenType() == StreamTokenizer::RESERVED_CHAR &&
        st.Peek() == "{";
    if (is_vector) {
      // Consume open brace.
      st.Next();
    }
    string next_tok = st.Peek();
    VarType type = InferType(st, is_vector);

    cerr << "Environment::ReadAndSet: "
         << "next_tok=" << next_tok << "; type=" << TypeName(type) << endl;

    if (type == OBJECT || type == OBJECT_VECTOR) {
      // Verify that next_tok is a concrete typename.
      if (factories_.find(next_tok) == factories_.end()) {
        ostringstream err_ss;
        err_ss << "Interpreter: error: variable "
               << varname << " appears to be of type " << TypeName(type)
               << " but token " << next_tok
               << " is not a concrete object typename";
        throw std::runtime_error(err_ss.str());
      }
    }
    if (type == NOT_A_TYPE) {
      // Error.
    }
    var_map_[type]->ReadAndSet(varname, st);
    types_[varname] = type;
  }

  /// Infer the type based on the next token and its token type.
  static VarType InferType(const StreamTokenizer &st, bool is_vector) {
    string next_tok = st.Peek();
    switch (st.PeekTokenType()) {
      case StreamTokenizer::RESERVED_WORD:
        if (next_tok == "true" || next_tok == "false") {
          return is_vector ? BOOL_VECTOR : BOOL;
        } else {
          return NOT_A_TYPE;
        }
        break;
      case StreamTokenizer::STRING:
        return is_vector ? STRING_VECTOR : STRING;
        break;
      case StreamTokenizer::NUMBER:
        {
          // If a token is a NUMBER, it is a double iff it contains a
          // decimal point.
          size_t dot_pos = next_tok.find('.');
          if (dot_pos != string::npos) {
            return is_vector ? DOUBLE_VECTOR : DOUBLE;
          } else {
            return is_vector ? INT_VECTOR : INT;
          }
        }
        break;
      case StreamTokenizer::IDENTIFIER:
        {
          return is_vector ? OBJECT_VECTOR : OBJECT;
        }
        break;
      default:
        return NOT_A_TYPE;
    }
  }

 private:

  /// A map from concrete typenames to the EnvFactory instances that can
  /// construct instances of those typenames.
  unordered_map<string, shared_ptr<EnvFactoryBase> > factories_;
  /// A map from all variable names to their types.
  unordered_map<string, VarType> types_;
  /// A map from all object variable names to their concrete types.
  /// The keys of this map are a subset of those in the <tt>types_</tt>
  /// data member.
  unordered_map<string, string> obj_types_;

  vector<VarMapBase *> var_map_;

  /// A map from variable names to the \link reranker::EnvFactoryBase
  /// EnvFactoryBase \endlink instances that contain their values,
  /// whether they are object or vector-of-object variables (the \link
  /// \link reranker::EnvFactory EnvFactory \endlink class holds
  /// both).
  unordered_map<string, shared_ptr<EnvFactoryBase> > obj_vars_;
};

}  // namespace reranker

#endif
