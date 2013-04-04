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
/// Implementation of the Environment class.
/// \author dbikel@google.com (Dan Bikel)

#include "environment.H"

namespace reranker {

Environment::Environment(int debug) {
  debug_ = debug;

  // Set up VarMap instances for each of the primitive types and their vectors.
  var_map_["bool"] = new VarMap<bool>("bool");
  var_map_["int"] = new VarMap<int>("int");
  var_map_["double"] = new VarMap<double>("double");
  var_map_["string"] = new VarMap<string>("string");
  var_map_["bool[]"] = new VarMap<vector<bool> >("bool[]");
  var_map_["int[]"] = new VarMap<vector<int> >("int[]");
  var_map_["double[]"] = new VarMap<vector<double> >("double[]");
  var_map_["string[]"] = new VarMap<vector<string> >("string[]");

  // Set up VarMap instances for each of the Factory-constructible types
  // and their vectors.
  for (FactoryContainer::iterator factory_it = FactoryContainer::begin();
       factory_it != FactoryContainer::end(); ++factory_it) {
    unordered_set<string> registered;
    (*factory_it)->CollectRegistered(registered);
    string base_name = (*factory_it)->BaseName();

    // Create type-specific VarMap from the Factory and add to var_map_.
    VarMapBase *obj_var_map = (*factory_it)->CreateVarMap();
    var_map_[obj_var_map->Name()] = obj_var_map;

    if (debug_ >= 2) {
      cerr << "Environment: created VarMap for " << obj_var_map->Name()
           << endl;
    }

    // Create VarMap for vectors of shared_object of T and add to var_map_.
    VarMapBase *obj_vector_var_map = (*factory_it)->CreateVectorVarMap();
    var_map_[obj_vector_var_map->Name()] = obj_vector_var_map;

    if (debug_ >= 2) {
      cerr << "Environment: created VarMap for " << obj_vector_var_map->Name()
           << endl;
    }

    for (unordered_set<string>::const_iterator it = registered.begin();
         it != registered.end(); ++it) {
      const string &concrete_type_name = *it;

      unordered_map<string, string>::const_iterator concrete_to_factory_it =
          concrete_to_factory_type_.find(concrete_type_name);
      if (concrete_to_factory_it != concrete_to_factory_type_.end()) {
        // Warn user that there are two entries for the same conrete type
        // (presumably to different abstract factory types).
        cerr << "Environment: WARNING: trying to override existing "
             << "concrete-to-factory type mapping ["
             << concrete_type_name << " --> " << concrete_to_factory_it->second
             << "] with [" << concrete_type_name << " --> " << base_name
             << endl;
      }
      concrete_to_factory_type_[concrete_type_name] = base_name;

      if (debug_ >= 2) {
        cerr << "Environment: associating concrete typename "
             << concrete_type_name
             << " with factory for " << base_name << endl;
      }
    }
  }
}

void
Environment::ReadAndSet(const string &varname, StreamTokenizer &st) {
  bool is_vector =
      st.PeekTokenType() == StreamTokenizer::RESERVED_CHAR &&
      st.Peek() == "{";
  if (is_vector) {
    // Consume open brace.
    st.Next();
  } else if (st.PeekTokenType() == StreamTokenizer::RESERVED_CHAR ||
             (st.PeekTokenType() == StreamTokenizer::RESERVED_WORD &&
              st.Peek() != "true" && st.Peek() != "false")) {
    ostringstream err_ss;
    err_ss << "Environment: error: expected type but found token \""
           << st.Peek() << "\" of type "
           << StreamTokenizer::TypeName(st.PeekTokenType());
    throw std::runtime_error(err_ss.str());
  }
  string next_tok = st.Peek();
  bool is_object_type = false;

  string type = InferType(st, is_vector, &is_object_type);

  if (is_object_type) {
    // Verify that next_tok is a concrete typename.
    unordered_map<string, string>::const_iterator it =
        concrete_to_factory_type_.find(next_tok);
    if (it == concrete_to_factory_type_.end()) {
      ostringstream err_ss;
      err_ss << "Environment: error: variable "
             << varname << " appears to be of type " << type
             << " but token " << next_tok
             << " is not a concrete object typename";
      throw std::runtime_error(err_ss.str());
    }

    // Set type to be abstract factory type.
    if (debug_ >= 1) {
      cerr << "Environment::ReadAndSet: concrete type is " << type
           << "; mapping to abstract Factory type " << it->second << endl;
    }
    type = it->second;
  }

  if (debug_ >= 1) {
    cerr << "Environment::ReadAndSet: "
         << "next_tok=" << next_tok << "; type=" << type << endl;
  }


  if (type == "") {
    ostringstream err_ss;
    err_ss << "Environment: error: could not infer type for variable "
           << varname;
    throw std::runtime_error(err_ss.str());
  }

  // Check that type is a key in var_map_.

  var_map_[type]->ReadAndSet(varname, st);
  types_[varname] = type;
}

string
Environment::InferType(const StreamTokenizer &st, bool is_vector,
                       bool *is_object_type) {
  *is_object_type = false;
  string next_tok = st.Peek();
  switch (st.PeekTokenType()) {
    case StreamTokenizer::RESERVED_WORD:
      if (next_tok == "true" || next_tok == "false") {
        return is_vector ? "bool[]" : "bool";
      } else {
        return "";
      }
      break;
    case StreamTokenizer::STRING:
      return is_vector ? "string[]" : "string";
      break;
    case StreamTokenizer::NUMBER:
      {
        // If a token is a NUMBER, it is a double iff it contains a
        // decimal point.
        size_t dot_pos = next_tok.find('.');
        if (dot_pos != string::npos) {
          return is_vector ? "double[]" : "double";
        } else {
          return is_vector ? "int[]" : "int";
        }
      }
      break;
    case StreamTokenizer::IDENTIFIER:
      {
        *is_object_type = true;
        string type = is_vector ? next_tok + "[]" : next_tok;
        return type;
      }
      break;
    default:
      return "";
  }
}

}  // namespace reranker
