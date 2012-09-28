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
/// \file symbol-table.C
/// Provides the implementation of the reranker::StaticSymbolTable,
/// which holds an ever-growing symbol table.
/// \author dbikel@google.com (Dan Bikel)

#include <string>
#include <tr1/unordered_map>

#include "symbol-table.H"

namespace reranker {

using std::string;
using std::tr1::unordered_map;


string Symbols::null_symbol("");

unordered_map<string, int> StaticSymbolTable::symbols_;
unordered_map<int, string> StaticSymbolTable::indices_to_symbols_;

int
StaticSymbolTable::GetIndex(const string &symbol) {
  unordered_map<string, int>::iterator it = symbols_.find(symbol);
  if (it == symbols_.end()) {
    size_t new_index = symbols_.size();
    symbols_[symbol] = new_index;
    indices_to_symbols_[new_index] = symbol;
    return new_index;
  } else {
    return it->second;
  }
}

const string &
StaticSymbolTable::GetSymbol(int index) const {
  unordered_map<int, string>::const_iterator it =
      indices_to_symbols_.find(index);
  return it == indices_to_symbols_.end() ? Symbols::null_symbol : it->second;
}

int
LocalSymbolTable::GetIndex(const string &symbol) {
  unordered_map<string, int>::iterator it = symbols_.find(symbol);
  if (it == symbols_.end()) {
    size_t new_index = symbols_.size();
    symbols_[symbol] = new_index;
    indices_to_symbols_[new_index] = symbol;
    return new_index;
  } else {
    return it->second;
  }
}

const string &
LocalSymbolTable::GetSymbol(int index) const {
  unordered_map<int, string>::const_iterator it =
      indices_to_symbols_.find(index);
  return it == indices_to_symbols_.end() ? Symbols::null_symbol : it->second;
}

}  // namespace reranker
