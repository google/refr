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
/// \file symbol-table.H
/// Provides the reranker::Symbols interface as well as the
/// reranker::StaticSymbolTable implementation.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_SYMBOL_TABLE_H_
#define RERANKER_SYMBOL_TABLE_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace reranker {

using std::cerr;
using std::endl;
using std::ostream;
using std::string;
using std::unordered_map;
using std::vector;

/// \class Symbols
///
/// An interface specifying a converter from symbols (strings) to int indices.
class Symbols {
 public:
  virtual ~Symbols() { }
  typedef unordered_map<string, int>::const_iterator const_iterator;

  virtual const_iterator begin() = 0;
  virtual const_iterator end() = 0;

  /// Returns the number of symbols in this table.
  virtual size_t size() const = 0;

  /// Converts the specified symbol to a unique integer.  The behavior of
  /// this class is undefined if the specified symbol is the empty string.
  ///
  /// \param symbol the symbol to convert
  /// \return a unique integer for the specified symbol
  virtual int GetIndex(const string &symbol) = 0;

  /// Returns the unique symbol for the specified index, or the empty string
  /// if the specified index does not map to any symbol in this symbol table.
  virtual const string &GetSymbol(int index) const = 0;

  virtual void SetIndex(const string &symbol, int index) = 0;

  /// Clears all symbols from this symbol table.
  virtual void Clear() = 0;

  /// Creates a newly-constructed clone of this Symbols instance that has
  /// the same runtime type.  The caller will be responsible for destroying
  /// the returned instance.
  ///
  /// \return a clone of this Symbols instance that has the same runtime
  ///         type
  virtual Symbols *Clone() const = 0;

  /// Outputs the symbol table to the specified output stream in a simple
  /// format, one symbol-to-index mapping per line:
  /// \code <symbol> <tab> <index> \endcode
  /// where
  /// \code
  /// <symbol> ::= an indexed string symbol
  /// <tab>    ::= a tab character, '\t'
  /// <index>  ::= a unique integer for <symbol>
  /// \endcode
  ///
  /// \param os the output stream to which to output the symbol table
  /// \return the specified output stream
  virtual ostream &Output(ostream &os) = 0;
 protected:
  static string null_symbol;  
};

/// \class StaticSymbolTable
///
/// A converter from symbols (strings) to int indices.  This symbol
/// table implementation always adds symbols to its internal symbol
/// table and never removes them.
/// \par Implementation note:
/// Instances of this class provide the interface to convert symbols
/// to <tt>int</tt>&rsquo;s, but underlyingly there is a static map,
/// so multiple instances of this class are all &ldquo;viewing&rdquo;
/// the same static data structure, hence the name of this Symbols
/// implementation.
class StaticSymbolTable : public Symbols {
 public:
  virtual const_iterator begin() {return symbols_.begin(); }
  virtual const_iterator end() { return symbols_.end(); }

  /// \copydoc Symbols::size
  virtual size_t size() const { return symbols_.size(); }

  /// Converts the specified symbol to a unique integer.
  ///
  /// TODO(dbikel,kbhall): Protect access to static symbol table via
  ///                      a mutex of some kind.
  ///
  /// \param symbol the symbol to convert
  /// \return a unique integer for the specified symbol
  virtual int GetIndex(const string &symbol);

  /// Returns the unique symbol for the specified index, or the empty string
  /// if the specified index does not map to any symbol in this symbol table.
  virtual const string &GetSymbol(int index) const;

  virtual void SetIndex(const string &symbol, int index) {
    unordered_map<string, int>::const_iterator it = symbols_.find(symbol);
    if (it != symbols_.end()) {
      // TODO(dbikel): Emit warning message?
      int index = GetIndex(symbol);
      symbols_.erase(symbol);
      indices_to_symbols_.erase(index);
    }
    symbols_[symbol] = index;
    indices_to_symbols_[index] = symbol;
  }

  /// \copydoc Symbols::Clear
  virtual void Clear() {
    symbols_.clear();
    indices_to_symbols_.clear();
  }

  /// \copydoc Symbols::Clone
  virtual Symbols *Clone() const {
    return new StaticSymbolTable();
  }

  /// \copydoc Symbols::Output
  virtual ostream &Output(ostream &os) {
    for (unordered_map<string, int>::const_iterator it = symbols_.begin();
         it != symbols_.end();
         ++it) {
      os << it->first << "\t" << it->second << "\n";
    }
    os.flush();
    return os;
  }

 private:
  // static data members
  static unordered_map<string, int> symbols_;
  static unordered_map<int, string> indices_to_symbols_;
};

/// \class LocalSymbolTable
///
/// A symbol table that stores the mapping from symbols to <tt>int</tt>&rsquo;s
/// and <i>vice versa</i> in local (non-static) data structures.
class LocalSymbolTable : public Symbols {
 public:
  virtual const_iterator begin() {return symbols_.begin(); }
  virtual const_iterator end() { return symbols_.end(); }

  /// \copydoc Symbols::size
  virtual size_t size() const { return symbols_.size(); }

  /// Returns the unique index for the specified symbol.
  ///
  /// \param symbol the symbol whose index is to be retrieved
  virtual int GetIndex(const string &symbol);

  /// \copydoc Symbols::GetSymbol
  virtual const string &GetSymbol(int index) const;

  virtual void SetIndex(const string &symbol, int index) {
    unordered_map<string, int>::const_iterator it = symbols_.find(symbol);
    if (it != symbols_.end()) {
      cerr << "LocalSymbolTable::SetIndex: warning: symbol \"" << symbol
           << "\" already has an index " << it->second << "; new index: "
           << index << endl;
      int index = GetIndex(symbol);
      symbols_.erase(symbol);
      indices_to_symbols_.erase(index);
    }
    symbols_[symbol] = index;
    indices_to_symbols_[index] = symbol;
  }

  /// \copydoc Symbols::Clear
  virtual void Clear() {
    symbols_.clear();
    indices_to_symbols_.clear();
  }

  /// \copydoc Symbols::Clone
  virtual Symbols *Clone() const {
    return new LocalSymbolTable(*this);
  }

  /// \copydoc Symbols::Output
  virtual ostream &Output(ostream &os) {
    for (unordered_map<string, int>::const_iterator it = symbols_.begin();
         it != symbols_.end();
         ++it) {
      os << it->first << "\t" << it->second << "\n";
    }
    os.flush();
    return os;
  }

 private:
  // data members
  unordered_map<string, int> symbols_;
  unordered_map<int, string> indices_to_symbols_;
};

}  // namespace reranker

#endif
