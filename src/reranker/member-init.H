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
/// Provides a generic dynamic object factory.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_MEMBER_INIT_H_
#define RERANKER_MEMBER_INIT_H_

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <tr1/memory>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <vector>
#include <stdexcept>

#include "stream-tokenizer.H"

namespace reranker {

using std::ostringstream;
using std::tr1::shared_ptr;
using std::tr1::unordered_map;
using std::vector;

/// \class MemberInitializer
///
/// An interface that allows for a data member of a Factory-constructible type
/// to be initialized based on the next token or tokens from a token stream.
/// The data member may be an <tt>int</tt>, a <tt>double</tt>, a <tt>string</tt>
/// or a <tt>shared_ptr</tt> to another Factory-constructible type.
class MemberInitializer {
 public:
  MemberInitializer() : initialized_(0), required_(false) { }
  MemberInitializer(bool required) : initialized_(0), required_(required) { }
  virtual void Init(StreamTokenizer &st) = 0;
  /// Returns the number of times this member initializer&rsquo;s
  /// \link Init \endlink method has been invoked.
  virtual int Initialized() const { return initialized_; }
  virtual bool Required() const { return required_; }
 protected:
  /// The number of times this member initializer&rsquo;s Init method has
  /// been invoked.
  int initialized_;
  /// Whether this member is required to be initialized.
  bool required_;
};

template <typename T> class Factory;

/// \class Initializer
///
/// A class to allow \link Factory\endlink-constructible objects to
/// initialize data members that point to other \link
/// Factory\endlink-constructible objects.
///
/// \tparam T a <tt>shared_ptr</tt> to any type constructible by a Factory
template <typename T>
class Initializer : public MemberInitializer {
 public:
  Initializer(T *member) : member_(member) { }
  Initializer(T *member, bool required) :
      MemberInitializer(required), member_(member) { }
  virtual void Init(StreamTokenizer &st) {
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::IDENTIFIER) {
      ostringstream err_ss;
      err_ss << "FactoryInitializer: expected IDENTIFIER token at stream "
             << "position " << st.PeekTokenStart() << " but found "
             << StreamTokenizer::TypeName(token_type) << " token: \""
             << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    Factory<typename T::element_type> factory;
    (*member_) = factory.CreateOrDie(st);
    ++initialized_;
  }
 private:
  T *member_;
};

/// \class Initializer<int>
///
/// A specialization to allow Factory-constructible objects to initialize
/// <tt>int</tt> data members.
template<>
class Initializer<int> : public MemberInitializer {
 public:
  Initializer(int *member) : member_(member) { }
  Initializer(int *member, bool required) :
      MemberInitializer(required), member_(member) { }
  virtual void Init(StreamTokenizer &st) {
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::NUMBER) {
      ostringstream err_ss;
      err_ss << "IntInitializer: expected NUMBER token at stream "
             << "position " << st.PeekTokenStart() << " but found " 
             << StreamTokenizer::TypeName(token_type) << " token: \""
             << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    (*member_) = atoi(st.Next().c_str());
    ++initialized_;
  }
 private:
  int *member_;
};

/// \class Initializer<double>
///
/// A specialization to allow Factory-constructible objects to initialize
/// <tt>double</tt> data members.
template<>
class Initializer<double> : public MemberInitializer {
 public:
  Initializer(double *member) : member_(member) { }
  Initializer(double *member, bool required) :
      MemberInitializer(required), member_(member) { }
  virtual void Init(StreamTokenizer &st) {
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::NUMBER) {
      ostringstream err_ss;
      err_ss << "DoubleInitializer: expected NUMBER token at stream "
             << "position " << st.PeekTokenStart() << " but found "
             << StreamTokenizer::TypeName(token_type) << " token: \""
             << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    (*member_) = atof(st.Next().c_str());
    ++initialized_;
  }
 private:
  double *member_;
};

/// \class Initializer<bool>
///
/// A specialization to allow Factory-constructible objects to initialize
/// <tt>bool</tt> data members.
template<>
class Initializer<bool> : public MemberInitializer {
 public:
  Initializer(bool *member) : member_(member) { }
  Initializer(bool *member, bool required) :
      MemberInitializer(required), member_(member) { }
  virtual void Init(StreamTokenizer &st) {
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::STRING &&
        token_type != StreamTokenizer::NUMBER) {
      ostringstream err_ss;
      err_ss << "BoolInitializer: expected STRING or NUMBER token at stream "
             << "position " << st.PeekTokenStart() << " but found "
             << StreamTokenizer::TypeName(token_type) << " token: \""
             << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    if (token_type == StreamTokenizer::STRING) {
      (*member_) = st.Next() != "false";
    } else {
      double val = atof(st.Next().c_str());
      (*member_) = val != 0.0;
    }
    ++initialized_;
  }
 private:
  bool *member_;
};

/// \class Initializer<string>
///
/// A specialization to allow Factory-constructible objects to initialize
/// <tt>string</tt> data members.
template<>
class Initializer<string> : public MemberInitializer {
 public:
  Initializer(string *member) : member_(member) { }
  Initializer(string *member, bool required) :
      MemberInitializer(required), member_(member) { }
  virtual void Init(StreamTokenizer &st) {
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::STRING) {
      ostringstream err_ss;
      err_ss << "StringInitializer: expected STRING token at stream "
             << "position " << st.PeekTokenStart() << " but found "
             << StreamTokenizer::TypeName(token_type) << " token: \""
             << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    (*member_) = st.Next();
    ++initialized_;
  }
 private:
  string *member_;
};

/// A partial specialization to allow initialization of a vector of
/// any primitive type or any \link Factory\endlink-constructible
/// type.  The syntax for a vector initialization is a comma-separated
/// list of values.
template<typename T>
class Initializer<vector<T> > : public MemberInitializer {
 public:
  Initializer(vector<T> *member) : member_(member) { }
  Initializer(vector<T> *member, bool required) :
      MemberInitializer(required), member_(member) { }
  virtual void Init(StreamTokenizer &st) {
    while (st.Peek() != ")") {
      T vector_element;
      Initializer<T> *element_init = new Initializer<T>(&vector_element);
      element_init->Init(st);
      member_->push_back(vector_element);
      delete element_init;
      // Each vector element initializer must be followed by a comma
      // or the final closing parenthesis.
      if (st.Peek() != ","  && st.Peek() != ")") {
        ostringstream err_ss;
        err_ss << "Initializer<vector<T>>: "
               << "error: expected ',' or ')' at stream position "
               << st.PeekTokenStart() << " but found \"" << st.Peek() << "\"";
        throw std::runtime_error(err_ss.str());
      }
      // Read comma, if present.
      if (st.Peek() == ",") {
        st.Next();
      }
    }
  }
 private:
  vector<T> *member_;
};

/// \class Initializers
///
/// A container for all the member initializers for a particular
/// Factory-constructible instance.  This class provides an easy, consistent
/// syntax for Factory-constructible classes to specify which members they
/// want/need initialized by the Factory based on the specification string.
class Initializers {
 public:
  /// Forward the <tt>const_iterator</tt> typedef of the internal data
  /// structure, to make code compact and readable.
  typedef unordered_map<string, MemberInitializer *>::const_iterator
      const_iterator;
  /// Forward the <tt>iterator</tt> typedef of the internal data
  /// structure, to make code compact and readable.
  typedef unordered_map<string, MemberInitializer *>::iterator iterator;

  /// Constructs a new instance.
  Initializers() { }
  /// Destroys this instance.
  virtual ~Initializers() {
    for (iterator init_it = initializers_.begin();
         init_it != initializers_.end();
         ++init_it) {
      delete init_it->second;
    }
  }

  template<typename T>
  void Add(const string &name, T *member, bool required = false) {
    initializers_[name] = new Initializer<T>(member, required);
  }

  /// Returns a const iterator pointing to the beginning of the map
  /// from member names to pointers to \link MemberInitializer
  /// \endlink instances.
  const_iterator begin() const { return initializers_.begin(); }
  /// Returns a const iterator pointing to the end of the map from
  /// member names to pointers to \link MemberInitializer \endlink
  /// instances.
  const_iterator end() const { return initializers_.end(); }

  /// Returns an iterator pointing to the beginning of the map from
  /// member names to pointers to \link MemberInitializer \endlink
  /// instances.
  iterator begin() { return initializers_.begin(); }
  /// Returns an iterator pointing to the end of the map from member
  /// names to pointers to \link MemberInitializer \endlink instances.
  iterator end() { return initializers_.end(); }

  /// Returns a <tt>const_iterator</tt> pointing to the \link
  /// MemberInitializer \endlink associated with the specified name,
  /// or else \link end \endlink if no such \link MemberInitializer
  /// \endlink exists.
  const_iterator find(const string &name) const {
    return initializers_.find(name);
  }
  /// Returns an <tt>iterator</tt> pointing to the \link
  /// MemberInitializer \endlink associated with the specified name,
  /// or else \link end \endlink if no such \link MemberInitializer
  /// \endlink exists.
  iterator find(const string &name) {
    return initializers_.find(name);
  }
 private:
  unordered_map<string, MemberInitializer *> initializers_;
};

}  // namespace reranker

#endif
