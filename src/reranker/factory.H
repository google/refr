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

#ifndef RERANKER_FACTORY_H_
#define RERANKER_FACTORY_H_

#include <iostream>
#include <sstream>
#include <tr1/memory>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <vector>
#include <stdexcept>

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

class FactoryBase;

/// A base class for all \link EnvFactory \endlink instances, which
/// provide type-specific mappings between variable names and
/// \link Factory\endlink-constructible objects.
class EnvFactoryBase {
 public:
  /// Sets the variable with the specified name to a newly constructed
  /// object by the following tokens in the specified stream tokenizer.
  virtual void SetVar(const string &varname, StreamTokenizer &st) = 0;

  /// Creates or appends to a vector associated with the variable of the
  /// specified name.  The object appended to the vector is the one
  /// created by consuming the next token or tokens from the specified
  /// token stream.
  virtual void VectorAppend(const string &varname,
                            StreamTokenizer &st) = 0;

  /// Returns whether the specified variable has been defined by this
  /// environment factory instance, either as an object variable
  /// or a vector-of-object variable.
  virtual bool Defined(const string &varname) const = 0;

  /// Returns whether the specified variable has been defined as an
  /// object variable by this environment factory instance.
  virtual bool DefinedObject(const string &varname) const = 0;

  /// Returns whether the specified variable has been defined as a
  /// vector-of-object variable by this environment factory instance.
  virtual bool DefinedVector(const string& varname) const = 0;

  /// Returns the base class of the \link Factory \endlink wrapped by
  /// this EnvFactory instance.
  virtual const FactoryBase &GetFactoryBase() const = 0;
};

/// A base class for all \link Factory \endlink instances, specifying a few
/// virtual methods.
class FactoryBase {
 public:
  /// Clears the (possibly static) data of this factory.
  /// \p
  /// Note that invoking this method will prevent the factory from functioning!
  /// It should only be invoked when the factory is no longer needed by
  /// the current process.
  virtual void Clear() = 0;
  /// Returns the name of the base type of objects constructed by this factory.
  virtual const string BaseName() = 0;
  /// Collects the names of types registered with this factory.
  ///
  /// \param[out] registered registered a set to be modified by this method
  ///                                   so that it contains the names of
  ///                                   concrete types registered with this
  ///                                   factory
  virtual void CollectRegistered(unordered_set<string> &registered) = 0;

  virtual shared_ptr<EnvFactoryBase> CreateEnvFactory() = 0;
};

/// A class to hold all \link Factory \endlink instances that have been created.
class FactoryContainer {
 public:
  typedef vector<FactoryBase *>::iterator iterator;

  /// Adds the specified factory to this container of factories.
  ///
  /// \param factory the factory to add to this container
  static void Add(FactoryBase *factory) {
    if (!initialized_) {
      factories_ = new vector<FactoryBase *>();
      initialized_ = 1;
    }
    factories_->push_back(factory);
  }
  /// Clears this container of factories.
  static void Clear() {
    if (initialized_) {
      for (vector<FactoryBase *>::iterator it = factories_->begin();
           it != factories_->end();
           ++it) {
        (*it)->Clear();
        delete *it;
      }
      delete factories_;
    }
  }

  // Provide two methods to iterate over the FactoryBase instances
  // held by this FactoryContainer.
  static iterator begin() {
    if (factories_ == NULL) {
      cerr << "FactoryContainer::begin: error: no FactoryBase instances!"
           << endl;
    }
    return factories_->begin();
  }
  static iterator end() {
    if (factories_ == NULL) {
      cerr << "FactoryContainer::begin: error: no FactoryBase instances!"
           << endl;
    }
    return factories_->end();
  }

  /// Prints the base typenames for all factories along with a list of all
  /// concrete subtypes those factories can construct, in a human-readable
  /// form, to the specified output stream.
  static void Print(ostream &os) {
    if (!initialized_) {
      return;
    }
    cerr << "Number of factories: " << factories_->size() << "." << endl;
    for (vector<FactoryBase *>::iterator factory_it = factories_->begin();
         factory_it != factories_->end();
         ++factory_it) {
      unordered_set<string> registered;
      (*factory_it)->CollectRegistered(registered);
      os << "Factory<" << (*factory_it)->BaseName() << "> can construct:\n";
      for (unordered_set<string>::const_iterator it = registered.begin();
           it != registered.end();
           ++it) {
        os << "\t" << *it << "\n";
      }
    }
    os.flush();
  }
 private:
  static int initialized_;
  static vector<FactoryBase *> *factories_;
};

/// \class Constructor
///
/// An interface with a single virtual method that constructs a
/// concrete instance of the abstract type <tt>T</tt>.
///
// \tparam T the abstract type that this <tt>Constructor</tt> constructs
template <typename T>
class Constructor {
 public:
  virtual T *NewInstance() const = 0;
};

/// An interface to make it easier to implement Factory-constructible
/// types by implementing both required methods to do nothing.
class FactoryConstructible {
 public:
  virtual void RegisterInitializers(Initializers &initializers) { }
  virtual void Init(const string &arg) { }
};

template <typename T>
class EnvFactory;

/// Factory for dynamically created instance of the specified type.
///
/// \tparam T the type of objects created by this factory, required to have
///           a method with the signature <tt>Init(const string &arg)</tt>
template <typename T>
class Factory : public FactoryBase {
 public:
  /// Constructs a new factory
  Factory() { }

  virtual shared_ptr<EnvFactoryBase> CreateEnvFactory() {
    return shared_ptr<EnvFactoryBase>(new EnvFactory<T>());
  }

  /// Clears this factory of all (possibly static) data.
  /// \p
  /// Note that invoking this method will prevent the factory from functioning!
  /// It should only be invoked when the factory is no longer needed by
  /// the current process.
  virtual void Clear() {
    ClearStatic();
  }

  /// Dynamically creates an object, whose type and initialization are
  /// contained in a specification string, the tokens of which are
  /// given by the specified \link StreamTokenizer\endlink.  A
  /// specification string has the form
  /// \code
  /// Typename(member1(init1), member2(init2), ...)
  /// \endcode
  /// where the type of a member can be
  /// <ul><li>a primitive (a <tt>string</tt>, <tt>double</tt>,
  ///         <tt>int</tt> or <tt>bool</tt>),
  ///     <li>a \link Factory\endlink-constructible type,
  ///     <li>a vector of primtives or
  ///     <li>a vector of types constructible by the same Factory.
  /// </ul>
  /// In the case of members that are vectors, the init string can be
  /// a comma-separated list of of initializers for its elements.  For
  /// example, the class ExampleFeatureExtractor class has three members
  /// that are registered to be initialized by \link
  /// Factory\endlink\<\link FeatureExtractor\endlink\> (via the \link
  /// ExampleFeatureExtractor::RegisterInitializers \endlink method):
  /// <ul><li>a member named <tt>arg</tt> of type <tt>string</tt>,
  ///     <li>a member named <tt>strvec</tt> of type <tt>vector\<string\></tt>
  ///         and
  ///     <li>a member named <tt>b</tt> of type <tt>bool</tt>.
  /// </ul>
  /// None of these is a &ldquo;required&rdquo; member, meaning they act
  /// like optional arguments to a constructor.  The following are all
  /// legal specification strings for constructing instances of
  /// ExampleFeatureExtractor:
  /// \code
  /// ExampleFeatureExtractor(arg("foo"), strvec({"foo", "bar", "baz"}), b(true))
  /// ExampleFeatureExtractor(strvec({"foo", "bar", "baz",}))
  /// ExampleFeatureExtractor(b(false))
  /// ExampleFeatureExtractor()
  /// \endcode
  /// Crucially, note how a vector can have an optional comma at the
  /// end of its list (the second example), and how a boolean may be
  /// initialized either with one of the two <i>reserved words</i>
  /// <tt>true</tt> or <tt>false</tt>, as in C and C++. Finally, unlike
  /// parameter lists to C++ constructors, since our members are always
  /// named, the grammar allows them to appear in any order, making the
  /// following two specification strings equivalent:
  /// \code
  /// ExampleFeatureExtractor(arg("foo"), strvec("foo", "bar", "baz"))
  /// ExampleFeatureExtractor(strvec("foo", "bar", "baz"), arg("foo"))
  /// \endcode
  ///
  /// More formally, the specification string must conform to the following
  /// grammar:
  /// <table border=0>
  /// <tr>
  ///   <td><tt>\<spec\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<type\> '(' \<member_init_list\> ')'</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<type\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>name of type constructible by a Factory</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<member_init_list\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<member_init\> [ ',' \<member_init\> ]* [',']</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<member_init\></tt>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<primitive_init\> | \<factory_init\> |
  ///           \<primitive_vector_init\> | \<factory_vector_init\> </tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<primitive_init\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<member_name\> '(' \<literal\> ')'</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<member_name\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>the name of the member to be initialized, as specified by
  ///       <tt>\<type\>&rsquo;s</tt> <tt>RegisterInitializers</tt> method</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<literal\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<string_literal\> | \<double_literal\> |
  ///           \<int_literal\> | \<bool_literal\></tt></td>
  /// </tr>
  /// <tr valign=top>
  ///   <td><tt>\<string_literal\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>a C++ string literal (a string of characters surrounded by
  ///       double quotes); double quotes and backslashes may be
  ///       escaped inside a string literal with a backslash; other
  ///       escape sequences, such as <tt>\\t</tt> for the tab
  ///       character, are currently not recognized</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<double_literal\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>a string that can be parsed by <tt>atof</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<int_literal\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td>a string that can be parsed by <tt>atoi</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<bool_literal\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>true | false</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<primitive_vector_init></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<member_name\> '(' \<literal_list\> ')'</tt></td>
  /// </tr>
  /// <tr>
  ///   <td valign=top><tt>\<literal_list\></tt></td>
  ///   <td valign=top><tt>::=</tt></td>
  ///   <td><tt>'{' \<string_literal\> [ ',' \<string_literal\> ]* [','] |<br>
  ///           \<double_literal\> [ ',' \<double_literal\> ]* [','] |<br>
  ///           \<int_literal\> [ ',' \<int_literal\> ]* [','] |<br>
  ///           \<bool_literal\> [ ',' \<bool_literal\> ]* [','] '}'</tt>
  ///   </td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<factory_init\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<member_name\> '(' \<spec\> ')'</tt></td>
  /// </tr>
  /// <tr>
  ///   <td><tt>\<factory_vector_init\></tt></td>
  ///   <td><tt>::=</tt></td>
  ///   <td><tt>\<member_name\> '(' \<spec_list\> ')'</tt></td>
  /// </tr>
  /// <tr>
  ///   <td valign=top><tt>\<spec_list\></tt></td>
  ///   <td valign=top><tt>::=</tt></td>
  ///   <td><tt>'{' \<spec\> [ ',' \<spec\> ]* [','] '}'</tt><br>
  ///       where every <tt>\<spec\></tt> has a <tt>\<type\></tt>
  ///       constructible by the same Factory (<i>i.e.</i>, all
  ///       <tt>\<type\></tt>&rsquo;s have a common abstract base class)
  ///   </td>
  /// </tr>
  /// </table>
  ///
  /// \param st the stream tokenizer providing tokens according to the
  ///           grammar shown above
  shared_ptr<T> CreateOrDie(StreamTokenizer &st) {
    size_t start = st.tellg();
    StreamTokenizer::TokenType token_type = st.PeekTokenType();
    if (token_type != StreamTokenizer::IDENTIFIER) {
      ostringstream err_ss;
      err_ss << "Factory<" << BaseName() << ">: "
             << "error: expected type specifier token but found "
             << StreamTokenizer::TypeName(token_type);
      throw std::runtime_error(err_ss.str());
    }

    // Read the concrete type of object to be created.
    string type = st.Next();

    // Read the open parenthesis token.
    if (st.Peek() != "(") {
      ostringstream err_ss;
      err_ss << "Factory<" << BaseName() << ">: "
             << "error: expected '(' at stream position "
             << st.PeekTokenStart() << " but found \"" << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    st.Next();

    // Attempt to create an instance of type.
    typename unordered_map<string, const Constructor<T> *>::iterator cons_it =
        cons_table_->find(type);
    if (cons_it == cons_table_->end()) {
      ostringstream err_ss;
      err_ss << "Factory<" << BaseName() << ">: "
             << "error: unknown type: \"" << type << "\"";
      throw std::runtime_error(err_ss.str());
    }
    shared_ptr<T> instance(cons_it->second->NewInstance());

    // Ask new instance to set up member initializers.
    Initializers initializers;
    instance->RegisterInitializers(initializers);

    // Parse initializer list.
    while (st.Peek() != ")") {
      token_type = st.PeekTokenType();
      if (token_type != StreamTokenizer::IDENTIFIER) {
        ostringstream err_ss;
        err_ss << "Factory<" << BaseName() << ">: "
               << "error: expected token of type IDENTIFIER at "
               << "stream position " << st.PeekTokenStart() << " but found "
               << StreamTokenizer::TypeName(token_type) << ": \""
               << st.Peek() << "\"";
        throw std::runtime_error(err_ss.str());
      }
      size_t member_name_start = st.PeekTokenStart();
      string member_name = st.Next();
      typename Initializers::iterator init_it = initializers.find(member_name);
      if (init_it == initializers.end()) {
        ostringstream err_ss;
        err_ss << "Factory<" << BaseName() << ">: "
               << "error: unknown member name \"" << member_name
               << "\" in initializer list for type " << type << " at stream "
               << "position " << member_name_start;
        throw std::runtime_error(err_ss.str());
      }
      MemberInitializer *member_initializer = init_it->second;

      // Read open parenthesis.
      if (st.Peek() != "(") {
        ostringstream err_ss;
        err_ss << "Factory<" << BaseName() << ">: "
               << "error: expected '(' at stream position "
               << st.PeekTokenStart() << " but found \"" << st.Peek() << "\"";
        throw std::runtime_error(err_ss.str());
      }
      st.Next();

      // Initialize member based on following token(s).
      member_initializer->Init(st);

      // Read close parenthesis for current member initializer.
      if (st.Peek() != ")") {
        ostringstream err_ss;
        err_ss << "Factory<" << BaseName() << ">: "
               << "error: expected ')' at stream position "
               << st.PeekTokenStart() << " but found \"" << st.Peek() << "\"";
        throw std::runtime_error(err_ss.str());
      }
      st.Next();

      // Each member initializer must be followed by a comma or the final
      // closing parenthesis.
      if (st.Peek() != ","  && st.Peek() != ")") {
        ostringstream err_ss;
        err_ss << "Factory<" << BaseName() << ">: "
               << "error: expected ',' or ')' at stream position "
               << st.PeekTokenStart() << " but found \"" << st.Peek() << "\"";
        throw std::runtime_error(err_ss.str());
      }
      // Read comma, if present.
      if (st.Peek() == ",") {
        st.Next();
      }
    }

    // Read the close parenthesis token for this factory type specification.
    if (st.Peek() != ")") {
      ostringstream err_ss;
      err_ss << "Factory<" << BaseName() << ">: "
             << "error: expected ')' at stream position "
             << st.PeekTokenStart() << " but found \"" << st.Peek() << "\"";
      throw std::runtime_error(err_ss.str());
    }
    st.Next();

    // Run through all member initializers: if any are required but haven't
    // been invoked, it is an error.
    for (typename Initializers::const_iterator init_it = initializers.begin();
         init_it != initializers.end();
         ++init_it) {
      MemberInitializer *member_initializer = init_it->second;
      if (member_initializer->Required() &&
          member_initializer->Initialized() == 0) {
        ostringstream err_ss;
        err_ss << "Factory<" << BaseName() << ">: "
               << "error: initialization for member with name \""
               << init_it->first << "\" required but not found (current "
               << "stream position: " << st.tellg() << ")";
        throw std::runtime_error(err_ss.str());
      }
    }

    size_t end = st.tellg();
    // Invoke new instance's Init method.
    // TODO(dbikel): Possibly add the unordered_map of initializers as
    // a parameter to the init method.
    string stream_str = st.str();
    //cerr << "Full stream string is: \"" << stream_str << "\"" << endl;
    string init_str = stream_str.substr(start, end - start);
    //cerr << "Init string is: \"" << init_str << "\"" << endl;
    instance->Init(init_str);

    return instance;
  }

  shared_ptr<T> CreateOrDie(const string &spec, const string err_msg) {
    StreamTokenizer st(spec);
    return CreateOrDie(st);
  }


  /// Returns the name of the base type of objects constructed by this factory.
  virtual const string BaseName() { return base_name_; }

  /// Returns whether the specified type has been registered with this
  /// factory (where registration happens typically via the \link
  /// REGISTER_NAMED \endlink macro).
  ///
  /// \param type the type to be tested
  /// \return whether the specified type has been registered with this
  ///         factory
  static bool IsRegistered(const string &type) {
    return initialized_ && cons_table_->find(type) != cons_table_.end();
  }

  /// \copydoc FactoryBase::CollectRegistered
  virtual void CollectRegistered(unordered_set<string> &registered) {
    if (initialized_) {
      for (typename unordered_map<string, const Constructor<T> *>::iterator it =
               cons_table_->begin();
           it != cons_table_->end();
           ++it) {
        registered.insert(it->first);
      }
    }
  }

  /// The method used by the \link REGISTER_NAMED \endlink macro to ensure
  /// that subclasses add themselves to the factory.
  ///
  /// \param type the type to be registered
  /// \param p    the constructor for the specified type
  static const Constructor<T> *Register(const string &type,
                                        const Constructor<T> *p) {
    if (!initialized_) {
      cons_table_ = new unordered_map<string, const Constructor<T> *>();
      initialized_ = 1;
      FactoryContainer::Add(new Factory<T>());
    }
    (*cons_table_)[type] = p;
    return p;
  }

  /// Clears all static data associated with this class.
  /// \p
  /// Note that invoking this method will prevent the factory from functioning!
  /// It should only be invoked when the factory is no longer needed by
  /// the current process.
  static void ClearStatic() {
    if (initialized_) {
      for (typename unordered_map<string, const Constructor<T> *>::iterator it =
               cons_table_->begin();
           it != cons_table_->end();
           ++it) {
        delete it->second;
      }
      delete cons_table_;
      initialized_ = 0;
    }
  }
 private:
  // data members
  /// Initialization flag.
  static int initialized_;
  /// Factory map of prototype objects.
  static unordered_map<string, const Constructor<T> *> *cons_table_;
  static const char *base_name_;
};

template <typename T>
class EnvFactory : public EnvFactoryBase {
 public:
  bool GetVar(const string &varname, shared_ptr<T> *value) const {
    typename unordered_map<string, shared_ptr<T> >::const_iterator it =
        vars_.find(varname);
    if (it == vars_.end()) {
      return false;
    } else {
      *value = it->second;
      return true;
    }
  }

  bool GetVectorVar(const string &varname,
                    vector<shared_ptr<T> > *value) const {
    typename unordered_map<string, vector<shared_ptr<T> > >::const_iterator it =
        vector_vars_.find(varname);
    if (it == vector_vars_.end()) {
      return false;
    } else {
      *value = it->second;
      return true;
    }
  }

  virtual void SetVar(const string &varname, StreamTokenizer &st) {
    vars_[varname] = factory_.CreateOrDie(st);
  }

  virtual void VectorAppend(const string &varname,
                            StreamTokenizer &st) {
    // First, create object from stream.
    shared_ptr<T> obj = factory_.CreateOrDie(st);

    typename unordered_map<string, vector<shared_ptr<T> > >::iterator it =
        vector_vars_.find(varname);

    if (it == vector_vars_.end()) {
      vector<shared_ptr<T> > vec;
      vec.push_back(obj);
      vector_vars_[varname] = vec;
    } else {
      it->second.push_back(obj);
    }
  }

  virtual bool Defined(const string& varname) const {
    return DefinedObject(varname) || DefinedVector(varname);
  }

  virtual bool DefinedObject(const string &varname) const {
    return vars_.find(varname) != vars_.end();
  }

  virtual bool DefinedVector(const string &varname) const {
    return vector_vars_.find(varname) != vector_vars_.end();
  }

  virtual const FactoryBase &GetFactoryBase() const {
    return factory_;
  }

 private:
  Factory<T> factory_;
  unordered_map<string, shared_ptr<T> > vars_;
  unordered_map<string, vector<shared_ptr<T> > > vector_vars_;
};

/// This is a helper macro used only by the <tt>REGISTER</tt> macro.
#define DEFINE_CONS_CLASS(TYPE,NAME,BASE) \
  class NAME ## Constructor : public Constructor<BASE> { \
   public: virtual BASE *NewInstance() const { return new TYPE(); } };

/// This macro registers the concrete subtype \a TYPE with the
/// specified factory for instances of type \a BASE; the \a TYPE is
/// associated with the specified \a NAME. This macro&mdash;or a macro
/// defined using this macro&mdash;should be used in the
/// implementation file for a concrete subclass \a TYPE of the
/// baseclass \a BASE.  Often, \a TYPE and \a NAME may be the exact
/// same string; however, they must be different when \a TYPE contains
/// characters that may not appear in C++ identifiers, such as colons
/// (<i>e.g.</i>, when \a TYPE is the fully-qualified name of an inner
/// class).
#define REGISTER_NAMED(TYPE,NAME,BASE)  \
  DEFINE_CONS_CLASS(TYPE,NAME,BASE) \
  const Constructor<BASE> *NAME ## _my_protoype = \
      Factory<BASE>::Register(string(#NAME), new NAME ## Constructor());

/// Provides the necessary implementation for a factory for the specified
/// <tt>BASE</tt> class type.
#define IMPLEMENT_FACTORY(BASE) \
  template<> int Factory<BASE>::initialized_ = 0; \
  template<> unordered_map<string, const Constructor<BASE> *> * \
    Factory<BASE>::cons_table_ = 0; \
  template<> const char *Factory<BASE>::base_name_ = #BASE;

}  // namespace reranker

#endif
