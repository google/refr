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
/// Provides an interface and some implementations for iterating over
/// \link reranker::CandidateSet CandidateSet \endlink instances.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_SET_ITERATOR_H_
#define RERANKER_CANDIDATE_SET_ITERATOR_H_

#include <memory>
#include <vector>

#include "candidate-set.H"
#include "candidate-set-reader.H"
#include "executive-feature-extractor.H"

namespace reranker {

using std::vector;
using std::shared_ptr;

/// An interface specifying iteration over \link CandidateSet \endlink
/// instances, using Java-style semantics (sorry, die-hard C++ iterator fans).
/// Unlike Java or C++ iterators, implementations of this interface are
/// always assumed to have some way of returning to the beginning of their
/// underlying collection of \link CandidateSet \endlink instances; this is
/// required by the \link Reset \endlink method.
class CandidateSetIterator {
 public:
  virtual ~CandidateSetIterator() { }
  /// Returns whether this iterator contains another \link
  /// CandidateSet \endlink.
  virtual bool HasNext() const = 0;
  /// Returns the next \link CandidateSet \endlink.
  virtual CandidateSet &Next() = 0;
  /// Resets this iterator back to the beginning of its backing collection.
  virtual void Reset() = 0;
};

/// An implementation of the \link CandidateSetIterator \endlink
/// interface that is backed by an arbitrary C++ collection of
/// pointers to \link CandidateSet\endlink&rsquo;s, where the
/// collection&rsquo;s iterators implement the
/// <tt>ForwardIterator</tt> concept.
/// \attention
/// It is a requirement of instances of this class that the backing
/// collections do not get destroyed during the lifetime of instances
/// of this class.  If they do, then the behavior of this candidate
/// set iterator is undefined.
///
/// \tparam Collection the type of collection that this candidate set
///                    iterator will wrap; the <tt>Collection</tt> must
///                    have <tt>begin</tt> and <tt>end</tt> methods,
///                    each returning an iterator of type
///                    <tt>Collection::iterator</tt> that implements
///                    the <tt>ForwardIterator</tt> concept
template <typename Collection>
class CollectionCandidateSetIterator : public CandidateSetIterator {
 public:
  CollectionCandidateSetIterator(Collection &collection) :
      collection_(collection) {
    it_ = collection_.begin();
  }

  virtual bool HasNext() const { return it_ != collection_.end(); }
  virtual CandidateSet &Next() { return **it_++; }
  virtual void Reset() { it_ = collection_.begin(); }
 private:
  /// An iterator pointing to an element of the underlying collection.
  typename Collection::iterator it_;
  /// The backing collection.
  Collection &collection_;
};


/// An implementation of the \link CandidateSetIterator \endlink
/// interface that iterates over \link CandidateSet \endlink instances
/// that have been serialized to <tt>CandidateSetMessage</tt> protocol
/// buffer messages in multiple files.  The sequence of files containing
/// serialized \link CandidateSet \endlink instances are specified
/// at construction time.
class MultiFileCandidateSetIterator : public CandidateSetIterator {
 public:
  /// Constructs a new instance that iterates over the \link CandidateSet
  /// \endlink instances serialized in the specified sequence of files.
  ///
  /// \param files              the sequence of files containing serialized
  ///                           \link CandidateSet \endlink instances
  /// \param efe                a pointer to an \link ExecutiveFeatureExtractor
  ///                           \endlink instance; if
  ///                           non-<tt>NULL</tt>, this iterator will
  ///                           invoke the \link
  ///                           ExecutiveFeatureExtractor::Extract
  ///                           \endlink on each \link CandidateSet
  ///                           \endlink instance returned by this
  ///                           iterator&rsquo;s \link Next \endlink
  ///                           method
  /// \param max_examples       the maximum number of examples to be read
  ///                           from each file specified by the
  ///                           <tt>files</tt> parameter (as per the
  ///                           first parameter of the \link
  ///                           CandidateSetReader::CandidateSetReader(int,int,long)
  ///                           \endlink constructor)
  /// \param max_candidates     the maximum number of candidates to be
  ///                           read for each de-serialized \link
  ///                           CandidateSet \endlink instance (as per
  ///                           the second parameter of the \link
  ///                           CandidateSetReader::CandidateSetReader(int,int,long)
  ///                           \endlink constructor)
  /// \param reporting_interval the number of \link CandidateSet
  ///                           \endlink instances read after which a
  ///                           message will be issued to <tt>cerr</tt>
  /// \param verbosity          the verbosity level of this instance and its
  ///                           underlying objects (see \link
  ///                           CandidateSetReader::set_verbosity\endlink)
  /// \param compressed         whether the files containing serialized \link
  ///                           CandidateSet \endlink instances are compressed
  /// \param use_base64         whether the files containing serialized \link
  ///                           CandidateSet \endlink instances use
  ///                           base64 encoding
  MultiFileCandidateSetIterator(vector<string> files,
                                shared_ptr<const ExecutiveFeatureExtractor> efe,
                                int max_examples,
                                int max_candidates,
                                int reporting_interval,
                                int verbosity,
                                bool compressed,
                                bool use_base64) :
      files_(files), efe_(efe),
      compressed_(compressed), use_base64_(use_base64),
      reader_valid_(false), file_open_(false),
      csr_(max_examples, max_candidates, reporting_interval),
      verbosity_(verbosity),
      next_candidate_set_(),
      prev_candidate_set_() {
    file_it_ = files_.begin();
    csr_.set_verbosity(0);
    Reset();
    csr_.set_verbosity(verbosity);
  }

  /// \copydoc CandidateSetIterator::HasNext
  virtual bool HasNext() const {
    return next_candidate_set_.get() != NULL;
  }

  /// \copydoc CandidateSetIterator::Next
  virtual CandidateSet &Next() {
    prev_candidate_set_ = next_candidate_set_;
    next_candidate_set_ = shared_ptr<CandidateSet>();
    ReadNext();
    return *prev_candidate_set_;
  }

  const string curr_file() const {
    return file_open_ ? *file_it_ : string("");
  }

  /// Resets this multi-file candidate set iterator so that the next
  /// \link CandidateSet \endlink retrieved will be the first serialized
  /// instance in the first of the sequence of files with which this
  /// \link MultiFileCandidateSetIterator \endlink was constructed.
  virtual void Reset() {
    if (file_open_) {
      csr_.Close();
      file_open_ = false;
    }
    file_it_ = files_.begin();
    if (file_it_ != files_.end()) {
      csr_.Open(*file_it_, compressed_, use_base64_);
      file_open_ = true;
    }
    next_candidate_set_ = shared_ptr<CandidateSet>();
    if (efe_.get() != NULL) {
      efe_->Reset();
    }
    ReadNext();
  }
 private:
  void ReadNext() {
    reader_valid_ = false;
    while (file_it_ != files_.end() && next_candidate_set_.get() == NULL) {
      next_candidate_set_ = csr_.ReadNext(reader_valid_);
      if (!reader_valid_ || next_candidate_set_.get() == NULL) {
        if (verbosity_ >= 1) {
          if (csr_.num_read() == 0) {
            cerr << "Warning: could not read any training examples from file \""
                 << *file_it_ << "\"." << endl;
          }
        }
        // Reached eof.  Try to open next file.
        csr_.Close();
        file_open_ = false;
        ++file_it_;
        if (file_it_ != files_.end()) {
          csr_.Open(*file_it_, compressed_, use_base64_);
          file_open_ = true;
        }
      }
    }
    if (efe_.get() != NULL && next_candidate_set_.get() != NULL) {
      efe_->Extract(*next_candidate_set_);
    }
  }

  // data members
  vector<string> files_;
  shared_ptr<const ExecutiveFeatureExtractor> efe_;
  bool compressed_;
  bool use_base64_;
  bool reader_valid_;
  bool file_open_;
  CandidateSetReader csr_;
  int verbosity_;
  vector<string>::const_iterator file_it_;
  shared_ptr<CandidateSet> next_candidate_set_;
  shared_ptr<CandidateSet> prev_candidate_set_;
};

}  // namespace reranker

#endif
