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
/// Class for reading streams of training or test instances, where each
/// training or test instance is a reranker::CandidateSet object.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_SET_READER_H_
#define RERANKER_CANDIDATE_SET_READER_H_

#include <iostream>
#include <tr1/memory>
#include <vector>

#include "candidate-set.H"
#include "candidate-set-proto-reader.H"
#include "../proto/dataio.h"

#define DEFAULT_READER_REPORTING_INTERVAL 100

namespace reranker {

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::tr1::shared_ptr;

/// \class CandidateSetReader
///
/// A class for reading streams of training or test instances, where each
/// training or test instance is a reranker::CandidateSet object.
class CandidateSetReader {
 public:
  /// Constructs a new instance.
  CandidateSetReader(long reporting_interval =
                     DEFAULT_READER_REPORTING_INTERVAL) :
      max_num_to_read_(-1),
      max_candidates_per_set_(-1),
      num_read_(0),
      num_read_from_file_(0),
      total_num_read_(0),
      interval_read_(0),
      reporting_interval_(reporting_interval) { }
  CandidateSetReader(int max_num_to_read,
                     int max_candidates_per_set,
                     long reporting_interval =
                     DEFAULT_READER_REPORTING_INTERVAL) :
      max_num_to_read_(max_num_to_read),
      max_candidates_per_set_(max_candidates_per_set),
      num_read_(0),
      num_read_from_file_(0),
      total_num_read_(0),
      interval_read_(0),
      reporting_interval_(reporting_interval) { }
  virtual ~CandidateSetReader() { }

  void Open(const string &filename, bool compressed, bool use_base64,
            bool reset_counters = true) {
    if (reset_counters) {
      Reset();
    }
    if (verbosity_ >= 1) {
      cerr << "CandidateSetReader: reading from file \"" << filename
           << "\"." << endl;
    }
    bool reading_from_stdin = filename == "-";
    ConfusionProtoIO::Mode mode =
        reading_from_stdin ? ConfusionProtoIO::READSTD : ConfusionProtoIO::READ;
    compressed = reading_from_stdin ? false : compressed;
    reader_ = new ConfusionProtoIO(filename, mode, compressed, use_base64);
    filename_ = filename;
    num_read_from_file_ = 0;
  }

  /// Reads a stream of CandidateSet instances from the specified file
  /// or from standard input.
  ///
  /// \param[in] filename           the filename from which to read; specify
  ///                               <tt>"-"</tt> to read from standard input
  /// \param[in] compressed         whether the input stream is compressed
  /// \param[in] use_base64         whether to do base64 decoding
  /// \param[in] reset_counters     whether to invoke \link Reset \endlink
  ///                               prior to opening the specified file
  /// \param[out] examples          the vector of pointers to CandidateSet
  ///                               instances to which to append (pointers to)
  ///                               newly-read and -constructed CandidateSet
  ///                               instances
  void Read(const string &filename,
            bool compressed,
            bool use_base64,
            bool reset_counters,
            vector<shared_ptr<CandidateSet> > &examples) {
    Open(filename, compressed, use_base64, reset_counters);

    bool reader_valid = true;
    while (reader_valid) {
      shared_ptr<CandidateSet> candidate_set = ReadNext(reader_valid);
      if (candidate_set == NULL) {
        break;
      }
      examples.push_back(candidate_set);
    }
    Close();
  }

  shared_ptr<CandidateSet> ReadNext(bool &reader_valid) {
    if (num_read_ == max_num_to_read_) {
      return shared_ptr<CandidateSet>();      
    }
    // First, de-serialize next CandidateSetMessage from stream.
    confusion_learning::CandidateSetMessage tmp_msg;
    reader_valid = reader_->Read(&tmp_msg);
    if (reader_valid) {
      if (verbosity_ >= 3) {
        cerr << "CandidateSetReader: most recent CandidateSetMessage: "
             << tmp_msg.Utf8DebugString();
      }
    } else {
      return shared_ptr<CandidateSet>();
    }

    shared_ptr<CandidateSet> candidate_set(new CandidateSet());
    candidate_set_proto_reader_.Read(tmp_msg, max_candidates_per_set_,
                                     *candidate_set);
    
    if (verbosity_ >= 2) {
      cerr << "CandidateSetReader: candidate set prior to "
           << "feature compilation:" << endl << *(candidate_set);
    }

    ++num_read_;
    ++num_read_from_file_;
    ++total_num_read_;
    ++interval_read_;

    if (interval_read_ == reporting_interval_) {
      if (verbosity_ >= 1) {
        cerr << "CandidateSetReader: read " << num_read_
             << " candidate sets." << endl;
      }
      interval_read_ = 0;
    }
    return candidate_set;
  }

  void Close() {
    if (verbosity_ >= 1) {
      cerr << "CandidateSetReader: read " << num_read_from_file_
           << " candidate sets from file \"" << filename_ << "\". Closing file."
           << endl;
    }
    reader_->Close();
    delete reader_;
  }

  /// Resets this reader so that its internal count of the number of
  /// CandidateSet&rsquo;s read goes back to zero.
  void Reset() {
    num_read_ = 0;
    interval_read_ = 0;
  }

  /// Invokes \link CandidateSetProtoReader::ClearStrings \endlink on the
  /// internal \link CandidateSetProtoReader \endlink instance.
  void ClearStrings() {
    candidate_set_proto_reader_.ClearStrings();
  }

  // accessors

  /// Returns the number of \link CandidateSet \endlink instances read
  /// since the last invocation of the \link Reset \endlink method. 
  long num_read() { return num_read_; }

  /// Returns the total number of \link CandidateSet \endlink instances read.
  long total_num_read() { return total_num_read_; }

  // mutators

  /// Sets the verbosity of this reader (mostly for debugging
  /// purposes).  There are currently four levels:
  /// <table>
  /// <tr><th>Level</th><th>Meaning</th></tr>
  /// <tr><td>0</td>    <td>None</td></tr>
  /// <tr><td>1</td>    <td>Basic output</td></tr>
  /// <tr><td>2</td>    <td>Very verbose output</td></tr>
  /// <tr><td>3</td>    <td>Extremely verbose output</td></tr>
  /// </table>
  ///
  /// \param verbosity the new verbosity level for this reader to have
  void set_verbosity(int verbosity) { verbosity_ = verbosity; }

 private:
  // data members
  ConfusionProtoIO *reader_;
  CandidateSetProtoReader candidate_set_proto_reader_;
  string filename_;
  int max_num_to_read_;
  int max_candidates_per_set_;
  long num_read_;
  long num_read_from_file_;
  long total_num_read_;
  long interval_read_;
  long reporting_interval_;
  int verbosity_;
};

}  // namespace reranker

#endif
