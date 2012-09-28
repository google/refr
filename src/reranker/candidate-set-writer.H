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
/// Class for writing streams of training or test instances, where each
/// training or test instance is a reranker::CandidateSet object.
/// \author dbikel@google.com (Dan Bikel)

#ifndef RERANKER_CANDIDATE_SET_WRITER_H_
#define RERANKER_CANDIDATE_SET_WRITER_H_

#include <iostream>
#include <tr1/memory>
#include <vector>

#include "candidate-set.H"
#include "candidate-set-proto-writer.H"
#include "../proto/dataio.h"

#define DEFAULT_WRITER_REPORTING_INTERVAL 1000

namespace reranker {

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::tr1::shared_ptr;

/// \class CandidateSetWriter
///
/// A class for writing streams of training or test instances, where each
/// training or test instance is a reranker::CandidateSet object.
class CandidateSetWriter {
 public:
  /// Constructs a new insta
  CandidateSetWriter(long reporting_interval =
                     DEFAULT_WRITER_REPORTING_INTERVAL) :
      max_num_to_write_(-1),
      num_written_(0),
      interval_written_(0),
      reporting_interval_(reporting_interval),
      verbosity_(0) { }
  virtual ~CandidateSetWriter() { }

  void Open(const string &filename,
            bool compressed,
            bool use_base64) {
    if (verbosity_ >= 1) {
      cerr << "CandidateSetWriter: writing to file \"" << filename
           << "\"." << endl;
    }
    bool writing_to_stdout = filename == "-";
    ConfusionProtoIO::Mode mode =
        writing_to_stdout ?
        ConfusionProtoIO::WRITESTD : ConfusionProtoIO::WRITE;
    compressed = writing_to_stdout ? false : compressed;
    writer_ = new ConfusionProtoIO(filename, mode, compressed, use_base64);
  }

  /// Writes a stream of CandidateSet instances to the specified file
  /// or to standard output.
  ///
  /// \param[in] examples   the vector of pointers to CandidateSet
  ///                       instances to write out
  /// \param[in] filename   the filename to which to write; specify
  ///                       <tt>"-"</tt> to write to standard output
  /// \param[in] compressed whether the output stream is compressed
  /// \param[in] use_base64 whether to use base64 encoding
  void Write(vector<shared_ptr<CandidateSet> > &examples,
             const string &filename,
             bool compressed,
             bool use_base64) {
    Open(filename, compressed, use_base64);
    bool writer_valid = true;
    for (vector<shared_ptr<CandidateSet> >::const_iterator it =
             examples.begin();
         writer_valid && it != examples.end();
         ++it) {
      // First, serialize current CandidateSet to a CandidateSetMessage.
      const CandidateSet &candidate_set = *(*it);
      writer_valid = WriteNext(candidate_set);
    }
    Close();
  }

  bool WriteNext(const CandidateSet &candidate_set) {
    if (num_written_ == max_num_to_write_) {
      return false;
    }

    confusion_learning::CandidateSetMessage tmp_msg;
    candidate_set_proto_writer_.Write(candidate_set, &tmp_msg);

    // Now write it out using ConfusionProtoIO instance.
    bool writer_valid = writer_->Write(tmp_msg);
    if (writer_valid) {
      if (verbosity_ >= 3) {
        cerr << "CandidateSetWriter: most recent CandidateSetMessage: "
             << tmp_msg.Utf8DebugString();
      }

      if (verbosity_ >= 2) {
        cerr << "CandidateSetWriter: candidate set " << candidate_set;
      }

      ++num_written_;
      ++interval_written_;

      if (interval_written_ == reporting_interval_) {
        if (verbosity_ >= 1) {
          cerr << "CandidateSetWriter: wrote " << num_written_
               << " candidate sets." << endl;
        }
        interval_written_ = 0;
      }
    }
    return writer_valid;
  }

  void Close() {
    writer_->Close();
    delete writer_;
  }

  /// Resets this writer so that its internal count of the number of
  /// CandidateSet&rsquo;s written goes back to zero.
  void Reset() {
    num_written_ = 0;
    interval_written_ = 0;
  }

  /// Sets the verbosity of this writer (mostly for debugging
  /// purposes).  There are currently four levels:
  /// <table>
  /// <tr><th>Level</th><th>Meaning</th></tr>
  /// <tr><td>0</td>    <td>None</td></tr>
  /// <tr><td>1</td>    <td>Basic output</td></tr>
  /// <tr><td>2</td>    <td>Very verbose output</td></tr>
  /// <tr><td>3</td>    <td>Extremely verbose output</td></tr>
  /// </table>
  ///
  /// \param verbosity the new verbosity level for this writer to have
  void set_verbosity(int verbosity) { verbosity_ = verbosity; }

  void set_max_num_to_write(int max_num_to_write) {
    max_num_to_write_ = max_num_to_write;
  }

 private:
  // data members
  ConfusionProtoIO *writer_;
  CandidateSetProtoWriter candidate_set_proto_writer_;
  int max_num_to_write_;
  long num_written_;
  long interval_written_;
  long reporting_interval_;
  int verbosity_;
};

}  // namespace reranker

#endif
