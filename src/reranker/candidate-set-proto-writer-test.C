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
/// \file candidate-set-proto-writer-test.C
/// Test driver for writer of reranker::CandidateSet messages.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>

#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "candidate.H"
#include "candidate-set.H"
#include "candidate-set-proto-reader.H"
#include "candidate-set-proto-writer.H"

using namespace reranker;
using namespace std;

int main(int argc, char **argv) {
  CandidateSetProtoReader cspr;
  bool is_compressed = true;
  bool use_base64 = true;
  ConfusionProtoIO* reader = NULL;
  if (argc >= 2) {
    reader = new ConfusionProtoIO(argv[1], ConfusionProtoIO::READ,
                                  is_compressed, use_base64);
  } else {
    reader = new ConfusionProtoIO("", ConfusionProtoIO::READSTD,
                                  is_compressed, use_base64);
  }
  bool reader_valid = true;
  CandidateSet candidate_set;
  while (reader_valid) {
    confusion_learning::CandidateSetMessage tmp_msg;
    reader_valid = reader->Read(&tmp_msg);
    cspr.Read(tmp_msg, 1, candidate_set);

    cout << "Here's the one candidate set:" << endl;
    cout << candidate_set;
    break; // read a single CandidateSet
  }
  reader->Close();
  delete reader;

  CandidateSetProtoWriter cspw;
  confusion_learning::CandidateSetMessage cs_msg;
  cspw.Write(candidate_set, &cs_msg);

  CandidateSet re_read_candidate_set;
  cspr.Read(cs_msg, re_read_candidate_set);
  cout << "Successfully wrote out and then re-read candidate set. Here it is:"
       << endl
       << re_read_candidate_set
       << endl
       << "Have a nice day!" << endl;
}
