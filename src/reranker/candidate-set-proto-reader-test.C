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
/// \file candidate-set-proto-reader-test.C
/// Test driver for reader of reranker::CandidateSet messages.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>

#include "candidate.H"
#include "candidate-set.H"
#include "candidate-set-proto-reader.H"
#include "../proto/data.pb.h"
#include "../proto/dataio.h"

using namespace reranker;
using namespace std;

#define DEBUG 1

int main(int argc, char **argv) {
  CandidateSetProtoReader cspr;
  ConfusionProtoIO* reader = NULL;
  bool is_compressed = true;
  bool use_base64 = true;
  if (argc >= 2) {
    reader = new ConfusionProtoIO(argv[1], ConfusionProtoIO::READ,
                                  is_compressed, use_base64);
  } else {
    reader = new ConfusionProtoIO("", ConfusionProtoIO::READSTD,
                                  is_compressed, use_base64);
  }
  bool reader_valid = true;
  while (reader_valid) {
    confusion_learning::CandidateSetMessage tmp_msg;
    reader_valid = reader->Read(&tmp_msg);
    if (DEBUG) {
      if (reader_valid) {
        cout << "Data: " << tmp_msg.Utf8DebugString();
      }
    }
    CandidateSet candidate_set;
    cspr.Read(tmp_msg, 5, candidate_set);
    cout << candidate_set;
    break;
  }
  reader->Close();
  delete reader;
}
