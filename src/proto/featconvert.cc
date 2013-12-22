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
// Convert multi-file ASR-style data to proto format.
// Author: kbhall@google.com (Keith Hall)

#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "../utils/kdebug.h"
#include "../proto/model.pb.h"

using namespace std;
using confusion_learning::FeatureMessage;
using confusion_learning::FeatureVecMessage;
using confusion_learning::CandidateSetMessage;

int main(int argc, char* argv[]) {
  bool compressed = true;
  bool base64 = true;
  bool output_featmsg = false;
  int option_char;
  string input_file;

  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "RUi:F")) != EOF) {
    switch (option_char) {  
    case 'F':
      output_featmsg = true;
      break;
    case 'U':
      compressed = false;
      break;
    case 'R':
      base64 = false;
      break;
    case 'i':
      input_file = optarg;
      break;
    case '?':
      cerr << "usage: [-R] [-U] [-i file] [-F]" << endl;
      cerr << "-R - raw, not b64 encoded" << endl;
      cerr << "-U - uncompressed" << endl;
      cerr << "-F - output FeatureMessage protos" << endl;
      return -1;
      break;
    }
  }
  // Read each input line - assume it's compressed.
  ConfusionProtoIO* reader;
  if (input_file.empty()) {
    reader = new ConfusionProtoIO("", ConfusionProtoIO::READSTD,
                                  false, true);
  } else {
    reader = new ConfusionProtoIO(input_file, ConfusionProtoIO::READ,
                                  compressed, true);
  }
  bool reader_valid = true;
  int num_proc = 0;
  ConfusionProtoIO encoder;
  while (reader_valid) {
    CandidateSetMessage set;
    reader_valid = reader->Read(&set);
    num_proc++;
    if (reader_valid) {
      CDEBUG(5, "Candidate Set has: " << set.candidate_size() << " candidates");
      for (int candix = 0; candix < set.candidate_size(); ++candix) {
        const FeatureVecMessage& featvec =
          set.candidate(candix).feats();
        CDEBUG(5, "Candidate has: " << featvec.feature_size() << " features");
        for (int featix = 0; featix < featvec.feature_size(); ++featix) {
          cout << featvec.feature(featix).name() << "|";
          if (output_featmsg) {
            FeatureMessage new_msg(featvec.feature(featix));
            new_msg.set_count(1);
            new_msg.set_value(1.0 / static_cast<double>(featix+1));
            string output;
            encoder.EncodeBase64(new_msg, &output);
            cout << output.c_str();
          } else {
            cout << "1" << endl;
          }
        }
      }
    }
  }
  cerr << "Processed " << num_proc << " records" << endl;
  reader->Close();
  delete reader;
  google::protobuf::ShutdownProtobufLibrary();
}
