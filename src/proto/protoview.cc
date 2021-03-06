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
// Tool to view protocol buffer files generated by the ReFr toolkit.
// Author: kbhall@google.com (Keith Hall)
//
// Example usage:
//  protoview -i training_data_file
//  protoview -M -i model_file
//  protoview -S -i symbol_table_file
//  protoview -F -i feature_message_file
//


#include <fstream>
#include <iostream>
#include <string>
#include <getopt.h>
#include <math.h>
#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "../utils/kdebug.h"
#include "../proto/model.pb.h"

using namespace std;

int main(int argc, char* argv[]) {
  bool compressed = true;
  bool base64 = true;
  int option_char;
  bool decode_model = false;
  bool decode_features = false;
  bool decode_symbols = false;
  string input_file;

  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "RUi:MFS")) != EOF) {
    switch (option_char) {  
    case 'S':
      decode_symbols = true;
      break;
    case 'M':
      decode_model = true;
      break;
    case 'F':
      decode_features = true;
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
      cerr << "usage: [-R] [-U] [-M] [-F] [-S] [-i file]" << endl;
      cerr << "-R - view raw encoded file (non-base64)" << endl;
      cerr << "-U - uncompressed input file" << endl;
      cerr << "-i - if empty, uses stdin" << endl;
      cerr << "-M - output model messages" << endl;
      cerr << "-F - output feature messages" << endl;
      cerr << "-S - output symbol messages" << endl;
      return -1;
      break;
    }
  }
  ConfusionProtoIO* reader;
  if (!input_file.empty()) {
    reader = new ConfusionProtoIO(input_file, ConfusionProtoIO::READ,
                                  compressed, base64);
  } else {
    reader = new ConfusionProtoIO("", ConfusionProtoIO::READSTD,
                                  false, base64);
  }
  bool reader_valid = true;
  bool first_message = true;
  while (reader_valid) {
    google::protobuf::Message* tmp_msg;
    if (decode_model) {
      if (first_message) {
        tmp_msg = new confusion_learning::ModelMessage;
        first_message = false;
      } else {
        tmp_msg = new confusion_learning::FeatureMessage;
      }
    } else if (decode_features) {
      tmp_msg = new confusion_learning::FeatureMessage;
    } else if (decode_symbols) {
      tmp_msg = new confusion_learning::SymbolMessage;
    } else {
      // Default is to decode candidate-set messages
      tmp_msg = new confusion_learning::CandidateSetMessage;
    }
    if (tmp_msg == NULL) {
      continue;
    }
    reader_valid = reader->Read(tmp_msg);
    if (reader_valid) {
      cout << "Data: " << tmp_msg->Utf8DebugString();
    }
    delete tmp_msg;
  }
  reader->Close();
  delete reader;
  google::protobuf::ShutdownProtobufLibrary();
}
