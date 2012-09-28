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
/// Combine the model symbols.
/// \author kbhall@google.com (Keith Hall)

#include <cstdio>
#include <iostream>
#include <string>
#include <tr1/memory>
#include "../proto/dataio.h"
#include "../proto/model.pb.h"
#include "../utils/kdebug.h"

using namespace std;
using confusion_learning::SymbolMessage;;


int main(int argc, char* argv[]) {
  int option_char;
  bool output_compressed = true;
  string output_name;

  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "Uo:")) != EOF) {
    switch (option_char) {  
    case 'U':
      output_compressed = false;
      break;
    case 'o':
      output_name = optarg;
      break;
    case '?':
      cerr << "usage: " << argv[0]
           << " [-U] [-o <output file>]"
           << endl;
      cerr << "-U - output to uncompressed file" << endl;
      cerr << "-o - output to filename (otherwise to uncompressed stdout" << endl;
      return -1;
      break;
    }
  }

  // Process each of the input records.  This reducer assumes that the input is
  // a stream of feature strings;
  ConfusionProtoIO reader;
  ConfusionProtoIO* writer;
  if (output_name.empty()) {
    writer = new ConfusionProtoIO("", ConfusionProtoIO::WRITESTD, false, true);
  } else {
    writer = new ConfusionProtoIO(output_name, ConfusionProtoIO::WRITE,
                                  output_compressed, true);
  }
  int feature_id = 0;
  SymbolMessage sym_msg;
  while (cin) {
    // Process input.
    string input_data;
    getline(cin, input_data);
    if (input_data.empty()) {
      break;
    }
    size_t delim_pos = input_data.find('\t');
    if (delim_pos != string::npos) {
      input_data.erase(delim_pos);
    }
    sym_msg.set_symbol(input_data);
    sym_msg.set_index(feature_id);
    feature_id++;
    writer->Write(sym_msg);
  }
  cerr << "Wrote " << feature_id << " feature messages to file: ";
  if (output_name.empty()) {
    cerr << " STDOUT " << endl;
  } else {
    cerr << output_name.c_str() << " " << endl;
  }
  delete writer;
  return 0;
}
