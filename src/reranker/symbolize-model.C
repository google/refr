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
/// Definition for executable that reads in a model and a symbol table
/// for that model and then writes that model back out to a file again,
/// ensuring that the symbol table is used to decompile any non-symbolic
/// feature uid&rsquo;s in the model.
/// \author dbikel@google.com (Dan Bikel)

#include <string>
#include <cstdlib>
#include <tr1/memory>
#include <vector>

#include "../proto/dataio.h"
#include "model-proto-writer.H"
#include "model-reader.H"
#include "symbol-table.H"

#define PROG_NAME "symbolize-model"

// We use two levels of macros to get the string version of an int constant.
#define XSTR(arg) STR(arg)
#define STR(arg) #arg

using namespace std;
using namespace std::tr1;
using namespace reranker;
using confusion_learning::SymbolMessage;

const char *usage_msg[] = {
  "Usage:\n",
  PROG_NAME " <model file> <symbol file> <output model file>\n",
};

/// \fn usage
/// Emits usage message to standard output.
void usage() {
  int usage_msg_len = sizeof(usage_msg)/sizeof(const char *);
  for (int i = 0; i < usage_msg_len; ++i) {
    cout << usage_msg[i];
  }
  cout.flush();
}

int
main(int argc, char **argv) {
  // Required parameters.
  string model_file;
  string symbol_file;
  string model_output_file;

  if (argc != 4) {
    usage();
    return -1;
  }

  model_file = argv[1];
  symbol_file = argv[2];
  model_output_file = argv[3];

  bool compressed = true;
  bool use_base64 = true;

  // Now, we finally get to the meat of the code for this executable.
  shared_ptr<Symbols> symbols(new LocalSymbolTable());
  if (symbol_file != "") {
    ConfusionProtoIO proto_reader(symbol_file,
                                  ConfusionProtoIO::READ,
                                  compressed, use_base64);
    SymbolMessage symbol_message;
    while (proto_reader.Read(&symbol_message)) {
      symbols->SetIndex(symbol_message.symbol(), symbol_message.index());
    }
    proto_reader.Close();
  }

  ModelReader model_reader(1);
  shared_ptr<Model> model =
      model_reader.Read(model_file, compressed, use_base64);
  model->set_symbols(symbols.get());
  
  // Serialize model.
  Factory<ModelProtoWriter> proto_writer_factory;
  shared_ptr<ModelProtoWriter> model_writer =
      proto_writer_factory.CreateOrDie(model->proto_writer_spec(),
                                       "model proto writer");
  if (model_writer.get() == NULL) {
    return -1;
  }

  cerr << "Writing out model to file \"" << model_output_file << "\"...";
  cerr.flush();
  confusion_learning::ModelMessage model_message;
  model_writer->Write(model.get(), &model_message, false);

  ConfusionProtoIO *proto_writer;
  proto_writer = new ConfusionProtoIO(model_output_file,
                                      ConfusionProtoIO::WRITE,
                                      compressed, use_base64);
  proto_writer->Write(model_message);
  // Write out features.
  bool output_best_epoch = true;
  bool output_key = false;
  model_writer->WriteFeatures(model.get(), 
                              *(proto_writer->outputstream()),
                              output_best_epoch,
                              model->num_training_errors(),
                              output_key);
  delete proto_writer;
  cerr << "done." << endl;  

  TearDown();
  google::protobuf::ShutdownProtobufLibrary();
}
