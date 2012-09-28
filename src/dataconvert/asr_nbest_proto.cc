// Copyright 2012, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//     * Neither the name of Google Inc. nor the names of its contributors may
//     be used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,           DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//
// ReFr - Reranking Framework
//
// Convert multi-file ASR-style data to proto format.
// Author: kbhall@google.com (Keith Hall)

#include <fstream>
#include <iostream>
#include <string>
#include <getopt.h>
#include "../gzstream/gzstream.h"
#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "../utils/kdebug.h"
#include "ConvertASR.h"

using namespace std;

int main(int argc, char* argv[]) {
  // b - base score file
  // l - loss file
  // f - feature file (there can be many of these)
  // t - raw text file
  // r - reference file
  // i - input proto to merge with.
  // U - don't compress output
  // B - Sentence level bleu
  // A - References are in ASR format (three fields before the reference)
  bool out_compressed = true;
  bool base64 = true;
  int option_char;
  string outfile_name;
  string inputfile_name;
  ConvertASR converter;

  igzstream* new_stream = NULL;
  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "ABUb:l:f:t:o:r:i:D:")) != EOF) {
    switch (option_char) {  
    case 'D':
      SETDEBUG(atoi(optarg));
      break;
    case 'A':
      converter.set_asrref();
      break;
    case 'B':
      converter.set_bleuloss();
      break;
    case 'U':
      // Assume compression for input/output is the same.
      out_compressed = false;
      break;
    case 'b':
      new_stream = new igzstream(optarg);
      if (new_stream == NULL || !new_stream->good()) {
        cerr << "Unable to open file: " << optarg << endl;
        return -1;
      }
      converter.set_baselinedata(new_stream);
      break;
    case 'l':
      new_stream = new igzstream(optarg);
      if (new_stream == NULL || !new_stream->good()) {
        cerr << "Unable to open file: " << optarg << endl;
        return -1;
      }
      converter.set_lossdata(new_stream);
      break;
    case 't':
      new_stream = new igzstream(optarg);
      if (new_stream == NULL || !new_stream->good()) {
        cerr << "Unable to open file: " << optarg << endl;
        return -1;
      }
      converter.set_rawtextdata(new_stream);
      break;
    case 'r':
      new_stream = new igzstream(optarg);
      if (new_stream == NULL || !new_stream->good()) {
        cerr << "Unable to open file: " << optarg << endl;
        return -1;
      }
      converter.set_refdata(new_stream);
      break;
    case 'f':
      {
        string optstring(optarg);
        size_t splitpos = optstring.find(':');
        if (splitpos == string::npos) {
          cerr << "Feature file argument not in feature_type:filename format"
               << endl;
          return -1;
        }
        string filename = optstring.substr(splitpos + 1);
        optstring.erase(splitpos);
        cerr << "Processing feature type: " << optstring
             << " filename: " << filename << endl;
        new_stream = new igzstream(filename.c_str());
        if (new_stream == NULL || !new_stream->good()) {
          cerr << "Unable to open " << optstring
               << " file: " << filename << endl;
          return -1;
        }
        converter.add_featdata(optstring, new_stream);
      }
      break;
    case 'o':
      outfile_name = optarg;
      break;
    case 'i':
      inputfile_name = optarg;
      break;
    case '?':
      cerr << "usage: " << argv[0]
           << " -l loss_file [-b baseline_file] [-t rawtext_file] [-r reference_file]"
           << "[-f feat_type:feat_file] [-f ...] [-o output_file] [-i input_proto_to_merge]"
           << "[-U] [-B] [-A] [-D]" << endl;
      cerr << "-B - Sentencen level BLEU loss in loss file." << endl;
      cerr << "-U - Write raw (non-compressed) output files." << endl;
      cerr << "-A - References are in ASR format (three fields before the reference)." << endl;
      cerr << "-D - Set the debug level." << endl;
      return -1;
      break;
    }
  }

  ConfusionProtoIO* reader = NULL;
  if (!inputfile_name.empty()) {
    reader = new ConfusionProtoIO(inputfile_name, ConfusionProtoIO::READ,
                                  out_compressed, base64);
  }
  ConfusionProtoIO* writer = NULL;
  if (outfile_name.empty()) {
    writer = new ConfusionProtoIO("", ConfusionProtoIO::WRITESTD,
                                  out_compressed, base64);
  } else {
    writer = new ConfusionProtoIO(outfile_name, ConfusionProtoIO::WRITE,
                                  out_compressed, base64);
  }

  CDEBUG(5, "Processing data");
  int index = 0;
  confusion_learning::CandidateSetMessage cand_set;
  if (reader != NULL) {
    reader->Read(&cand_set);
  }
  bool error = false;
  while (converter.AddData(&cand_set)) {
    index++;
    if ((index % 100) == 0) {
      cerr << "Processed " << index << " examples" << endl;
    }
    CDEBUG(5, "Processed exmaple: " << index);
    if (cand_set.candidate_size() > 0) {
      if (!writer->Write(cand_set)) {
        error = true;
      }
    }
    cand_set.Clear();
    if (reader != NULL) {
      reader->Read(&cand_set);
    }
  }
  CDEBUG(5, "Finished processing data, number exmaples: " << index);
  if (cand_set.candidate_size() > 0) {
    if (!writer->Write(cand_set)) {
      error = true;
    }
  }
  CDEBUG(5, "Cleaning up");
  writer->Close();
  delete writer;
  google::protobuf::ShutdownProtobufLibrary();
  if (error) {
    cerr << "Possible error during processing" << endl;
  }
  return 0;
}
