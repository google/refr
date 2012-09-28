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
// Author: kbhall@google.com (Keith Hall)
// Convert Philipp's MT n-best list format to a DataSet protocol buffer.
// Author: kbhall@google.com (Keith Hall)

#include <fstream>
#include <iostream>
#include <string>
#include <getopt.h>
#include "../gzstream/gzstream.h"
#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "../utils/kdebug.h"
#include "ConvertMT.h"

using namespace std;

int main(int argc, char* argv[]) {
  confusion_dataconvert::ConvertMT converter;

  string outfile_name;
  igzstream* feat_stream = NULL;
  igzstream* ref_stream = NULL;
  bool is_compressed = false;
  int option_char;
  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "Cr:f:o:")) != EOF) {
    switch (option_char) {  
    case 'C':
      // Assume compression for input/output is the same.
      is_compressed = true;
      break;
    case 'f':
      feat_stream = new igzstream(optarg);
      if (feat_stream == NULL || !feat_stream->good()) {
        cerr << "Unable to open file: " << optarg << endl;
        return -1;
      }
      break;
    case 'r':
      ref_stream = new igzstream(optarg);
      if (ref_stream == NULL || !ref_stream->good()) {
        cerr << "Unable to open file: " << optarg << endl;
        return -1;
      }
      break;
    case 'o':
      outfile_name = optarg;
      break;
    case '?':
      cerr << "usage: " << argv[0]
           << "-f feature_stream [-r ref_stream] [-o output_file]" << endl;
      return -1;
      break;
    }
  }
  istream* feature_stream;
  if (feat_stream == NULL ) {
    feature_stream = &cin;
  } else {
    feature_stream = feat_stream;
  }

  ConfusionProtoIO* writer = NULL;
  if (outfile_name.empty()) {
    writer = new ConfusionProtoIO("", ConfusionProtoIO::WRITESTD, is_compressed);
  } else {
    writer = new ConfusionProtoIO(outfile_name, ConfusionProtoIO::WRITE, is_compressed);
  }
  confusion_learning::CandidateSetMessage cand_set;
  string prev_key = "";
  // Reference ID
  int ref_id = 0;
  string candidate_key;
  if (!feature_stream->good()) {
    cerr << "Feature file is invalid" << endl;
  }
  if (feature_stream->eof()) {
    cerr << "Input is empty" << endl;
  }
  while (feature_stream->good() && !feature_stream->eof()) {
    string input_data;
    getline(*feature_stream, input_data);
    CDEBUG(5, "Read input data: " << input_data);
    if (input_data.empty()) {
      break;
    }
    confusion_learning::CandidateMessage tmp_msg;
    if (!converter.ConvertCandidate(input_data, &tmp_msg, &candidate_key)) {
      continue;
    }
    if (candidate_key != prev_key) {
      prev_key = candidate_key;
      if (cand_set.candidate_size() > 0) {
        // Get the refernce string for this candidate set.
        if (!ref_stream) {
          cerr << "Reference shorter than feature file" << endl;
          return -1;
        }
        string ref_string;
        getline(*ref_stream, ref_string);
        cand_set.set_reference_string(ref_string);
        cand_set.set_source_key(candidate_key);
        int candidate_id = atoi(candidate_key.c_str());
        if (ref_id != candidate_id) {
          cerr << "Refernce index: " << ref_id << " candidate set key: "
               << candidate_id << endl;
        }
        CDEBUG(5, "Writing candidate set");
        if (!writer->Write(cand_set)) {
          return -1;
        }
      }
      ref_id++;
      cand_set.Clear();
    } else {
      CDEBUG(5, "Adding new example to set");
    }
    cand_set.add_candidate()->CopyFrom(tmp_msg);
  }
  if (cand_set.candidate_size() > 0) {
    // Get the refernce string for this candidate set.
    if (!ref_stream) {
      cerr << "Reference shorter than feature file";
      return -1;
    }
    string ref_string;
    getline(*ref_stream, ref_string);
    cand_set.set_reference_string(ref_string);
    cand_set.set_source_key(candidate_key);
    CDEBUG(5, "Writing final candidate set");
    if (!writer->Write(cand_set)) {
      return -1;
    }
  }
  writer->Close();
  delete writer;
  delete feat_stream;
  delete ref_stream;
  google::protobuf::ShutdownProtobufLibrary();
}
