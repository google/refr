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
/// Definition for executable that reads in reranker::CandidateSet
/// instances from a stream and then runs feature extractors on those
/// instances using an reranker::ExecutiveFeatureExtractor.  (Recall
/// that a reranker::ExecutiveFeatureExtractor is like a regular
/// feature extractor, but wears fancypants.)
/// \author dbikel@google.com (Dan Bikel)

#include <string>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../proto/dataio.h"
#include "candidate-set.H"
#include "candidate-set-iterator.H"
#include "candidate-set-writer.H"
#include "executive-feature-extractor.H"
#include "symbol-table.H"

#define PROG_NAME "extract-features"

#define DEFAULT_MAX_EXAMPLES -1
#define DEFAULT_MAX_CANDIDATES -1
#define DEFAULT_REPORTING_INTERVAL 1000

// We use two levels of macros to get the string version of an int constant.
#define XSTR(arg) STR(arg)
#define STR(arg) #arg

using namespace std;
using namespace reranker;
using confusion_learning::SymbolMessage;

const char *usage_msg[] = {
  "Usage:\n",
  PROG_NAME " [-c|--config <feature extractor config file>]\n",
  "\t-i|--input <candidate set input file>+\n",
  "\t-o|--output <output directory>\n",
  "\t[--input-symbols <input symbol table>]\n",
  "\t[--output-symbols <output symbol table>]\n",
  "\t[-u] [--no-base64] [--compile] [--clear-raw]\n",
  "\t[--max-examples <max num examples>]\n",
  "\t[--max-candidates <max num candidates>]\n",
  "\t[-r <reporting interval>]\n",
  "where\n",
  "\t<feature extractor config file> is the name of a configuration file\n",
  "\t\tto be read by the ExecutiveFeatureExtractor class\n",
  "\t<candidate set input file> is the name of a stream of serialized\n",
  "\t\tCandidateSet instances, or \"-\" for input from standard input\n",
  "\t<output dirctory> is the directory to output each input file after\n",
  "\t\textracting features\n",
  "\t<input symbol table> is an optional input file containing a Symbols\n",
  "\t\tinstance serialized as a sequence of Symbol messages\n",
  "\t<output symbol table> is an optional output file to which a Symbols\n",
  "\t\tinstance will be serialized as a sequence of Symbol messages\n",
  "\t-u specifies that the input files should be uncompressed (compression\n",
  "\t\tis used by default)\n",
  "\t--no-base64 specifies not to use base64 encoding/decoding\n",
  "\t--compile specifies to compile features after each CandidateSet is read\n",
  "\t--clear-raw specified to clear each Candidate of its raw data string\n",
  "\t--max-examples specifies the maximum number of examples to read from\n",
  "\t\tany input file (defaults to " XSTR(DEFAULT_MAX_EXAMPLES) ")\n",
  "\t--max-candidates specifies the maximum number of candidates to read\n",
  "\t\tfor any candidate set (defaults to " XSTR(DEFAULT_MAX_CANDIDATES) ")\n",
  "\t-r specifies the interval at which the CandidateSetReader reports how\n",
  "\t\tmany candidate sets it has read (defaults to "
  XSTR(DEFAULT_REPORTING_INTERVAL) ")\n",
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

bool check_for_required_arg(int argc, int i, string err_msg) {
  if (i + 1 >= argc) {
    cerr << PROG_NAME << ": error: " << err_msg << endl;
    usage();
    return false;
  } else {
    return true;
  }
}

int
main(int argc, char **argv) {
  // Required parameters.
  bool do_feature_extraction = false;
  string feature_extractor_config_file;
  vector<string> input_files;
  bool compressed = true;
  bool use_base64 = true;
  bool compile = false;
  bool clear_raw = false;
  string output_dir;
  string symbol_table_input_file = "";
  string symbol_table_output_file = "";
  int max_examples = DEFAULT_MAX_EXAMPLES;
  int max_candidates = DEFAULT_MAX_CANDIDATES;
  int reporting_interval = DEFAULT_REPORTING_INTERVAL;

  // Process options.  The majority of code in this file is devoted to this.
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-c" || arg == "-config" || arg == "--config") {
      string err_msg =
          string("no feature extractor config file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      do_feature_extraction = true;
      feature_extractor_config_file = argv[++i];
    } else if (arg == "-i" || arg == "-input" || arg == "--input") {
      string err_msg = string("no input files specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      // Keep reading args until next option or until no more args.
      ++i;
      for ( ; i < argc; ++i) {
        if (argv[i][0] == '-') {
          --i;
          break;
        }
        input_files.push_back(argv[i]);
      }
    } else if (arg == "-o" || arg == "-output" || arg == "--output") {
      string err_msg = string("no output directory specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      output_dir = argv[++i];
      // Remove final slash, if present.
      if (output_dir.size() > 0 && output_dir[output_dir.size() - 1] == '/') {
        output_dir = output_dir.substr(0, output_dir.size() - 1);
      }
    } else if (arg == "-input-symbols" || arg == "--input-symbols") {
      string err_msg =
          string("no symbol table input file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      symbol_table_input_file = argv[++i];      
    } else if (arg == "-output-symbols" || arg == "--output-symbols") {
      string err_msg =
          string("no symbol table output file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      symbol_table_output_file = argv[++i];      
    } else if (arg == "-u") {
      compressed = false;
    } else if (arg == "--no-base64") {
      use_base64 = false;
    } else if (arg == "-compile" || arg == "--compile") {
      compile = true;
    } else if (arg == "-clear-raw" || arg == "--clear-raw") {
      clear_raw = true;
    } else if (arg == "-max-examples" || arg == "--max-examples") {
      string err_msg = string("no arg specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      max_examples = atoi(argv[++i]);
    } else if (arg == "-max-candidates" || arg == "--max-candidates") {
      string err_msg = string("no arg specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      max_candidates = atoi(argv[++i]);
    } else if (arg == "-r") {
      string err_msg = string("no arg specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      reporting_interval = atoi(argv[++i]);
    } else if (arg.size() > 0 && arg[0] == '-') {
      cerr << PROG_NAME << ": error: unrecognized option: " << arg << endl;
      usage();
      return -1;
    }
  }

  // Check that user specified required args.
  if (input_files.size() == 0) {
    cerr << PROG_NAME << ": error: no candidate set input files specified"
         << endl;
    usage();
    return -1;
  }

  if (output_dir == "") {
    cerr << PROG_NAME << ": error: no output directory specified" << endl;
    usage();
    return -1;
  }

  // Now, we finally get to the meat of the code for this executable.
  shared_ptr<Symbols> symbols;
  if (symbol_table_input_file != "") {
    ConfusionProtoIO proto_reader(symbol_table_input_file,
                                  ConfusionProtoIO::READ,
                                  compressed, use_base64);
    SymbolMessage symbol_message;
    while (proto_reader.Read(&symbol_message)) {
      symbols->SetIndex(symbol_message.symbol(), symbol_message.index());
    }
    proto_reader.Close();
  } else {
    symbols = shared_ptr<Symbols>(new LocalSymbolTable());
  }

  shared_ptr<ExecutiveFeatureExtractor> efe;
  if (do_feature_extraction) {
    efe =
        ExecutiveFeatureExtractor::InitFromSpec(feature_extractor_config_file);
  }

  int verbosity = 1;
  MultiFileCandidateSetIterator csi(input_files,
                                    efe,
                                    max_examples,
                                    max_candidates,
                                    reporting_interval,
                                    verbosity,
                                    compressed,
                                    use_base64);

  // Set things up for streaming output.
  CandidateSetWriter csw(reporting_interval);
  csw.set_verbosity(1);
  string input_file("");

  while (csi.HasNext()) {
    if (csi.curr_file() != input_file) {
      if (input_file != "") {
        csw.Close();
      }
      input_file = csi.curr_file();
      size_t slash_idx = input_file.find_last_of("/");
      string tail =
          input_file.substr(slash_idx != string::npos ? slash_idx + 1 : 0);
      string output_file = output_dir + "/" + tail;
      csw.Reset();
      csw.Open(output_file, compressed, use_base64);
    }
    CandidateSet &candidate_set = csi.Next();
    if (compile) {
      candidate_set.CompileFeatures(symbols.get());
    }
    if (clear_raw) {
      candidate_set.ClearRawData();
    }
    bool success = csw.WriteNext(candidate_set);
    if (!success) {
      cerr << "Uh-oh! Couldn't write " << candidate_set.reference_string() << endl;
    }
  }
  csw.Close();

  // Finally, output a symbol table if user specified one.
  if (symbol_table_output_file != "") {
    cerr << "Writing out Symbol protocol buffer messages to file \""
         << symbol_table_output_file << "\"." << endl;
    ConfusionProtoIO proto_writer(symbol_table_output_file,
                                  ConfusionProtoIO::WRITE,
                                  compressed, use_base64);
    for (Symbols::const_iterator it = symbols->begin();
         it != symbols->end();
         ++it) {
      SymbolMessage symbol_message;
      symbol_message.set_symbol(it->first);
      symbol_message.set_index(it->second);
      proto_writer.Write(symbol_message);
    }
    proto_writer.Close();
  }

  TearDown();
  google::protobuf::ShutdownProtobufLibrary();
}
