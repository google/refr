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
/// instances and either outputs all symbolic feature names, or else
/// reads in a symbol table from file and compiles all feature vectors
/// in each reranker::CandidateSet instance using that supplied symbol table,
/// serializing the reranker::CandidateSet instances to <tt>stdout</tt>.
/// \author dbikel@google.com (Dan Bikel)

#include <string>
#include <cstdlib>
#include <tr1/memory>
#include <vector>

#include "../proto/dataio.h"
#include "candidate-set.H"
#include "candidate-set-iterator.H"
#include "candidate-set-writer.H"
#include "executive-feature-extractor.H"
#include "symbol-table.H"

#define PROG_NAME "compile-features"

#define DEFAULT_MAX_EXAMPLES -1
#define DEFAULT_MAX_CANDIDATES -1
#define DEFAULT_REPORTING_INTERVAL 1000

// We use two levels of macros to get the string version of an int constant.
#define XSTR(arg) STR(arg)
#define STR(arg) #arg

using namespace std;
using namespace std::tr1;
using namespace reranker;
using confusion_learning::SymbolMessage;

const char *usage_msg[] = {
  "Usage:\n",
  PROG_NAME " -i|--input <candidate set input file>+\n",
  "\t[-d|--decompile]\n",
  "\t[--input-symbols <input symbol table>]\n",
  "\t[--clear-raw]\n",
  "\t[--max-examples <max num examples>]\n",
  "\t[--max-candidates <max num candidates>]\n",
  "\t[-r <reporting interval>]\n",
  "where\n",
  "\t<candidate set input file> is the name of a stream of serialized\n",
  "\t\tCandidateSet instances, or \"-\" for input from standard input\n",
  "\t<input symbol table> is an optional input file containing a Symbols\n",
  "\t\tinstance serialized as a sequence of Symbol messages\n",
  "\t-d|--decompile indicates to decompile features\n",
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
  vector<string> input_files;
  bool compile_or_decompile = false;
  bool decompile = false;
  bool clear_raw = false;
  string symbol_table_input_file = "";
  int max_examples = DEFAULT_MAX_EXAMPLES;
  int max_candidates = DEFAULT_MAX_CANDIDATES;
  int reporting_interval = DEFAULT_REPORTING_INTERVAL;

  // Process options.  The majority of code in this file is devoted to this.
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-i" || arg == "-input" || arg == "--input") {
      string err_msg = string("no input files specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      // Keep reading args until next option or until no more args.
      ++i;
      for ( ; i < argc; ++i) {
        if (argv[i][0] == '-' && strlen(argv[i]) > 1) {
          --i;
          break;
        }
        input_files.push_back(argv[i]);
      }
    } else if (arg == "-input-symbols" || arg == "--input-symbols") {
      string err_msg =
          string("no symbol table input file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      symbol_table_input_file = argv[++i];
      compile_or_decompile = true;
    } else if (arg == "-d" || arg == "-decompile" || arg == "--decompile") {
      decompile = true;
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

  if (decompile && !compile_or_decompile) {
    cerr << PROG_NAME << ": error: cannot specify -d|--decompile without "
         << "--input-symbols" << endl;
    usage();
    return -1;
  }

  bool compressed = true;
  bool uncompressed = false;
  bool use_base64 = true;

  // Now, we finally get to the meat of the code for this executable.
  shared_ptr<Symbols> symbols(new LocalSymbolTable());
  if (symbol_table_input_file != "") {
    ConfusionProtoIO proto_reader(symbol_table_input_file,
                                  ConfusionProtoIO::READ,
                                  compressed, use_base64);
    SymbolMessage symbol_message;
    while (proto_reader.Read(&symbol_message)) {
      symbols->SetIndex(symbol_message.symbol(), symbol_message.index());
    }
    proto_reader.Close();
  }

  CandidateSetWriter csw;

  if (compile_or_decompile) {
    csw.Open("-", uncompressed, use_base64);
  }

  int verbosity = 1;
  shared_ptr<ExecutiveFeatureExtractor> null_efe;
  MultiFileCandidateSetIterator csi(input_files,
                                    null_efe,
                                    max_examples,
                                    max_candidates,
                                    reporting_interval,
                                    verbosity,
                                    compressed,
                                    use_base64);

  while (csi.HasNext()) {
    CandidateSet &candidate_set = csi.Next();
    if (decompile) {
      candidate_set.DecompileFeatures(symbols.get());
    } else {
      // Whether we're in "collect symbols" or "compile features" mode, we
      // invoke CandidateSet::CompileFeatures, because in it collects
      // symbols in the symbol table by default as well as compiling features.
      candidate_set.CompileFeatures(symbols.get());
    }
    if (clear_raw) {
      candidate_set.ClearRawData();
    }
    if (compile_or_decompile) {
      csw.WriteNext(candidate_set);
    }
  }
  if (compile_or_decompile) {
    csw.Close();
  } else {
    // If we're in "collect symbols" mode, write out symbols to cout,
    // one symbol per line (in plain text).
    for (Symbols::const_iterator it = symbols->begin();
         it != symbols->end();
         ++it) {
      cout << it->first << "\n";
    }
    cout.flush();
  }

  TearDown();
  google::protobuf::ShutdownProtobufLibrary();
}
