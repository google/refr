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
/// An executable that loads one or more development test files into memory
/// and then continuously reads filenames from <tt>stdin</tt>, loading the \link
/// reranker::Model Model \endlink instance from each file and evaluating
/// on the development test data, printing the loss to <tt>stdout</tt>.

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <unistd.h>

#include "candidate-set-iterator.H"
#include "candidate-set-reader.H"
#include "model.H"
#include "model-reader.H"

#define DEBUG 0

#define PROG_NAME "piped-model-evaluator"

#define DEFAULT_MAX_EXAMPLES -1
#define DEFAULT_MAX_CANDIDATES -1
#define DEFAULT_REPORTING_INTERVAL 1000
#define DEFAULT_USE_WEIGHTED_LOSS true

// We use two levels of macros to get the string version of an int constant.
/// Expands the string value of the specified argument using the \link
/// STR \endlink macro.
#define XSTR(arg) STR(arg)
/// &ldquo;Returns&rdquo; the string value of the specified argument.
#define STR(arg) #arg

using namespace std;
using namespace reranker;

const char *usage_msg[] = {
  "Usage:\n",
  PROG_NAME " -d|--devtest <devtest input file>+\n",
  "\t[--dev-config <devtest feature extractor config file>]\n",
  "\t[--model-files <file with model filenames>\n",
  "\t[-u] [--no-base64]\n",
  "\t[--max-examples <max num examples>]\n",
  "\t[--max-candidates <max num candidates>]\n",
  "\t[-r <reporting interval>] [ --use-weighted-loss[=][true|false] ]\n",
  "where\n",
  "\t<devtest input file> is the name of a stream of serialized\n",
  "\t\tCandidateSet instances, or \"-\" for input from standard input\n",
  "\t\t(required unless training in mapper mode)\n",
  "\t--model-files specifies the name of a file from which to read model\n",
  "\t\tmodel filenames (use this option for debugging; defaults to stdin)\n",
  "\t-u specifies that the input files are uncompressed\n",
  "\t--no-base64 specifies not to use base64 encoding/decoding\n",
  "\t--max-examples specifies the maximum number of examples to read from\n",
  "\t\tany input file (defaults to " XSTR(DEFAULT_MAX_EXAMPLES) ")\n",
  "\t--max-candidates specifies the maximum number of candidates to read\n",
  "\t\tfor any candidate set (defaults to " XSTR(DEFAULT_MAX_CANDIDATES) ")\n",
  "\t-r specifies the interval at which the CandidateSetReader reports how\n",
  "\t\tmany candidate sets it has read (defaults to "
  XSTR(DEFAULT_REPORTING_INTERVAL) ")\n",
  "\t--use-weighted-loss specifies whether to weight losses on devtest\n",
  "\t\texamples by the number of tokens in the reference, where, e.g.,\n",
  "\t\tweighted loss is appropriate for computing WER, but not BLEU\n",
  "\t\t(defaults to " XSTR(DEFAULT_USE_WEIGHTED_LOSS) ")\n"
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
  string model_file;
  bool using_model_filenames_file = false;
  string model_filenames_file;
  vector<string> devtest_files;
  string devtest_feature_extractor_config_file;
  bool compressed = true;
  bool use_base64 = true;
  bool use_weighted_loss = DEFAULT_USE_WEIGHTED_LOSS;
  string use_weighted_loss_arg_prefix = "--use-weighted-loss";
  size_t use_weighted_loss_arg_prefix_len =
      use_weighted_loss_arg_prefix.length();
  int max_examples = DEFAULT_MAX_EXAMPLES;
  int max_candidates = DEFAULT_MAX_CANDIDATES;
  int reporting_interval = DEFAULT_REPORTING_INTERVAL;

  // Process options.  The majority of code in this file is devoted to this.
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-d" || arg == "-devtest" || arg == "--devtest") {
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
        devtest_files.push_back(argv[i]);
      }
    } else if (arg == "-dev-config" || arg == "--dev-config") {
      string err_msg =
          string("no feature extractor config file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      devtest_feature_extractor_config_file = argv[++i];
    } else if (arg == "-model-files" || arg == "--model-files") {
      string err_msg = string("no model filenames file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      model_filenames_file = argv[++i];
      using_model_filenames_file = true;
    } else if (arg == "-u") {
      compressed = false;
    } else if (arg == "--no-base64") {
      use_base64 = false;
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
    } else if (arg.substr(0, use_weighted_loss_arg_prefix_len) ==
               use_weighted_loss_arg_prefix) {
      string use_weighted_loss_str;
      if (arg.length() > use_weighted_loss_arg_prefix_len &&
          arg[use_weighted_loss_arg_prefix_len] == '=') {
        use_weighted_loss_str =
            arg.substr(use_weighted_loss_arg_prefix_len + 1);
      } else {
        string err_msg =
            string("no \"true\" or \"false\" arg specified with ") + arg;
        if (!check_for_required_arg(argc, i, err_msg)) {
          return -1;
        }
        use_weighted_loss_str = argv[++i];
      }
      if (use_weighted_loss_str != "true" &&
          use_weighted_loss_str != "false") {
        cerr << PROG_NAME << ": error: must specify \"true\" or \"false\""
             << " with --use-weighted-loss" << endl;
        usage();
        return -1;
      }
      if (use_weighted_loss_str != "true") {
        use_weighted_loss = false;
      }
    } else if (arg.size() > 0 && arg[0] == '-') {
      cerr << PROG_NAME << ": error: unrecognized option: " << arg << endl;
      usage();
      return -1;
    }
  }

  if (devtest_files.size() == 0) {
    cerr << PROG_NAME << ": error: must specify devtest input files when "
         << "not in mapper mode" << endl;
    usage();
    return -1;
  }

  shared_ptr<ExecutiveFeatureExtractor> devtest_efe;
  if (devtest_feature_extractor_config_file != "") {
    devtest_efe = ExecutiveFeatureExtractor::InitFromSpec(
        devtest_feature_extractor_config_file);
  }

  CandidateSetReader csr(max_examples, max_candidates, reporting_interval);
  csr.set_verbosity(1);
  bool reset_counters = true;

  cerr << "Reading devtest examples." << endl;

  vector<shared_ptr<CandidateSet> > devtest_examples;
  for (vector<string>::const_iterator file_it = devtest_files.begin();
       file_it != devtest_files.end();
       ++file_it) {
    csr.Read(*file_it, compressed, use_base64, reset_counters,
             devtest_examples);
  }
  // Extract features for CandidateSet instances in situ.
  for (vector<shared_ptr<CandidateSet> >::iterator it =
           devtest_examples.begin();
       it != devtest_examples.end();
       ++it) {
    devtest_efe->Extract(*(*it));
  }

  cerr << "Done reading devtest examples." << endl;

  if (devtest_examples.size() == 0) {
    cerr << "Could not read any devtest examples.  Exiting." << endl;
    return -1;
  }

  typedef CollectionCandidateSetIterator<vector<shared_ptr<CandidateSet> > >
      CandidateSetVectorIt;

  istream *model_filenames_stream =
      using_model_filenames_file ?
      new ifstream(model_filenames_file.c_str()) : &cin;

  ModelReader model_reader(1);
  while (getline(*model_filenames_stream, model_file)) {
    cerr << "Evaluating model \"" << model_file << "\"." << endl;
    shared_ptr<Model> model =
        model_reader.Read(model_file, compressed, use_base64);
    model->set_use_weighted_loss(use_weighted_loss);
    CandidateSetVectorIt devtest_examples_it(devtest_examples);
    model->NewEpoch(); // sets epoch to 0
    cout << model->Evaluate(devtest_examples_it) << endl;

    // Decompile all features in devtest examples (will do nothing if there
    // were no symbolic features to begin with; see "dont_force" below).
    devtest_examples_it.Reset();
    while (devtest_examples_it.HasNext()) {
      CandidateSet &candidate_set = devtest_examples_it.Next();
      bool dont_force = false;
      candidate_set.DecompileFeatures(model->symbols(), true, true, dont_force);
    }
  }
  if (using_model_filenames_file) {
    delete model_filenames_stream;
  }
}
