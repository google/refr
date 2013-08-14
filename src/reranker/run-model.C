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
/// instances from a stream and then optionally runs feature extractors
/// on those instances using an reranker::ExecutiveFeatureExtractor and,
/// finally, trains or tests a model on those instances.  (Recall
/// that a reranker::ExecutiveFeatureExtractor is like a regular
/// feature extractor, but wears fancypants.)
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <tr1/memory>
#include <vector>

#include "candidate.H"
#include "candidate-set.H"
#include "candidate-set-reader.H"
#include "candidate-set-writer.H"
#include "executive-feature-extractor.H"
#include "model.H"
#include "perceptron-model.H"
#include "model-merge-reducer.H"
#include "model-reader.H"
#include "model-proto-writer.H"
#include "symbol-table.H"

#define DEBUG 0

#define PROG_NAME "run-model"

#define DEFAULT_MAX_EXAMPLES -1
#define DEFAULT_MAX_CANDIDATES -1
#define DEFAULT_MODEL_CONFIG "PerceptronModel(name(\"MyPerceptronModel\"))"
#define DEFAULT_REPORTING_INTERVAL 1000
#define DEFAULT_COMPACTIFY_INTERVAL 10000
#define DEFAULT_USE_WEIGHTED_LOSS true

// We use two levels of macros to get the string version of an int constant.
/// Expands the string value of the specified argument using the \link
/// STR \endlink macro.
#define XSTR(arg) STR(arg)
/// &ldquo;Returns&rdquo; the string value of the specified argument.
#define STR(arg) #arg

using namespace std;
using namespace std::tr1;
using namespace reranker;

const char *usage_msg[] = {
  "Usage:\n",
  PROG_NAME " -m|--model-file <model file> [--model-config <model config>]\n"
  "\t[-t|--train <training input file>+ [-i <input model file>] [--mapper] ]\n",
  "\t-d|--devtest <devtest input file>+\n",
  "\t[-o|--output <candidate set output file>]\n",
  "\t[-h <hyp output file>] [--scores <score output file>]\n",
  "\t[--train-config <training feature extractor config file>]\n",
  "\t[--dev-config <devtest feature extractor config file>]\n",
  "\t[--compactify-feature-uids]\n",
  "\t[-s|--streaming [--compactify-interval <interval>] ] [-u]\n",
  "\t[--no-base64]\n",
  "\t[--min-epochs <min epochs>] [--max-epochs <max epochs>]\n",
  "\t[--max-examples <max num examples>]\n",
  "\t[--max-candidates <max num candidates>]\n",
  "\t[-r <reporting interval>] [ --use-weighted-loss[=][true|false] ]\n",
  "where\n",
  "\t<model file> is the name of the file to which to write out a\n",
  "\t\tnewly-trained model when training (one or more\n",
  "\t\t<training input file>'s specified), or the name of a file\n",
  "\t\tfrom which to load a serialized model when decoding\n",
  "\t<input model file> is an optional input model file as a starting\n",
  "\t\tmodel when training\n",
  "\t<model config> is the optional configuration string for constructing\n",
  "\t\ta new Model instance\n",
  "\t\t(defaults to \"" DEFAULT_MODEL_CONFIG "\")\n",
  "\t<training input file> is the name of a stream of serialized\n",
  "\t\tCandidateSet instances, or \"-\" for input from standard input\n",
  "\t--mapper specifies to train a single epoch and output features to\n",
  "\t\tstandard output\n",
  "\t<devtest input file> is the name of a stream of serialized\n",
  "\t\tCandidateSet instances, or \"-\" for input from standard input\n",
  "\t\t(required unless training in mapper mode)\n",
  "\t<candidate set output file> is the name of the file to which to output\n",
  "\t\tcandidate sets that have been scored by the model (in\n",
  "\t\tdecoding mode)\n",
  "\t<training feature extractor config file> is the name of a configuration\n",
  "\t\tfile to be read by the ExecutiveFeatureExtractor instance\n"
  "\t\textracting features on training examples\n",
  "\t<devtest feature extractor config file> is the name of a configuration\n",
  "\t\tfile to be read by the ExecutiveFeatureExtractor instance\n",
  "\t\textracting features on devtest examples\n",
  "\t--compactify-feature-uids specifies to re-map all feature uids to the\n",
  "\t\t[0,n-1] interval, where n is the number of non-zero features\n",
  "\t--streaming specifies to train in streaming mode (i.e., do not\n",
  "\t\tread in all training instances into memory)\n",
  "\t--compactify-interval specifies the interval after which to compactify\n",
  "\t\tfeature uid's and remove unused symbols (only available when\n",
  "\t\ttraining in streaming mode; defaults to "
  XSTR(DEFAULT_COMPACTIFY_INTERVAL) ")\n",
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

void read_and_extract_features(const vector<string> &files,
                               CandidateSetReader &csr,
                               bool compressed,
                               bool use_base64,
                               ExecutiveFeatureExtractor &efe,
                               vector<shared_ptr<CandidateSet> > &examples) {
  bool reset_counters = true;
  for (vector<string>::const_iterator file_it = files.begin();
       file_it != files.end();
       ++file_it) {
    csr.Read(*file_it, compressed, use_base64, reset_counters, examples);
  }
  // Extract features for CandidateSet instances in situ.
  for (vector<shared_ptr<CandidateSet> >::iterator it = examples.begin();
       it != examples.end();
       ++it) {
    efe.Extract(*(*it));
  }
}

int
main(int argc, char **argv) {
  // Required parameters.
  string model_file;
  string input_model_file;
  string model_config = DEFAULT_MODEL_CONFIG;
  vector<string> training_files;
  vector<string> devtest_files;
  bool mapper_mode = false;
  string output_file;
  string hyp_output_file;
  string score_output_file;
  string training_feature_extractor_config_file;
  string devtest_feature_extractor_config_file;
  bool compressed = true;
  bool use_base64 = true;
  bool streaming = false;
  bool use_weighted_loss = DEFAULT_USE_WEIGHTED_LOSS;
  string use_weighted_loss_arg_prefix = "--use-weighted-loss";
  size_t use_weighted_loss_arg_prefix_len =
      use_weighted_loss_arg_prefix.length();
  bool compactify_feature_uids = false;
  int compactify_interval = DEFAULT_COMPACTIFY_INTERVAL;
  int min_epochs = -1;
  int max_epochs = -1;
  int max_examples = DEFAULT_MAX_EXAMPLES;
  int max_candidates = DEFAULT_MAX_CANDIDATES;
  int reporting_interval = DEFAULT_REPORTING_INTERVAL;

  // Process options.  The majority of code in this file is devoted to this.
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-m" || arg == "-model" || arg == "--model") {
      string err_msg = string("no model file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      model_file = argv[++i];
    } else if (arg == "-i" || arg == "--i") {
      string err_msg = string("no input model file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      input_model_file = argv[++i];
    } else if (arg == "-model-config" || arg == "--model-config") {
      string err_msg =
          string("no model configuration string specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      model_config = argv[++i];
    } else if (arg == "-t" || arg == "-train" || arg == "--train") {
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
        training_files.push_back(argv[i]);
      }
    } else if (arg == "-mapper" || arg == "--mapper") {
      mapper_mode = true;
    } else if (arg == "-d" || arg == "-devtest" || arg == "--devtest") {
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
    } else if (arg == "-o" || arg == "-output" || arg == "--output") {
      string err_msg = string("no output file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      output_file = argv[++i];
    } else if (arg == "-h") {
      string err_msg =
          string("no hypothesis output file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      hyp_output_file = argv[++i];
    } else if (arg == "-scores" || arg == "--scores") {
      string err_msg =
          string("no score output file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      score_output_file = argv[++i];
    } else if (arg == "-train-config" || arg == "--train-config") {
      string err_msg =
          string("no feature extractor config file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      training_feature_extractor_config_file = argv[++i];
    } else if (arg == "-dev-config" || arg == "--dev-config") {
      string err_msg =
          string("no feature extractor config file specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      devtest_feature_extractor_config_file = argv[++i];
    } else if (arg == "-compactify-feature-uids" ||
               arg == "--compactify-feature-uids") {
      compactify_feature_uids = true;
    } else if (arg == "-s" || arg == "-streaming" || arg == "--streaming") {
      streaming = true;
    } else if (arg == "--compactify-interval") {
      string err_msg = string("no interval specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      compactify_interval = atoi(argv[++i]);
    } else if (arg == "-u") {
      compressed = false;
    } else if (arg == "--no-base64") {
      use_base64 = false;
    } else if (arg == "-min-epochs" || arg == "--min-epochs") {
      string err_msg = string("no arg specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      min_epochs = atoi(argv[++i]);
    } else if (arg == "-max-epochs" || arg == "--max-epochs") {
      string err_msg = string("no arg specified with ") + arg;
      if (!check_for_required_arg(argc, i, err_msg)) {
        return -1;
      }
      max_epochs = atoi(argv[++i]);
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

  bool training = training_files.size() > 0;

  // Check that user specified required args.
  if (model_file == "") {
    cerr << PROG_NAME << ": error: must specify model file" << endl;
    usage();
    return -1;
  }
  if (!mapper_mode && devtest_files.size() == 0) {
    cerr << PROG_NAME << ": error: must specify devtest input files when "
         << "not in mapper mode" << endl;
    usage();
    return -1;
  }
  if (output_file != "" && training) {
    cerr << PROG_NAME << ": error: cannot specify output file when training"
         << endl;
    usage();
    return -1;
  }
  if (hyp_output_file != "" && training) {
    cerr << PROG_NAME
         << ": error: cannot specify hypothesis output file when training"
         << endl;
    usage();
    return -1;
  }
  bool reading_from_stdin = false;
  for (vector<string>::const_iterator training_file_it = training_files.begin();
       training_file_it != training_files.end();
       ++training_file_it) {
    if (*training_file_it == "-") {
      reading_from_stdin = true;
      break;
    }
  }
  if (training_files.size() > 1 && reading_from_stdin) {
    cerr << PROG_NAME << ": error: cannot read from standard input and "
         << "specify other training files" << endl;
    usage();
    return -1;
  }
  if (!training && input_model_file != "") {
    cerr << PROG_NAME << ": error: can only specify <input model file> "
         << "when in training mode" << endl;
    usage();
    return -1;
  }

  // Now, we finally get to the meat of the code for this executable.
  ExecutiveFeatureExtractor training_efe;
  if (training_feature_extractor_config_file != "") {
    training_efe.Init(training_feature_extractor_config_file);
  }
  ExecutiveFeatureExtractor devtest_efe;
  if (devtest_feature_extractor_config_file != "") {
    devtest_efe.Init(devtest_feature_extractor_config_file);
  }

  CandidateSetReader csr(max_examples, max_candidates, reporting_interval);
  csr.set_verbosity(1);

  shared_ptr<Model> model;
  Factory<Model> model_factory;

  if (!training || input_model_file != "") {
    // We're here because we're not training, or else we are training and
    // the user specified an input model file.
    string model_file_to_load = training ? input_model_file : model_file;

    ModelReader model_reader(1);
    model = model_reader.Read(model_file_to_load, compressed, use_base64);
  } else {
    // First, see if model_config is the name of a file.
    ifstream model_config_is(model_config.c_str());
    if (model_config_is) {
      cerr << "Reading model config from file \"" << model_config << "\"."
           << endl;
    }

    StreamTokenizer *st = model_config_is.good() ?
        new StreamTokenizer(model_config_is) :
        new StreamTokenizer(model_config);
    model = model_factory.CreateOrDie(*st);
    delete st;
  }
  if (model.get() == NULL) {
    return -1;
  }

  Factory<ModelProtoWriter> proto_writer_factory;
  shared_ptr<ModelProtoWriter> model_writer =
      proto_writer_factory.CreateOrDie(model->proto_writer_spec(),
                                       "model proto writer");
  if (model_writer.get() == NULL) {
    return -1;
  }

  if (!mapper_mode) {
    model->set_end_of_epoch_hook(new EndOfEpochModelWriter(model_file,
                                                           model_writer,
                                                           compressed,
                                                           use_base64));
  }
  model->set_use_weighted_loss(use_weighted_loss);
  model->set_min_epochs(min_epochs);
  model->set_max_epochs(max_epochs);

  vector<shared_ptr<CandidateSet> > training_examples;
  vector<shared_ptr<CandidateSet> > devtest_examples;
  if (!streaming && !mapper_mode) {
    cerr << "Loading devtest examples." << endl;
    read_and_extract_features(devtest_files, csr, compressed, use_base64,
                              devtest_efe, devtest_examples);
    if (devtest_examples.size() == 0) {
      cerr << "Could not read any devtest examples.  Exiting." << endl;
      return -1;
    }
  }

  typedef CollectionCandidateSetIterator<vector<shared_ptr<CandidateSet> > >
      CandidateSetVectorIt;

  CandidateSetIterator *training_it;
  CandidateSetIterator *devtest_it;

  if (training_files.size() > 0) {
    cerr << "Training." << endl;
    if (streaming) {
      training_it = new MultiFileCandidateSetIterator(training_files,
                                                      &training_efe,
                                                      max_examples,
                                                      max_candidates,
                                                      reporting_interval,
                                                      1,
                                                      compressed, use_base64);
      devtest_it = new MultiFileCandidateSetIterator(devtest_files,
                                                     &devtest_efe,
                                                     max_examples,
                                                     max_candidates,
                                                     reporting_interval,
                                                     1,
                                                     compressed, use_base64);
      // TODO(dbikel): Make sure to add setter method to Model and
      //               PerceptronModel to tell model to invoke its
      //               CompactifyFeatureUids method after a specified
      //               interval.  This new setter method should only
      //               be invoked here, when in streaming mode.
    } else {
      // Regular, in-memory, non-streaming training.
      read_and_extract_features(training_files, csr, compressed, use_base64,
                                training_efe, training_examples);
      if (training_examples.size() == 0) {
        cerr << "Could not read any training examples from training files."
             << "  Exiting." << endl;
        return -1;
      }
      csr.ClearStrings();

      training_it = new CandidateSetVectorIt(training_examples);
      devtest_it = new CandidateSetVectorIt(devtest_examples);
    }

    if (mapper_mode) {
      // In mapper mode, train a single epoch, then write out features
      // to stdout, and serialize model.
      model->NewEpoch();
      model->TrainOneEpoch(*training_it);
    } else {
      model->Train(*training_it, *devtest_it);
      delete training_it;
      delete devtest_it;
    }

    if (compactify_feature_uids) {
      cerr << "Compactifying feature uid's...";
      cerr.flush();
      model->CompactifyFeatureUids();
      cerr << "done." << endl;
    }

    // Serialize model.
    cerr << "Writing out model to file \"" << model_file << "\"...";
    cerr.flush();
    confusion_learning::ModelMessage model_message;
    model_writer->Write(model.get(), &model_message, false);

    ConfusionProtoIO* proto_writer;
    if (mapper_mode) {
      cerr << "Writing ModelMessage (without features) and FeatureMessage "
           << "instances to standard output." << endl;
      proto_writer = new ConfusionProtoIO(model_file, ConfusionProtoIO::WRITESTD,
                                          false, use_base64);
      cout << ModelInfoReducer::kModelMessageFeatureName << "\t";
    } else {
      proto_writer = new ConfusionProtoIO(model_file, ConfusionProtoIO::WRITE,
                                          compressed, use_base64);
    }
    proto_writer->Write(model_message);
    // Write out features.
    bool output_best_epoch = !mapper_mode;
    bool output_key = mapper_mode;
    model_writer->WriteFeatures(model.get(), 
                                *(proto_writer->outputstream()),
                                output_best_epoch,
                                model->num_training_errors(),
                                output_key);
    delete proto_writer;
    cerr << "done." << endl;
  } else {
    CandidateSetVectorIt devtest_examples_it(devtest_examples);
    model->NewEpoch(); // sets epoch to 0
    model->Evaluate(devtest_examples_it);

    if (output_file != "") {
      CandidateSetWriter csw;
      csw.set_verbosity(1);
      csw.Write(devtest_examples, output_file, compressed, use_base64);
    }
    bool output_hyps = hyp_output_file != "";
    bool output_scores = score_output_file != "";
    if (output_hyps || output_scores) {
      ofstream hyp_os(hyp_output_file.c_str());
      ofstream score_os(score_output_file.c_str());
      devtest_examples_it.Reset();
      while (devtest_examples_it.HasNext()) {
        CandidateSet &candidate_set = devtest_examples_it.Next();
        if (output_hyps) {
          hyp_os << candidate_set.GetBestScoring().raw_data() << "\n";
        }
        if (output_scores) {
          for (CandidateSet::const_iterator cand_it = candidate_set.begin();
               cand_it != candidate_set.end();
               ++cand_it) {
            score_os << (*cand_it)->score() << "\n";
          }
        }
      }
      if (output_hyps) {
        hyp_os.flush();
      }
      if (output_scores) {
        score_os.flush();
      }
    }
  }
  TearDown();
  google::protobuf::ShutdownProtobufLibrary();
}

/// \mainpage Reranker Framework
/// \section welcome_sec Welcome to the Reranker Framework!
/// This package provides ways to train and use discriminative models
/// in a reranking framework.  There is some special handling for
/// building discriminative language models.
///
/// \section contents_sec Contents
/// <ul>
/// <li> \ref building_sec
///   <ul>
///     <li> \ref quick_start_subsec
///     <li> \ref detailed_instructions_subsec
///   </ul>
/// <li> \ref whats_in_here_sec
/// <li> \ref extracting_features_sec
///   <ul>
///     <li> \ref io_subsec
///     <li> \ref bootstrap_protobuf
///       <ul>
///         <li> \ref candidate_set_creation_1
///         <li> \ref candidate_set_creation_2
///       </ul>
///     <li> \ref classes_subsec
///     <li> \ref building_a_feat_extractor_subsec
///     <li> \ref extract_features_subsec
///     <li> \ref bonus_subsec
///   </ul>
/// <li> \ref training_a_model_sec
/// <li> \ref running_a_model_sec
/// <li> \ref appendix_sec
///   <ul>
///     <li> \ref appendix_cpp_example
///     <li> \ref appendix_details
///     <li> \ref appendix_language
///     <li> \ref appendix_putting_together
///   </ul>
/// </ul>
///
/// \section building_sec Building and installation
///
/// \subsection quick_start_subsec Quick start
/// To build and install, run the following command sequence:
/// \code ./configure ; make ; make install \endcode
///
/// \subsection detailed_instructions_subsec Detailed instructions
/// <b>Requirements</b>:
/// <ul>
/// <li><tt>autoconf</tt> 2.68 or higher
/// <li><tt>automake</tt> 1.11 or higher
/// </ul>
/// Additional requirements are checked by the supplied <tt>configure</tt>
/// script; they comprise:
/// <ul>
/// <li>a recent version of Python (v2.4 or higher)
/// <li><a href="http://code.google.com/p/protobuf/">Google Protocol Buffers</a>
/// <li><tt>pkg-config</tt> (a requirement of Google Protocol Buffers)
/// </ul>
/// Please make sure you have at least the preceding three packages installed
/// prior to building the ReFr.
///
/// To build the Reranker Framework package, you must first run the supplied
/// <tt>configure</tt> script.  Please run
/// \code ./configure --help \endcode
/// to see common options.  In particular, you can use the <tt>--prefix</tt>
/// option to specify the installation directory, which defaults to
/// <tt>/usr/local/</tt>.
///
/// After running <tt>./configure</tt> with any desired options, you can build
/// the entire package by simply issuing the make command:
/// \code make \endcode
///
/// Installation of the package is completed by running
/// \code make install \endcode
///
/// Finally, there are a number of additional make targets supplied
/// &ldquo;for free&rdquo; with the GNU autoconf build system, the most useful
/// of which is
/// \code make clean \endcode
/// which clean the build directory and
/// \code make distclean \endcode
/// which cleans everything, including files auto-generated by the
/// <tt>configure</tt> script.
///
/// \section whats_in_here_sec What&rsquo;s in the installation directory
/// Executables are in the <tt>bin</tt> subdirectory, and a library is built in
/// the <tt>lib</tt> subdirectory. There are many unit test executables, but
/// the two &ldquo;real&rdquo; binaries you&rsquo;ll care about are:
/// \code bin/extract-features \endcode
/// and
/// \code bin/run-model \endcode
///
/// \section extracting_features_sec Extracting features
///
/// When dealing with feature extraction, you can work in an off-line
/// mode, where you read in candidate hypotheses, extract features for
/// those hypotheses and then write those back out to a file.  You can
/// also work in an on-line mode, where you train a model by reading
/// in sets of candidates (hereafter referred to as &ldquo;candidate
/// sets&rdquo;) where features for each candidate in each set are
/// extracted &ldquo;on the fly&rdquo;.  Finally, you can mix these
/// two ways of working, as we&rsquo;ll see below.
///
/// \subsection io_subsec I/O
/// The Reranker framework uses protocol buffers for all low-level
/// I/O. (See <a href="http://code.google.com/p/protobuf/">
/// http://code.google.com/p/protobuf/</a> for more information.)  In
/// short, protocol buffers provide a way to serialize and de-serialize
/// things that look a lot like C <tt>struct</tt>&rsquo;s.  You specify
/// a protocol buffer in a format very familiar to C/C++/Java programmers.
/// The protocol buffer definitions for the Reranker framework are all
/// specified in two files
/// \code src/proto/data.proto \endcode
/// and
/// \code src/proto/model.proto \endcode
///
/// While you might be interested in perusing these files for your own
/// edification, the Reranker framework has reader and writer classes
/// that abstract away from this low-level representation.
///
/// \subsection bootstrap_protobuf Creating protocol buffer files
/// You <i>do</i> need to get some candidate sets into this protocol buffer
/// format to begin working with the Reranker framework, and so to
/// bootstrap the process, you can use the executables in the
/// <tt>src/dataconvert</tt> directory (see the <tt>README</tt> file
/// in that directory for example usage).  The <tt>asr_nbest_proto</tt>
/// executable can read sets of files in Brian Roark&rsquo;s format,
/// and the <tt>mt_nbest_proto</tt> can read files in Philipp Koehn&rsquo;s
/// format.
///
/// What if you have files that are not in either of those two
/// formats?  The answer is that you can easily construct your own
/// \link reranker::CandidateSet CandidateSet \endlink instances in
/// memory and write them out to file, serialized using their protocol
/// buffer equivalent, <tt>CandidateSetMessage</tt>.  The only
/// requirements are that each \link reranker::CandidateSet
/// CandidateSet \endlink needs to have a reference string, and each
/// \link reranker::Candidate Candidate \endlink needs to have a baseline
/// score, a loss value, a string consisting of its tokens and the number
/// of its tokens.  Here are the two methods:
///
/// \subsubsection candidate_set_creation_1 Method 1: Batch
///  This method creates a sequence of \link reranker::CandidateSet
/// CandidateSet \endlink in memory, pushing each into an STL
/// <tt>std::vector</tt>, and then writes that vector out to disk.
/// Below is a rough idea of what your code would look like.  The following
/// invents methods/functions for grabbing the data for each new
/// \link reranker::CandidateSet CandidateSet \endlink and
/// \link reranker::Candidate Candidate \endlink and instance, and assumes
/// you want to output to the file named by the variable <tt>filename</tt>.
/// \code
/// #include <vector>
/// #include <tr1/memory>
/// #include "candidate-set.H"
/// #include "candidate-set-writer.H"
/// ...
/// using std::vector;
/// using std::tr1::shared_ptr;
/// ...
/// vector<shared_ptr<CandidateSet> > candidate_sets;
/// while (there_are_more_candidate_sets()) {
///    string reference = get_candidate_set_reference_string();
///    shared_ptr<CandidateSet> candidate_set(new CandidateSet());
///    for (int i = 0; i < number_of_candidates; ++i) {
///      // Assemble the data for current Candidate, build it and add it.
///      double loss = get_curr_candidate_loss();
///      double baseline_score = get_curr_candidate_baseline_score();
///      int num_tokens = get_curr_candidate_num_tokens();
///      string raw_data = get_curr_candidate_string();
///      shared_ptr<Candidate> candidate(new Candidate(i, loss, baseline_score,
///                                                    num_tokens, raw_data));
///      candidate_set->AddCandidate(candidate);
///    }
///    candidate_sets.push_back(candidate_set);
/// }
/// // Finally, write out entire vector of CandidateSet instances.
/// CandidateSetWriter candidate_set_writer;
/// bool compressed = true;
/// bool use_base64 = true;
/// candidate_set_writer.Write(candidate_sets, filename, compressed, use_base64);
/// \endcode
///
/// \subsubsection candidate_set_creation_2 Method 2: Serial
///
/// This method is nearly identical to \ref candidate_set_creation_1
/// "Method 1", but does not try to assemble all \link
/// reranker::CandidateSet CandidateSet \endlink into a single
/// <tt>std::vector</tt> before writing them all out to disk.
/// \code
/// #include <tr1/memory>
/// #include "candidate-set.H"
/// #include "candidate-set-writer.H"
/// ...
/// using std::tr1::shared_ptr;
/// ...
/// // Set up CandidateSetWriter to begin serial writing to file.
/// bool compressed = true;
/// bool use_base64 = true;
/// CandidateSetWriter candidate_set_writer;
/// candidate_set_writer.Open(filename, compressed, use_base64);
/// while (there_are_more_candidate_sets()) {
///    string reference = get_candidate_set_reference_string();
///    CandidateSet candidate_set;
///    for (int i = 0; i < number_of_candidates; ++i) {
///      // Assemble the data for current Candidate, build it and add it.
///      double loss = get_curr_candidate_loss();
///      double baseline_score = get_curr_candidate_baseline_score();
///      int num_tokens = get_curr_candidate_num_tokens();
///      string raw_data = get_curr_candidate_string();
///      shared_ptr<Candidate> candidate(new Candidate(i, loss, baseline_score,
///                                                    num_tokens, raw_data));
///      candidate_set.AddCandidate(candidate);
///    }
///    // Serialize this newly constructed CandidateSet to file.
///    candidate_set_writer.WriteNext(candidate_set);
/// }
/// candidate_set_writer.Close();
/// \endcode
///
/// There&rsquo;s a third, secret method for reading in candidate sets
/// from arbitrary formats.  You can build an implementation of the
/// rather simple \link reranker::CandidateSetIterator
/// CandidateSetIterator \endlink interface, which is what the \link
/// reranker::Model Model \endlink interface uses to iterate over a sequence of
/// candidate sets during training or decoding.  With this approach,
/// your data never gets stored as protocol buffer messages. Given the
/// utility of storing information in protocol buffers, however, we
/// strongly advise against using this method.
///
/// \subsection classes_subsec Classes
/// If you want to extract features, there are just four classes in
/// the Reranker framework you&rsquo;ll want to know about:
/// <table>
/// <tr><th>Class name</th><th>Brief description</th></tr>
/// <tr>
///   <td>\link reranker::Candidate Candidate \endlink</td>
///   <td>Describes a candidate hypothesis put forth by a baseline model
///       for some problem instance (<i>e.g.</i>, a sentence in the case of MT
///       or an utterance in the case of speech recognition).
///   </td>
/// </tr>
/// <tr>
///   <td>\link reranker::CandidateSet CandidateSet \endlink</td>
///   <td>A set of candidate hypotheses for a single problem instance.</td>
/// </tr>
/// <tr>
///   <td>\link reranker::FeatureVector FeatureVector \endlink</td>
///   <td>A mapping from feature uid&rsquo;s (either
///       <tt>string</tt>&rsquo;s or <tt>int</tt>&rsquo;s) to their
///       values (<tt>double</tt>&rsquo;s).
///   </td>
/// </tr>
/// <tr>
///   <td>\link reranker::FeatureExtractor FeatureExtractor \endlink</td>
///   <td>An interface/abstract base class that you will extend to
///       write your own feature extractors.
///   </td>
/// </tr>
///
/// \subsection building_a_feat_extractor_subsec Building a FeatureExtractor
///
/// To build your own \link reranker::FeatureExtractor
/// FeatureExtractor\endlink, follow these steps:
/// <ol>
/// <li> Create a class that derives from \link
///      reranker::FeatureExtractor FeatureExtractor\endlink.
/// <li> <i>(optional)</i> Override the \link
///      reranker::FeatureExtractor::RegisterInitializers
///      FeatureExtractor::RegisterInitializers \endlink method in
///      case your \link reranker::FeatureExtractor FeatureExtractor
///      \endlink needs to set certain of its data members when
///      constructed by a \link reranker::Factory
///      Factory\endlink. Also, one may override the \link
///      reranker::FeatureExtractor::Init FeatureExtractor::Init
///      \endlink method if the feature extractor requires more object
///      initialization after its data members have been
///      initialized. See \ref appendix_sec for more information about
///      how various objects are constructed via \link
///      reranker::Factory Factory\endlink instances.
/// <li> Register your \link reranker::FeatureExtractor
///      FeatureExtractor \endlink using the \link
///      REGISTER_FEATURE_EXTRACTOR \endlink macro.  This is also
///      required to be able to construct your \link
///      reranker::FeatureExtractor FeatureExtractor \endlink by the
///      \link reranker::Factory Factory \endlink class.
/// <li> Implement either the \link
///      reranker::FeatureExtractor::Extract FeatureExtractor::Extract
///      \endlink or the \link
///      reranker::FeatureExtractor::ExtractSymbolic
///      FeatureExtractor::ExtractSymbolic \endlink method.  See below
///      for more information about implementing these methods.
/// </ol>
///
/// See \link example-feature-extractor.H \endlink for a
/// fully-functional (albeit boring) \link reranker::FeatureExtractor
/// FeatureExtractor \endlink implementation.  Please note that
/// normally, one would use the \link REGISTER_FEATURE_EXTRACTOR
/// \endlink macro in one&rsquo;s feature extractor&rsquo;s <tt>.C</tt> file,
/// but for the \link reranker::ExampleFeatureExtractor
/// ExampleFeatureExtractor \endlink this is done in the <tt>.H</tt>
/// to keep things simple.
///
/// As mentioned in Step 4 above, in most cases, you&rsquo;ll either
/// want to provide an implementation for the \link
/// reranker::FeatureExtractor::Extract FeatureExtractor::Extract
/// \endlink or the \link reranker::FeatureExtractor::ExtractSymbolic
/// FeatureExtractor::ExtractSymbolic \endlink methods, but not both.
/// In fact, you&rsquo;ll most likely just want to implement \link
/// reranker::FeatureExtractor::ExtractSymbolic
/// ExtractSymbolic\endlink, which allows you to extract features that
/// are <tt>string</tt>&rsquo;s that map to <tt>double</tt> values.
/// (Since both methods are pure virtual in the \link
/// reranker::FeatureExtractor FeatureExtractor \endlink definition,
/// you&rsquo;ll have to implement both, but either&mdash;or
/// both&mdash;can be implemented to do nothing.)
///
/// When implementing the \link
/// reranker::FeatureExtractor::ExtractSymbolic
/// FeatureExtractor::ExtractSymbolic \endlink method, you will
/// normally modify just the second parameter, which is a reference to
/// a \link reranker::FeatureVector FeatureVector\<string,
/// double\>\endlink.  You&rsquo;ll typically modify it using the
/// \link reranker::FeatureVector::IncrementWeight
/// FeatureVector::IncrementWeight \endlink method. (There&rsquo;s
/// also a \link reranker::FeatureVector::SetWeight
/// FeatureVector::SetWeight \endlink method, but that will blow away
/// any existing weight for the specified feature, and so it should
/// not normally be used.)
///
/// \subsection extract_features_subsec Extracting Features (Finally!)
///
/// If you want to extract features for an existing file containing
/// serialized \link reranker::CandidateSet CandidateSet \endlink
/// instances (again, for now, created using the tools in
/// <tt>src/dataconvert</tt>), you can write a short configuration
/// file that specifies which \link reranker::FeatureExtractor
/// FeatureExtractor \endlink implementations to instantiate at
/// run-time and execute for each \link reranker::Candidate Candidate
/// \endlink of each \link reranker::CandidateSet
/// CandidateSet\endlink.  (The \link
/// reranker::ExecutiveFeatureExtractor ExecutiveFeatureExtractor
/// \endlink class is responsible for &ldquo;executing&rdquo; the
/// feature extraction from this user-specified suite of feature
/// extractors, all of which are built by a \link reranker::Factory
/// Factory \endlink inside the \link
/// reranker::ExecutiveFeatureExtractor
/// ExecutiveFeatureExtractor\endlink.)
///
/// An example of such a configuration file is <tt>test-fe.config</tt>
/// in the directory <tt>learning/reranker/config</tt>. The format of
/// a feature extractor configuration file should be a sequence of
/// <i>specification strings</i>, each of which looks like \code
/// FeatureExtractorClassName(init_string) \endcode Please see \ref
/// appendix_sec for more details on factories and the ability to
/// construct objects from specification strings.  (For a formal, BNF
/// description of the format of a <i>specification string</i>, please
/// see the documentation for the \link reranker::Factory::CreateOrDie
/// \endlink method.)
///
/// You can then pass this configuration file, along with one or more
/// input files and an output directory, to the
/// <tt>bin/extract-features</tt> executable.  The executable will
/// read the \link reranker::CandidateSet CandidateSet \endlink
/// instances from the input files, and, for each \link
/// reranker::Candidate Candidate \endlink in each \link
/// reranker::CandidateSet CandidateSet\endlink, will run the \link
/// reranker::FeatureExtractor FeatureExtractor\endlink&rsquo;s
/// specified in the config file, in order, on that \link
/// reranker::Candidate Candidate\endlink.  The &ldquo;in order&rdquo;
/// part is significant, if, <i>e.g.</i>, you have a \link
/// reranker::FeatureExtractor FeatureExtractor \endlink
/// implementation that expressly uses features generated by a
/// previously-run \link reranker::FeatureExtractor
/// FeatureExtractor\endlink.
///
/// You can execute <tt>extract-features</tt> with no arguments to get
/// the usage.  Here&rsquo;s what your command will look like
/// \code extract-features -c <config file> -i <input file>+ -o <output directory> \endcode
///
/// Your input files will each be read in, new features will be
/// extracted and then the resulting, modified streams of \link
/// reranker::CandidateSet CandidateSet \endlink objects will be
/// written out to a file of the same name in the <tt>\<output
/// directory\></tt>.
///
/// \subsection bonus_subsec Bonus subsection: extracting features already sitting in a file
///
/// If you generate features offline as a text file, where each line
/// corresponds to the features for a single candidate hypothesis,
/// you&rsquo;re in luck.  There&rsquo;s an abstract base class called
/// \link reranker::AbstractFileBackedFeatureExtractor
/// AbstractFileBackedFeatureExtractor \endlink that you can extend to
/// implement a \link reranker::FeatureExtractor FeatureExtractor
/// \endlink that doesn&rsquo;t do any real work, but rather uses
/// whatever it finds in its &ldquo;backing file&rdquo;.  In fact,
/// there's already a concrete implementation in the form of \link
/// reranker::BasicFileBackedFeatureExtractor
/// BasicFileBackedFeatureExtractor\endlink, so you might be able to
/// use that class &ldquo;as is&rdquo;.
///
/// Since the feature extractor configuration file lets you specify
/// any sequence of \link reranker::FeatureExtractor FeatureExtractor
/// \endlink instances, you can mix and match, using some \link
/// reranker::FeatureExtractor FeatureExtractor\endlink&rsquo;s that
/// are truly computing feature functions &ldquo;on the fly&rdquo;,
/// and others that are simply reading in pre-computed features
/// sitting in a file.
///
/// \section training_a_model_sec Training a model
///
/// To train a model, you&rsquo;ll run the <tt>bin/run-model</tt>
/// executable, which does both model training and inference on test
/// data.
///
/// Here&rsquo;s a sample command:
/// \code bin/run-model -t train_file1.gz train_file2.gz -d dev_file1.gz dev_file2.gz -m model_output_file.gz \endcode
///
/// This builds a model based on the serialized \link
/// reranker::CandidateSet CandidateSet\endlink&rsquo;s in
/// <tt>train_file1.gz</tt> and <tt>train_file2.gz</tt>, using
/// <tt>dev_file1.gz</tt> and <tt>dev_file2.gz</tt> for held-out
/// evaluation (as a stopping criterion for the perceptron),
/// outputting the model to <tt>model.gz</tt>. As with the
/// <tt>bin/extract-features</tt> executable, running
/// <tt>bin/run-model</tt> with no arguments prints out a detailed
/// usage message.
///
/// Two options that are common to both the <tt>bin/extract-features</tt>
/// and <tt>bin/run-model</tt> executables are worth mentioning:
/// <ul>
/// <li> The <tt>--max-examples</tt> option specifies the maximum number of
/// training examples (<i>i.e.</i>, \link reranker::CandidateSet CandidateSet
/// \endlink instances) to be read from each input file.
/// <li> The <tt>--max-candidates</tt> option specifies the maxiumum
/// number of candidates to be read per \link reranker::CandidateSet
/// CandidateSet \endlink.  So, even if your input file contains, say,
/// 1000-best hypothesis sets, you can effectively turn them into
/// 100-best hypothesis sets by specifying
/// \code --max-candidates 100 \endcode on the command line.
/// </ul>
///
/// \section running_a_model_sec Running a model
///
/// So you&rsquo;ve trained a model and saved it to a file.  Now what?
/// To run a model on some data (<i>i.e.</i> to do inference), use the
/// <tt>bin/run-model</tt> executable and supply the same command-line
/// arguments as you would for \ref training_a_model_sec "training",
/// except omit the training files that you would specify with the
/// <tt>-t</tt> flag.  In this mode, the model file specified with the
/// <tt>-m</tt> flag is the name of the file from which to load a
/// model that had been trained previously.  That model will then be
/// run on the &ldquo;dev&rdquo; data files you supply with the
/// <tt>-d</tt> flag.
///
/// A command might look like this:
/// \code bin/run-model -d dev_file1.gz dev_file2.gz -m model_input_file.gz \endcode
///
/// \section appendix_sec Appendix: Dynamic object instantiation
///
/// There&rsquo;s a famous quotation of Philip Greenspun known as <a
/// href="http://en.wikipedia.org/wiki/Greenspun's_Tenth_Rule">Greenspun&rsquo;s
/// Tenth Rule</a>:
/// \par Greenspun&rsquo;s Tenth Rule
/// Any sufficiently complicated C or Fortran program contains an ad
/// hoc, informally-specified, bug-ridden, slow implementation of half
/// of Common Lisp.
///
/// This statement is remarkably true in practice, and no less so
/// here.  C++ lacks convenient support for dynamic object
/// instantiation, but the Reranker Framework uses it extensively via
/// a \link reranker::Factory Factory \endlink class and a C++-style
/// (yet simple) syntax.
///
/// \subsection appendix_cpp_example An example: The way C++ does it
///
/// To motivate the C++-style syntax used by the Reranker Framework&rsquo;s
/// \link reranker::Factory Factory \endlink class, let&rsquo;s look
/// at a simple example of a C++ class <tt>Person</tt> and its constructor:
/// \code
/// // A class to represent a date in the standard Gregorian calendar.
/// class Date {
///  public:
///    Date(int year, int month, int day) :
///      year_(year), month_(month), day_(day) { }
///  private:
///    int year_;
///    int month_;
///    int day_;
/// };
///
/// // A class to represent a few facts about a person.
/// class Person {
///  public:
///    Person(const string &name, int cm_height, const Date &birthday) :
///      name_(name), cm_height_(cm_height), birthday_(birthday) { }
///  private:
///   string name_;
///   int cm_height_;
///   Date birthday_;
/// };
/// \endcode
/// As you can see, the <tt>Person</tt> class has three data members,
/// one of which happens to be an instance of another class called
/// <tt>Date</tt>.  In this case, all of the initialization of a
/// <tt>Person</tt> happens in the <i>initialization phase</i> of the
/// constructor&mdash;the part after the colon but before the
/// <i>declaration phase</i> block.  By convention, each parameter to
/// the constructor has a name nearly identical to the data member
/// that will be initialized from it. If we wanted to construct a
/// <tt>Person</tt> instance for someone named &ldquo;Fred&rdquo; who
/// was 180 cm tall and was born January 10th, 1990, we could write
/// the following:
/// \code
///   Person fred("Fred", 180, Date(1990, 1, 10));
/// \endcode
///
/// If <tt>Person</tt> were a \link reranker::Factory
/// Factory\endlink-constructible type in the Reranker Framework,
/// we would be able to specify the following as a <i>specification string</i>
/// to tell the \link reranker::Factory Factory \endlink how to
/// construct a <tt>Person</tt> instance for Fred:
/// \code
///   Person(name("Fred"), cm_height(180), birthday(Date(year(1990), month(1), day(10))))
/// \endcode
/// As you can see, the syntax is very similar to that of
/// C++. It&rsquo;s kind of a combination of the parameter list and
/// the initialization phase of a C++ constructor.  Unfortunately, we
/// can&rsquo;t get this kind of dynamic instantiation in C++ for
/// free; we need some help from the programmer.  However, we&rsquo;ve
/// tried to make the burden on the programmer fairly low, using just
/// a couple of macros to help declare a \link reranker::Factory
/// Factory \endlink for an abstract base class, as well as to make it
/// easy to make that \link reranker::Factory Factory \endlink aware of
/// the concrete subtypes of that base class that it can construct.
///
/// \subsection appendix_details Some nitty gritty details: declaring factories for abstract types and registering concrete subtypes
///
/// Every \link reranker::Factory Factory\endlink-constructible
/// abstract type needs to declare its factory via the \link
/// IMPLEMENT_FACTORY \endlink macro.  For example, since the Reranker
/// Framework uses a \link reranker::Factory Factory \endlink to
/// construct concrete instances of the abstract type \link
/// reranker::FeatureExtractor FeatureExtractor \endlink, the line
/// \code
///   IMPLEMENT_FACTORY(FeatureExtractor)
/// \endcode
/// appears in the file <tt>feature-extractor.C</tt>.  (It is
/// unfortunate that we have to resort to using macros, but the point
/// is that the burden on the programmer to create a factory is
/// extremely low, and therefore so is the risk of introducing bugs.)
///
/// By convention every \link reranker::Factory
/// Factory\endlink-constructible abstract type defines one or two
/// macros in terms of the \link REGISTER_NAMED \endlink macro defined
/// in \link factory.H\endlink to allow concrete subtypes to register
/// themselves with the \link reranker::Factory Factory\endlink, so
/// that they may be instantiated.  For example, since the \link
/// reranker::FeatureExtractor FeatureExtractor \endlink class is an
/// abstract base class in the Reranker Framework that has a \link
/// reranker::Factory Factory\endlink, in feature-extractor.H you
/// can find the declaration of two macros, \link
/// REGISTER_NAMED_FEATURE_EXTRACTOR \endlink and \link
/// REGISTER_FEATURE_EXTRACTOR \endlink.  The \link
/// reranker::NgramFeatureExtractor NgramFeatureExtractor \endlink
/// class is a concrete subclass of \link reranker::FeatureExtractor
/// FeatureExtractor\endlink, and so it registers itself with \link
/// reranker::Factory Factory\endlink\<\link
/// reranker::FeatureExtractor FeatureExtractor\endlink\> by having
/// \code REGISTER_FEATURE_EXTRACTOR(NgramFeatureExtractor) \endcode
/// in <tt>ngram-feature-extractor.C</tt>. That macro expands to \code
/// REGISTER_NAMED(NgramFeatureExtractor, NgramFeatureExtractor,
/// FeatureExtractor) \endcode which tells the \link reranker::Factory
/// Factory\endlink\<FeatureExtractor\> that there is a class
/// <tt>NgramFeatureExtractor</tt> whose &ldquo;factory name&rdquo;
/// (the string that can appear in <i>specification
/// strings</i>&mdash;more on these in a moment) is
/// <tt>&quot;NgramFeatureExtractor&quot;</tt> and that the class
/// \link reranker::NgramFeatureExtractor NgramFeatureExtractor
/// \endlink is a concrete subclass of \link
/// reranker::FeatureExtractor FeatureExtractor \endlink, <i>i.e.</i>,
/// that it can be constructed by
/// <tt>Factory\<FeatureExtractor\></tt>, as opposed to some other
/// <tt>Factory</tt> for a different abstract base class.
///
/// Every \link reranker::Factory Factory\endlink-constructible
/// abstract type must also specify two methods, a
/// <tt>RegisterInitializers(Initializers&)</tt> method and an
/// <tt>Init(const string&)</tt> method. Both methods are guaranteed
/// to be invoked, in order, just after construction of every object
/// by the \link reranker::Factory Factory\endlink.  To reduce the
/// burden on the programmer, you can derive your abstract class from
/// \link reranker::FactoryConstructible FactoryConstructible
/// \endlink, which implements both methods to do nothing.  (All of
/// the abstract base classes that can be constructed via Factory in
/// the Reranker Framework already do this.) For most concrete
/// subtypes, most of the work of initialization is done inside the
/// factory to initialize registered data members, handled by the
/// class&rsquo;s <tt>RegisterInitializers(Initializers&)</tt> method.
/// The implementation of this method generally contains a set of
/// invocations to the various <tt>Add</tt> methods of the \link
/// reranker::Initializers Initializers \endlink class,
/// &ldquo;registering&rdquo; each variable with a name that will be
/// recognized by the \link reranker::Factory Factory \endlink when it
/// parses the specification string. When member
/// initializations are added to an \link reranker::Initializers
/// Initializers \endlink instance, they are optional by default.  By
/// including a third argument that is <tt>true</tt>, one may specify
/// a member whose initialization string <i>must</i> appear within the
/// specification.  If it does not contain it, a runtime error will be
/// raised.
///
/// For completeness, post&ndash;member-initialization may be
/// performed by the class&rsquo;s <tt>Init(const string &)</tt>
/// method, which is guaranteed to be invoked with the complete string
/// that was parsed by the \link reranker::Factory Factory\endlink.
/// The code executed by a class&rsquo; <tt>Init(cosnt string &)</tt>
/// method is very much akin to the <i>declaration phase</i> of a
/// C++ constructor, because it is the code that gets executed just
/// after the members have been initialized.
///
/// For example, \link reranker::FeatureExtractor FeatureExtractor
/// \endlink instances are \link reranker::Factory
/// Factory\endlink-constructible, and so the \link
/// reranker::FeatureExtractor FeatureExtractor \endlink class ensures
/// its concrete subclasses have a \link
/// reranker::FeatureExtractor::RegisterInitializers
/// RegisterInitializers \endlink method and an \link
/// reranker::FeatureExtractor::Init Init \endlink method by being a
/// subclass of \link reranker::FactoryConstructible\endlink. As we
/// saw above, \link reranker::NgramFeatureExtractor
/// NgramFeatureExtractor \endlink is a concrete subtype of \link
/// reranker::FeatureExtractor FeatureExtractor\endlink .  That class
/// has two data members that can be initialized by a factory, one
/// required and one optional.  To show you how easy it is to
/// &ldquo;declare&rdquo; data members that need initialization, here
/// is the exact code from the \link
/// reranker::NgramFeatureExtractor::RegisterInitializers
/// NgramFeatureExtractor::RegisterInitializers \endlink method:
/// \code
/// virtual void RegisterInitializers(Initializers &initializers) {
///   bool required = true;
///   initializers.Add("n",      &n_, required);
///   initializers.Add("prefix", &prefix_);
/// }
/// \endcode
/// The above code says that the \link reranker::NgramFeatureExtractor
/// NgramFeatureExtractor \endlink has a data member <tt>n_</tt>,
/// which happens to be an <tt>int</tt>, that is required to be
/// initialized when an \link reranker::NgramFeatureExtractor
/// NgramFeatureExtractor \endlink instance is constructed by a \link
/// reranker::Factory Factory\endlink, and that the name of this
/// variable will be <tt>&quot;n&quot;</tt> as far as the factory is
/// concerned. It also says that it has a data member
/// <tt>prefix_</tt>, which happens to be of type <tt>string</tt>,
/// whose factory name will be <tt>&quot;prefix&quot;</tt>, and that
/// is not required to be present in a <i>specification string</i> for
/// an \link reranker::NgramFeatureExtractor
/// NgramFeatureExtractor\endlink.
///
/// \subsection appendix_language The Factory language
///
/// As we&rsquo;ve seen, the language used to instantiate objects is
/// quite simple.  An object is constructed via a <i>specification
/// string</i> of the following form: \code
/// RegisteredClassName(member1(init1), member2(init2), ...) \endcode
/// where <tt>RegisteredClassName</tt> is the concrete subtype&rsquo;s
/// name specified with the \link REGISTER_NAMED \endlink macro (or,
/// more likely, one of the convenience macros that is
/// &ldquo;implemented&rdquo; in terms of the \link REGISTER_NAMED
/// \endlink macro, such as \link REGISTER_MODEL \endlink or \link
/// REGISTER_FEATURE_EXTRACTOR\endlink).  The comma-separated list
/// inside the outermost set of parentheses is the set of <i>member
/// initializations</i>, which looks, as we saw \ref
/// appendix_cpp_example "above", intentionally similar to the format
/// of a C++ constructor&rsquo;s initialization phase.  The names of
/// class members that can be initialized are specified via repeated
/// invocations of the various overloaded \link reranker::Initializers
/// \endlink <tt>Add</tt> methods. There is essentially one
/// <tt>Add</tt> method per primitive C++ type, as well as an
/// <tt>Add</tt> method for \link reranker::Factory
/// Factory\endlink-constructible types.
///
/// If you love Backus-Naur Form specifications, please see the
/// documentation for the \link reranker::Factory::CreateOrDie
/// Factory::CreateOrDie \endlink method for the formal description of
/// the grammar for specification strings.
///
/// To continue our example with \link reranker::NgramFeatureExtractor
/// NgramFeatureExtractor \endlink, the following are all legal
/// specification strings for constructing \link
/// reranker::NgramFeatureExtractor NgramFeatureExtractor \endlink
/// instances:
/// \code
/// NgramFeatureExtractor(n(3))
/// NgramFeatureExtractor(n(2), prefix("foo:"))
/// NgramFeatureExtractor(prefix("bar"), n(4))
/// NgramFeatureExtractor(n(2),)
/// \endcode
/// As you can see, the order of member initializers is not important
/// (because each has a unique name), and you can optionally put a
/// comma after the last initializer.  The following are
/// <i><b>illegal</b></i> specification strings for \link
/// reranker::NgramFeatureExtractor NgramFeatureExtractor \endlink
/// instances:
/// \code
/// // Illegal specification strings:
/// NgramFeatureExtractor(prefix("foo"))
/// NgramFeatureExtractor()
/// NgramFeatureExtractor(n(3), prefix(4))
/// \endcode
/// In the first two cases, the specification strings are missing the required
/// variable <tt>n</tt>, and in the final case, the optional <tt>prefix</tt>
/// member is being initialized, but with an <tt>int</tt> literal instead of
/// a <tt>string</tt> literal.
///
/// In most cases, you will never need to directly use a \link
/// reranker::Factory Factory \endlink instance, but they are often at
/// work behind the scenes.  For example, every \link reranker::Model
/// Model \endlink instance uses a factory to construct its internal
/// \link reranker::Candidate::Comparator Candidate::Comparator
/// \endlink instances that it uses to determine the
/// &ldquo;gold&rdquo; and top-scoring candidates when training.  In
/// fact, the <i>specification strings</i> for constructing \link
/// reranker::Model Model \endlink instances reveal how an
/// <tt>init_string</tt> can itself contain other <i>specification
/// strings</i>.  For example, to construct a \link
/// reranker::PerceptronModel PerceptronModel \endlink instance with a
/// \link reranker::DirectLossScoreComparator
/// DirectLossScoreComparator \endlink, you&rsquo;d use the following
/// specification string:
/// \code PerceptronModel(name("MyPerceptronModel"), score_comparator(DirectLossScoreComparator())) \endcode
///
/// The first member initialization, for the member called
/// <tt>name</tt>, specifies the unique name you can give to each
/// \link reranker::Model Model \endlink instance (which is strictly
/// for human consumption).  The second member initialization, for the
/// member called <tt>score_comparator</tt>, overrides the default
/// \link reranker::Candidate::Comparator Candidate::Comparator
/// \endlink used to compare candidate scores, and illustrates how
/// this simple language is recursive, in that specification strings
/// may contain other specification strings for other
/// Factory-constructible objects.
///
///
/// \subsection appendix_putting_together Putting it all together
///
/// Here is a template illustrating how one creates a \link
/// reranker::Factory Factory \endlink for an abstract base class
/// called &ldquo;<tt>Abby</tt>&rdquo; and declares a concrete subtype
/// &ldquo;<tt>Concky</tt>&rdquo; to that Factory.  Most users
/// of the Reranker Framework are likely only to build concrete
/// subtypes of abstract classes that already have factories, and so
/// those users can safely ignore the <tt>abby.H</tt> and <tt>abby.C</tt>
/// files.
/// <ul>
///   <li> <tt>abby.H</tt>
/// \code
/// #include "factory.H"
/// class Abby : public FactoryConstructible {
///   // .. the code for Abby ...
/// };
/// #define REGISTER_NAMED_ABBY(TYPE,NAME) REGISTER_NAMED(TYPE,NAME,Abby)
/// #define REGISTER_ABBY(TYPE) REGISTER_NAMED_ABBY(TYPE,TYPE)
/// \endcode
///   <li> <tt>abby.C</tt>
/// \code
/// IMPLEMENT_FACTORY(Abby)
/// \endcode
///   <li> <tt>concky.H</tt>
/// \code
/// #include "abby.H"
/// class Concky : public Abby {
///  public:
///    virtual void RegisterInitializers(Initialiizers &initializers) {
///       // various calls to the overloaded Initializers::Add methods,
///       // one per data member that the Factory can initialize
///    }
/// };
/// \endcode
///   <li> <tt> concky.C </tt>
/// \code
/// REGISTER_ABBY(Concky)
/// \endcode
/// </ul>
///
/// So what about Greenspun&rsquo;s Tenth Rule?  Well, the idea that
/// initialization strings can themselves contain specification
/// strings suggests that there is a full-blown language being
/// interpreted here, complete with a proper tokenizer and a
/// recursive-descent parser.  There is.  It is a simple language, and
/// one that is formally specified.  To the extent that it mirrors the
/// way C++ does things, it is not quite <i>ad hoc</i>; rather, it is
/// (close to being) an exceedingly small subset of C++ that can be
/// executed dynamically.  We <i>hope</i> it is not bug-ridden, but
/// we&rsquo;ll let you, the user, be the judge of that.
