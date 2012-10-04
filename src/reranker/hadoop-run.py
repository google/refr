#!/usr/bin/python
#-----------------------------------------------------------------------
# Copyright 2012, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Google Inc. nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,           
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
## @file hadoop-run.py
#  A python program which will train a reranking model on a Hadoop cluster using
#  the Iterative Parameter Mixtures perceptron training algorithm.
#
#  You must first have a Hadoop account configured.  In order to train, you will
#  need to have the following:
#   - Training data locally accessible (accessible by the script)
#   - A HadoopFS (HDFS) directory with enough space to store the input
#   training data, the intermediate models and the final model.
#
#  The program will attempt to locate the Hadoop binary and the
#  Hadoop streaming library.  If this fails, you can specify these 
#  via command-line parameters (--hadooproot and --streamingloc).
#
#  Usage:
#   hadoop-run.py --input InputData --hdfsinputdir HDFSIndir \\
#                 --hdfsoutputdir HDFSOutDir --outputdir OutputDir
#
#   InputData - A comma-separated list of file globs containing the training data.
#               These must be accessible by script.
#   OutputDir - The local directory where the trained model(s) are written.  The
#               default model name is 'model'.  You can change this using the
#               --modelname command-line parameter.
#   HDFSInDir - A directory on HDFS where the input data will be copied to.
#   HDFSOutDir - A directory on HDFS where the temporary data and output data
#                will be written to.
#                The final models are copied to the locally-accessible OutputDir.
#
# Check  input command line options.
#  @author kbhall@google.com (Keith Hall)
#

from optparse import OptionParser
import os, sys, re, gzip, glob, signal, atexit, operator, random
import codecs
import pyutil
import hadooputil
import defs

##
#  The following arguments are available to hadoop-run.py
#
#  @param[in] hadooproot Location of hadoop installation.
#  @param[in] refrbin Location of the Reranker Framework bin directory.
#  @param[in] develdata Location of development data.
#  @param[in] input Location of input data on local FS.
#  @param[in] hdfsinputdir Location of input data on HDFS.
#  @param[in] hdfsoutputdir Output directory (on HDFS) - will be removed before each iteration.
#  @param[in] outputdir Output directory.
#  @param[in] inputmodel Name of model to start with.
#  @param[in] inputmodeliter Iteration number of input model (will start with next iteration).
#  @param[in] modelname Name of model file (new models written to --outputdir).
#  @param[in] maxiter Maximum number of iterations to run.
#  @param[in] numreducer Number of reducers.
#  @param[in] streamingloc Location under hadooproot for streaming jar file.
#  @param[in] libpath Specify the LD_LIBRARY_PATH for jobs run on Hadoop.
#  @param[in] splitsize Min size f each data split.
#  @param[in] tasktimeout Amount of time (seconds) for task to run
#                         (e.g., loading mode) before processing the next input record.
#  @param[in] force Force all data processing even if files exist.
#  @param[in] forcecompile Force precomilation if applicable.
#  @param[in] compilefeatures Compile features before processing.
#  @param[in] maxdecline Number of iterations in decline before stopping
#  @param[in] model-config Model configuration file
#  @param[in] train-config Feature extractor configuration file for training
#  @param[in] dev-config Feature extractor configuration file for dev

optParse = OptionParser()
optParse.add_option("-H", "--hadooproot", dest="hadooproot",
                    help = "Location of hadoop installation.  If not set, " +
                           "the script will attempt to find it.",
                    default = "")
optParse.add_option("--refrbin", dest="refrbin",
                    help = "Location of the Reranker Framework (ReFr) bin directory",
                    default = defs.refrbin + "/")
optParse.add_option("-d", "--develdata", dest="develdata",
                    help = "Location of development data")
optParse.add_option("-i", "--input", dest="inputlist",
                    help = "Location of input data on local FS",
                    action = "append")
optParse.add_option("-I", "--hdfsinputdir", dest="hdfsinputdir",
                    help = "Location of input data on HDFS")
optParse.add_option("-O", "--hdfsoutputdir", dest="hdfsoutputdir",
                    help = "Output directory (on HDFS) - will be removed before each iteration")
optParse.add_option("-o", "--outputdir", dest="outputdir", help = "Output directory ")
optParse.add_option("-M", "--inputmodel", dest="inputmodel",
                    help = "name of model to start with")
optParse.add_option("-S", "--inputmodeliter", dest="startiter",
                    help = "Iteration number of input model (will start with next iteration)",
                    default = 0)
optParse.add_option("-m", "--modelname", dest="modelname",
                    help = "name of model file (new models written to --outputdir)",
                    default = "model")
optParse.add_option("--maxiter", dest="maxiter",
                    help = "maximum number of iterations to run", default = 100)
optParse.add_option("--numreducer", dest="numreducer",
                    help = "Number of reducers.", default = 1)
optParse.add_option("--streamingloc", dest="streamingloc",
                    help = "Location streaming jar file.  " +
                           "An empty string will force the script to attempt to find the streaming jar file.",
                    default = "")
#                    default = "contrib/streaming/hadoop-0.20.2-streaming.jar")
optParse.add_option("--libpath", dest="libpath",
                    help = "Specify the LD_LIBRARY_PATH",
                    default = "/usr/local/lib:")
optParse.add_option("--splitsize", dest="minsplitsize",
                    help = "Min size of each data split",
                    default = 0)
optParse.add_option("--tasktimeout", dest="tasktimeout",
                    help = "Amount of time (seconds) for task to run (e.g., loading mode) " +
                           " before processing the next input record",
                    default = 0)
optParse.add_option("--force", dest="force",
                    help = "Force all data processing even if files exist",
                    action = "store_true",
                    default = False)
optParse.add_option("--forcecompile", dest="forcecompile",
                    help = "Force precomilation if applicable",
                    action = "store_true",
                    default = False)
optParse.add_option("--compilefeatures", dest="compilefeatures",
                    help = "Compile features before processing",
                    action = "store_true",
                    default = False)
optParse.add_option("--maxdecline", dest="max_num_in_decline",
                    help = "Number of iterations of an increasing loss before we stop training",
                    default = 5)
optParse.add_option("-v", "--verbosity", dest="verbosity",
                    help = "Set the verbosity of the debugging output",
                    default = 0)
optParse.add_option("--no-weighted-loss", dest="weightedloss",
                    help = "Do not use a weighted loss (e.g., when there is no reference)",
                    action = "store_false",
                    default = True)
optParse.add_option("--model-config", dest="modelconfig",
                    help = "Specifies the model configuration file")
optParse.add_option("--train-config", dest="trainconfig",
                    help = "Specifies the feature extractor configuration " +
                           "file for training instances")
optParse.add_option("--dev-config", dest="devconfig",
                    help = "Specifies the feature extractor configuraiton " +
                           "file for devtest instances")
optParse.add_option("--mapperfiles", dest="mapperfiles",
                    help = "A list of files to be passed to the training mapper",
                    action = "append")

(options, args) = optParse.parse_args()

# Check  input command line options.
if (not options.inputlist):
  optParse.error("--input option is required")
if (not options.hdfsinputdir):
  optParse.error("--hdfsinputdir option is required")
if (not options.hdfsoutputdir):
  optParse.error("--hdfsoutputdir option is required")
if (not options.outputdir):
  optParse.error("--outputdir option is required")

pyutil.DEBUG = options.verbosity

# Attempt to find the hadoop installation.
hadooproot = options.hadooproot
if not hadooproot:
  if os.path.isdir("/usr/lib/hadoop"):
    hadooproot = "/usr/lib/hadoop"
  elif os.path.isdir("/usr/local/lib/hadoop"):
    hadooproot = "/usr/local/lib/hadoop"
  elif os.path.isdir("/opt/lib/hadoop"):
    hadooproot = "/opt/lib/hadoop"
  else:
    pyutil.printError(10, "Unable to find the hadoop installation.  " +
                      "Please specify with --hadooproot.")

streamingloc = options.streamingloc
if not streamingloc:
  if os.path.exists(hadooproot + "/hadoop-streaming.jar"):
    streamingloc = hadooproot + "/hadoop-streaming.jar"
  else:
    tmppath = hadooproot + "/contrib/streaming"
    if not os.path.isdir(tmppath):
      pyutil.printError(10, hadooproot + "/contrib/streaming does not exist.  " +
                        "Please specify location of hadoop streaming jar file with " +
                        "--streamingloc")
    streamingjar = glob.glob(tmppath + "/hadoop-streaming*.jar")
    if len(streamingjar) != 1:
      pyutil.printError(10, "Unable to find streaming jar, please specify with --streamingloc")
    streamingloc = streamingjar[0]

# Sanity check of Directories.
if (not os.path.isdir(hadooproot) or
    not os.path.exists(hadooproot + "/bin/hadoop")):
  optParse.error("--hadooproot must be the base directory of the " +
                 "hadoop installation")

if (not os.path.exists(streamingloc)):
  optParse.error("--streamingloc does not specify a valid jar files for the " + 
                 "streaming interface (checked: " + streamingloc)

if (not os.path.isdir(options.refrbin) or
    not os.path.exists(options.refrbin + "/run-model")):
  optParse.error("--refrbin directory must be the Reranker Framework bin " +
                 "direcotry.  Checked: " + options.refrbin)


## Collect input filenames.
filenames = []
for inputstring in options.inputlist:
  for tmpfile in inputstring.split():
    filenames += glob.glob(tmpfile)

for input in filenames:
  pyutil.printInfo("Input file: " + input)
  if (not os.path.exists(input)):
    pyutil.printError(130, "Input file not found: " + input)

if (options.develdata and not os.path.exists(options.develdata)):
  pyutil.printError(131, "Specified devel data file not found: " + options.develdata)

## Create output directory if it does not exist.
if (not os.path.isdir(options.outputdir)):
  os.makedirs(options.outputdir)

## @var hdproc
#  HadoopInterface object used to process all Hadoop MR utils.
hdproc = hadooputil.HadoopInterface(hadooproot,
                                    streamingloc,
                                    options.minsplitsize,
                                    options.tasktimeout, 
                                    options.libpath)

## Configuration for training options
# @var train_map_options
# Options passed to the mapper binary.
train_map_options = ""
# @var train_files
# string containing '-file filename' for all external files.
train_files = ""
if (options.modelconfig):
  train_map_options += " --model-config ./" + os.path.basename(options.modelconfig)
  train_files += " -file " + options.modelconfig
if (options.trainconfig):
  train_map_options += " --train-config ./" + os.path.basename(options.trainconfig)
  train_files += " -file " + options.trainconfig
train_map = ("'" + options.refrbin + "/run-model" + train_map_options +
            " --train - --mapper -m -")


if options.mapperfiles:
  for mapperfile in options.mapperfiles:
    train_files += " -file " + mapperfile

## Shortcuts to command-line programs.
extractsym_map = "'" + options.refrbin + "/compile-features -i -'"
compiledata_map = "'" + options.refrbin + "/compile-features -i - --clear-raw --input-symbols "
train_reduce = options.refrbin + "/model-merge-reducer"
train_recomb = options.refrbin + "/model-combine-shards"
symbol_recomb = options.refrbin + "/model-combine-symbols"
pipeeval_options = ""
if (options.devconfig):
  pipeeval_options = " --dev-config " + options.devconfig
pipeeval = options.refrbin + "/piped-model-evaluator" + pipeeval_options

hadoop_inputfiles = ""
for inputfile in filenames:
  hadoop_inputfiles += hdproc.CheckInputFile(inputfile, options.hdfsinputdir,
                                             options.outputdir, options.force,
                                             True)
                                             #not options.compilefeatures)

precompdevfile = options.develdata

## Precopilation of string features.
#  Optional - reduces the size of the models, but takes time to create initial precompiled data.
#
if (options.compilefeatures):
  pyutil.printInfo("Precompiling feature indices")
  if (options.develdata):
    precompdevfile = options.outputdir + "/"
    precompdevfile += os.path.basename(options.develdata).replace(".gz","")
    precompdevfile += ".compiled.gz"
  symbol_dir = options.hdfsinputdir + "/Symbols/"
  precomp_dir = options.hdfsinputdir + "/Precompiled/"
  precompdev_dir = options.hdfsinputdir + "/PrecompiledDev/"

  # Extract all features.
  if (hdproc.CheckRemoveHDir(precomp_dir, (options.force or options.forcecompile)) or
      options.forcecompile):
    addl_data = ""
    if (options.develdata):
      addl_data = hdproc.CheckInputFile(options.develdata, options.hdfsinputdir,
                                        options.outputdir, options.force,
                                        True)
      pyutil.printInfo("Dev data file: " + addl_data)
    # Copy data to HDFS
    symfile_name = options.outputdir + "/" + options.modelname + ".symbols.gz"
    if (not os.path.exists(symfile_name)):
      hdproc.CheckRemoveHDir(symbol_dir, True)
      hdproc.RunMR(hadoop_inputfiles + addl_data, symbol_dir, 100,
                   "'" + train_reduce +  " -S'", extractsym_map, "")
      # Concatenate symbols to local symbol table.
      hdproc.CatPipe(symbol_dir + "/part-*", symbol_recomb + " -o " + symfile_name)

    # Convert the original input data.
    hdproc.RunMR(hadoop_inputfiles, precomp_dir, 0, "",
                 compiledata_map + "./" + os.path.basename(symfile_name) +
                 "' -file " + symfile_name, "")
    if (options.develdata):
      hdproc.CheckRemoveHDir(precompdev_dir, True)
      hdproc.RunMR(addl_data, precompdev_dir, 0, "",
                   compiledata_map + "./" + os.path.basename(symfile_name) +
                   "' -file " + symfile_name, "")
      hdproc.CatPipe(precompdev_dir + "/part-*", " gzip -c > " + precompdevfile)
      hdproc.CheckRemoveHDir(precompdev_dir, True)
  hadoop_inputfiles = " --input " + precomp_dir

#------------
# Run Hadoop Iterative MapReduce
# (Iterative Parameter Mixtures)
#------------
cur_model = options.inputmodel
converged = False

iteration = int(options.startiter)
prev_loss = -9999
loss_history = []
num_in_decline = 0
best_loss_index = 0
if (options.develdata):
  eval_cmd = pipeeval + " -d " + precompdevfile
  if (not options.weightedloss):
    eval_cmd += " --use-weighted-loss false"
  evalio = pyutil.CommandIO(eval_cmd)

while (not converged and iteration < int(options.maxiter)):
  iteration += 1
  pyutil.printInfo("Training iteration: " + str(iteration))
  # Make sure the output directory is 
  # Run the MapReducer Job
  hdproc.CheckRemoveHDir(options.hdfsoutputdir, True)

  # Create the MR string and run the MR
  iter_str = "'"
  if (cur_model):
    iter_str = " -i ./" + os.path.basename(cur_model) + "' -file " + cur_model

  hdproc.RunMR(hadoop_inputfiles, options.hdfsoutputdir, options.numreducer,
               train_reduce, train_map + iter_str + train_files, "")

  # Copy data form the mapreduce to the local file-system
  model_output = options.outputdir + "/" + options.modelname + "_iter" + str(iteration) + ".gz"
  proc_cmd = train_recomb + " -o " + model_output
  hdproc.CatPipe(options.hdfsoutputdir + "/part-*", proc_cmd)

  devtest_score = 0
  if (options.develdata):
    devtest_score = evalio.sendreceive(model_output)

  loss = 0.0;
  if (devtest_score):
    # Get the score returned on STDOUT
    loss = float(devtest_score)
  if (not loss_history):
    loss_history.append(loss)
    pyutil.printInfo("Loss for iteration " + str(iteration) + ": " + str(loss))
  else:
    diff = loss_history[-1] - loss
    if (loss < loss_history[best_loss_index]):
      # Loss is appended below:
      best_loss_index = len(loss_history)
    if (loss > loss_history[-1]):
      num_in_decline += 1
    else:
      num_in_decline = 0
    # Append loss to end of history.
    loss_history.append(loss)
    pyutil.printInfo("Loss for iteration " + str(iteration) + ": " + str(loss) +
                     " loss-delta: " + str(diff))
    if (num_in_decline < options.max_num_in_decline):
      pyutil.printInfo("Continuing to train as number epochs in decline is: " +
                       str(num_in_decline) + ", which is less than " +
                       str(options.max_num_in_decline))
  # if not done...
  cur_model = model_output
pyutil.printInfo("Best model is from iteration: " + str(best_loss_index + 1) +
                 " with a devset loss of: " + str(loss_history[best_loss_index]))
