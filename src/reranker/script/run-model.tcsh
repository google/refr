#!/bin/tcsh -f

# First, figure out where this script is.
set scriptDir = `dirname $0`
set scriptDir = `cd $scriptDir; echo $cwd`

# Set parentDir and binDir.
set parentDir = `cd $scriptDir/..; echo $cwd`
set binDir = $parentDir/bin

set runModelCmd = $binDir/run-model

set trainingDir = /export/ws11/cslmmtsr/dbikel/ASRData/TrigramData

set devtestDir = $trainingDir
set suffix = 100best.proto.gz

set modelFile = model.gz

set modelConfigString = "PerceptronModel(MyPerceptronModel)"
set tmpModelConfigFilename = $parentDir/config/tmp-model.$$.config
echo $modelConfigString > $tmpModelConfigFilename

set trainingFiles = ($trainingDir/16.fsh2.$suffix \
                     $trainingDir/{17,18,19}.swb.$suffix)

set devtestFiles = ($devtestDir/16.swb.$suffix)

# Set the following to a positive, non-zero integer to limit the
# number of training examples, i.e., utterances or candidate sets,
# read from *each* input file.
set maxExamples = -1

# Set the following to a positive, non-zero integer to specify the
# maximum number of candidates per problem instance (utterance or
# sentence).  E.g., even if your files contain 100-best output, if you
# set this to 10, you'll read only 10-best.
set maxCandidates = -1

set streaming = ""
# Uncomment the next line to enable training in streaming mode.
# set streaming = "-streaming"

set featureExtractorConfig = $parentDir/config/default.config

# Now, train the model!
set cmd = ($runModelCmd -m $modelFile --model-config  $tmpModelConfigFilename \
           -t $trainingFiles -d $devtestFiles \
           -c $featureExtractorConfig $streaming \
           --max-examples $maxExamples --max-candidates $maxCandidates)
echo "Executing command:"
echo $cmd
$cmd
/bin/rm -f $tmpModelConfigFilename
