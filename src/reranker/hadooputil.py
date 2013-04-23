#!/usr/bin/env python
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
## @file hadooputil.py
#  A set of utilities to help interface with Hadoop.
#

import os,sys, math
import pyutil

## @class HadoopInterface
#  A simple class interface for running hadoop commands.
#
class HadoopInterface:
  def __init__(self, hbasedir, streamingloc, minsplitsize, tasktimeout, libpath):
    self.hadoopmr_ = hbasedir + "/bin/hadoop "
    self.hadoopmr_ += " jar " + streamingloc
    if (int(minsplitsize) > 0):
      self.hadoopmr_ += " -Dmapred.min.split.size=" + str(minsplitsize)
    if (int(tasktimeout) >= 0):
      self.hadoopmr_ += " -Dmapred.task.timeout=" + str(tasktimeout)
    self.hadooplibpath_ = ""
    if (libpath):
      self.hadooplibpath_ = " --cmdenv LD_LIBRARY_PATH=" + libpath + " "
    self.hadoopfs_ = hbasedir + "/bin/hadoop fs "
    self.hadooptest_ = hbasedir + "/bin/hadoop fs -test "
    self.hadoopcat_ = hbasedir + "/bin/hadoop fs -cat " 
    self.hadoopput_ = hbasedir + "/bin/hadoop fs -put "
    self.hadoopmove_ = hbasedir + "/bin/hadoop fs -moveFromLocal "
    self.hadoopget_ = hbasedir + "/bin/hadoop fs -get "
    self.hadoopmkdir_ = hbasedir + "/bin/hadoop fs -mkdir "
    self.hadooprmr_ = hbasedir + "/bin/hadoop fs -rmr "

  ## Check if a directory exists on HDFS.
  #  @param[in] directory name of directory to check
  #  @return True if directory exits.
  def CheckHDir(self, directory):
    test_str = self.hadooptest_ + "-d " + directory
    return (pyutil.runCommand(test_str) == 0)

  ## Function to check for the existence of a directory on the HDFS.
  #  @param[in] directory Direcotry to check.
  #  @param[in] remove Remove the directory if it exists.
  #  @return True if it did not exist or was removed.
  def CheckRemoveHDir(self, directory, remove):
    nodir = False
    if (self.CheckHDir(directory)):
      if (remove):
        rm_str = self.hadooprmr_ + directory
        pyutil.runCommand(rm_str)
        nodir = True
    else:
      nodir = True
    return nodir

  ## Function to check for a file on HDFS.
  #  @param[in] filename The file to check for.
  def CheckHDFSFile(self, filename):
    hdinput_test = self.hadooptest_ + "-e " + filename
    return (pyutil.runCommand(hdinput_test) == 0)

  ## Check for an input file and prepare it for MR processing.
  #  @param[in] inputfile Name of the local file to prepare.
  #  @param[in] hdfsinputdir HDFS directory for data staging.
  #  @param[in] outputdir Local file system directory for the output.
  #  @param[in] force Reprocess data even if files already exist.
  #  @param[in] uncompress Uncompress data, if compressed, before running.
  #
  #  @return A MR input string.
  #  Stage the input data for MapReduce processing.
  #  If we are uncompressing compressed files, then we move uncompressed data
  #  HDFS; otherwise, we simply copy the data to HDFS.
  #
  def CheckInputFile(self, inputfile, hdfsinputdir, outputdir, force, uncompress):
    input_file_list = ""
    if (inputfile.endswith(".gz") and uncompress):
      input_filename = os.path.basename(inputfile).replace(".gz","")
    else:
      input_filename = os.path.basename(inputfile)
    pyutil.printDebug(1, "Processing input " + input_filename + "\n")
    # Copy the input data to HDFS
    # Check that the input data exists and move to HDFS if necessary.
    hdfsinputfile = hdfsinputdir + "/" + input_filename
    if (not self.CheckHDFSFile(hdfsinputfile) or force):
      pyutil.printInfo("Regenerating HDFS input: " + hdfsinputfile)
      if (not self.CheckHDir(hdfsinputdir)):
        pyutil.runCommand(self.hadoopmkdir_ + hdfsinputdir)
      if (inputfile.endswith(".gz") and uncompress):
        new_input = outputdir + "/" + input_filename
        unzipcmd = "gunzip -c " + inputfile + " > " + new_input
        if (pyutil.runCommand(unzipcmd) != 0):
          pyutil.printError(12, "Unable to unzip file: " + inputfile)
        pyutil.runCommand(self.hadoopmove_ + new_input + " " + hdfsinputdir)
        input_file_list += " --input " + hdfsinputdir + "/" + input_filename
      else:
        pyutil.runCommand(self.hadoopput_ + inputfile + " " + hdfsinputdir)
        input_file_list += " --input " + hdfsinputdir + "/" + input_filename
      if (not self.CheckHDFSFile(hdfsinputfile)):
        pyutil.printError(10, "Unable to create input on HDFS: " + hdfsinputfile)
    else:
      input_file_list += " --input " + hdfsinputdir + "/" + input_filename
      pyutil.printDebug(5, "Found file on HDFS: " + hdfsinputdir + "/" + input_filename)
    return input_file_list

  def CatPipe(self, hdfsfiles, pipecmd):
    catcmd = self.hadoopcat_ + " " + hdfsfiles + " | " + pipecmd
    if(pyutil.runCommand(catcmd)):
      pyutil.printError(34, "Error: " + catcmd)

  def CatPipeRead(self, hdfsfiles, pipecmd, retval):
    catcmd = self.hadoopcat_ + " " + hdfsfiles + " | " + pipecmd
    if (not pyutil.readCommand(catcmd, retval)):
      pyutil.printError(30, "Error running command " + catcmd)

  ## RunMR
  #  Run a MapReduce
  #  @param[in] input_files HDFS location of input files.
  #  @param[in] outputdir HDFS location of output.
  #  @param[in] reduce_tasks Number of reducer tasks (0 = use default).
  #  @param[in] reducer Full string of streaming reducer command.
  #  @param[in] mapper Full string of streaming mapper command.
  #  @param[in] mroptions Addition streaming MR options (usually specified with -D).
  def RunMR(self, input_files, outputdir, reduce_tasks, reducer, mapper, mroptions):
    mr_str = self.hadoopmr_
    if (mroptions):
      mr_str += mroptions + " "
    mr_str += self.hadooplibpath_ + input_files
    if not reducer:
      mr_str += " -numReduceTasks 0 --reducer None "
      #mr_str += " -numReduceTasks 100 --reducer cat "
    else:
      if (int(reduce_tasks) >= 0):
        mr_str += " -numReduceTasks " + str(reduce_tasks)
      mr_str += " --reducer " + reducer
    mr_str += " --output " + outputdir
    mr_str += " --mapper " + mapper
    pyutil.printInfo("Running MR on: " + input_files)
    if (pyutil.runCommand(mr_str) != 0):
      pyutil.printError(33, "Error running MR" + mr_str)
