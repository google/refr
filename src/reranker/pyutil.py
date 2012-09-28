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
# File: khutil
# Version ID: $Id: khutil.py,v 1.8 2007/11/16 21:35:40 khall Exp $
#   various utility functions
#
#-----------------------------------------------------------------------
import subprocess,os,sys,math 
DEBUG = 7

def checkFile(fileName):
  if (not os.path.isfile(fileName)):
    sys.stderr.write('ERROR: file not found: ' + fileName + '\n')
    sys.exit(10)
  return

def runCommand(sysCommand):
  printDebug(7, "Executing command: " + sysCommand + "\n")
  return os.system(sysCommand)

def readCommand(sysCommand, outList):
  printDebug(7, "Executing command: " + sysCommand + "\n")
  syspipe = os.popen(sysCommand)
  for sysLine in syspipe:
    outList.append(sysLine .strip())
  return (syspipe.close() == None)

def writeCommand(sysCommand, toWrite):
  printDebug(7, "Writing " + toWrite + "\n   to: " + sysCommand + "\n")
  newfsmpipe = os.popen(sysCommand, 'w')
  newfsmpipe.write(toWrite.strip() + "\n")
  newfsmpipe.close()

def rwCommand(sysCommand, toWrite, outList):
  printDebug(7, "Reading output from writing " + toWrite + "\n   to: " + sysCommand + "\n")
  (pin, pout) = os.popen2(sysCommand)
  pin.write(toWrite.strip() + "\n")
  pin.close()
  for sysLine in pout:
    outList.append(sysLine.strip())
  pout.close()
  return

def printDebug(debLevel, errStr):
  if (DEBUG >= debLevel):
    sys.stderr.write(errStr)
  return

def printInfo(errStr):
  sys.stderr.write("INFO: " + errStr + "\n")

def printWarn(errStr):
  sys.stderr.write("WARN: " + errStr + "\n")

def printError(errorCode, errStr):
  sys.stderr.write("ERROR: " + errStr + "\n")
  sys.exit(errorCode)

def logSum(x, y):
  if (x < (y - math.log(1e200))):
    return y
  if (y < (x - math.log(1e200))):
    return x
  logdiff = x - y
  # make sure the difference is not too large
  if (x > y and logdiff > x):
    return x
  if (y > x and logdiff > y):
    return y
  #otherwise return the actual sum
  return y + math.log(1.0 + math.exp(logdiff))

# A class which let's us start up a command, send requests and read replies
# via stdio.
class CommandIO:
  def __init__(self, sysCommand):
    printDebug(7, "Starting up process: " + sysCommand)
    self.evalp_ = subprocess.Popen(sysCommand, shell=True, bufsize=0,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   close_fds=True)
    if self.evalp_.poll():
      printError(100, "Command " + sysCommand + " did not start up correctly.")

  def sendreceive(self, toWrite):
    self.evalp_.stdin.write(toWrite.strip() + "\n")
    return self.evalp_.stdout.readline().strip()
