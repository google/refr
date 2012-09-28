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

#ifndef DATACONVERT_ASRConverter_H_
#define DATACONVERT_ASRConverter_H_

#include <iostream>
#include <string>
#include "../proto/data.pb.h"

using namespace std;

class ConvertASR {
 public:
  ConvertASR(bool bleuloss = false, bool asrref = false) 
    : bleuloss_(bleuloss), asrref_(asrref), baselinedata_(NULL), lossdata_(NULL),
      rawtextdata_(NULL), refdata_(NULL) {
  }
  bool AddData(confusion_learning::CandidateSetMessage* set);

  void set_baselinedata(istream* new_stream) { baselinedata_ = new_stream; }
  void set_lossdata(istream* new_stream) { lossdata_ = new_stream; }
  void set_rawtextdata(istream* new_stream) { rawtextdata_ = new_stream; }
  void set_refdata(istream* new_stream) { refdata_ = new_stream; }
  void add_featdata(const string& name, istream* new_stream) {
    featdata_.push_back(new_stream);
    featnames_.push_back(name);
  }
  void set_bleuloss(void) { bleuloss_ = true; }
  void set_asrref(void) { asrref_ = true; }
 protected:
  bool AddFeature(const string& name_prefix,
                  const string& fstring,
                  confusion_learning::CandidateMessage* cand);

  int SplitLoss(const string& bstring, int prev_index, bool add,
                confusion_learning::CandidateSetMessage* set);
  bool AddUpdateLoss(confusion_learning::CandidateSetMessage* set);
  bool AddBaseline(const string& bstring,
                   confusion_learning::CandidateMessage* cand);
  bool AddRawtext(const string& tstring,
                  confusion_learning::CandidateMessage* cand);
  bool AddReference(const string& rstring,
                    confusion_learning::CandidateSetMessage* set);
  bool AddFeatures(const string& bstring,
                   confusion_learning::CandidateSetMessage* set);
  bool ComputeBLEU(const string& lstring,
                   confusion_learning::CandidateMessage* cand);


 private:
  bool bleuloss_;
  bool asrref_;
  istream* baselinedata_;
  istream* lossdata_;
  istream* rawtextdata_;
  istream* refdata_;
  vector<istream*> featdata_;
  vector<string> featnames_;
  string nextloss_string_;
};

#endif
