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

#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <math.h>
#include "../proto/data.pb.h"
#include "../proto/dataio.h"
#include "../utils/kdebug.h"
#include "ConvertMT.h"

namespace confusion_dataconvert {

const char* FIELD_DELIM = "|||";
const char* SPACE_DELIM = " 	";

ConvertMT::ConvertMT() {
  dname_.push_back("distort1");
  dname_.push_back("distort2");
  dname_.push_back("distort3");
  dname_.push_back("distort4");
  dname_.push_back("distort5");
  dname_.push_back("distort6");
  dname_.push_back("distort7");
  tmname_.push_back("trans1");
  tmname_.push_back("trans2");
  tmname_.push_back("trans3");
  tmname_.push_back("trans4");
  tmname_.push_back("trans5");
}

int ConvertMT::AddFeature(const string& input, int prev_pos, const string& feat_name,
                          confusion_learning::FeatureVecMessage* featvec) {
  int cur_pos = input.find_first_not_of(SPACE_DELIM, prev_pos);
  int end_pos = input.find_first_of(string(SPACE_DELIM) + string(FIELD_DELIM), cur_pos);
  float new_val = atof(input.substr(cur_pos, end_pos - cur_pos).c_str());
  if (fabs(new_val) > 1e-10) {
    confusion_learning::FeatureMessage* feature = featvec->add_feature();
    feature->set_name(feat_name);
    feature->set_value(atof(input.substr(cur_pos, end_pos - cur_pos).c_str()));
    CDEBUG(4, "Adding feature: " << feat_name << " with value: "
           << feature->value());
  }
  return end_pos;
}

// This should probably be a hadoop mapper class.
bool ConvertMT::ConvertCandidate(const string& input, confusion_learning::CandidateMessage* hyp,
                                 string* candidate_key) {
  // Split the string on '|||' substrings.
  if (hyp == NULL) {
    cerr << "ConvertMTCandidate passed a NULL";
    return false;
  }
  CDEBUG(2, "Processing candidate: " << input);
  size_t prev_pos = 0;
  int field_ix;
  for (field_ix = 0; prev_pos != string::npos && field_ix < 4; ++field_ix) {
    // Skip any white-space.
    CDEBUG(3, "Processing field: " << field_ix);
    prev_pos = input.find_first_not_of(SPACE_DELIM, prev_pos);
    switch (field_ix) {
    case 0:
      {
      int space_pos = input.find_first_of(
          string(SPACE_DELIM) + string(FIELD_DELIM), prev_pos);
      CDEBUG(3, "Looking for space in exmpale index, position: "
             << space_pos << "(starting pos:" << prev_pos <<")");
      *candidate_key = input.substr(prev_pos, space_pos - prev_pos);
      prev_pos = input.find(FIELD_DELIM, space_pos) + 3;
      CDEBUG(3, "Index of candidate: " << *candidate_key);
      }
      break;
    case 1:
      {
      // Raw string
      int delim_pos = input.find(FIELD_DELIM, prev_pos);
      CDEBUG(3, "Looking for delimiter in exmpale index, position: "
             << delim_pos << "(starting pos:" << prev_pos <<")");
      int last_char_pos = input.find_last_not_of(SPACE_DELIM, delim_pos - 1);
      hyp->set_raw_data(input.substr(prev_pos, last_char_pos - prev_pos + 1));
      prev_pos = delim_pos + 3;  // skip delim.
      CDEBUG(3, "Raw String: " << hyp->raw_data());
      }
      break;
    case 2:
      // MT Model Features
      prev_pos = input.find_first_not_of(SPACE_DELIM, prev_pos);
      while (input.substr(prev_pos, 3) != "|||") {
        int space_pos = input.find_first_of(string(SPACE_DELIM) + string(FIELD_DELIM), prev_pos);
        string feature_type = input.substr(prev_pos, space_pos - prev_pos);
        CDEBUG(3, "Feature type: " << feature_type); 
        prev_pos = space_pos;
        if (feature_type == "d:") {
          for (int dist_ix = 0; dist_ix < 7; ++dist_ix) {
            prev_pos = AddFeature(input, prev_pos, dname_[dist_ix],
                                  hyp->mutable_feats());
          }
        } else if (feature_type == "tm:") {
          for (int tmix = 0; tmix < 5; ++tmix) {
            prev_pos = AddFeature(input, prev_pos, tmname_[tmix],
                                  hyp->mutable_feats());
          }
        } else if (feature_type == "lm:") {
            prev_pos = AddFeature(input, prev_pos, "lm",
                                  hyp->mutable_feats());
        } else if (feature_type == "w:") {
            prev_pos = AddFeature(input, prev_pos, "wordcount",
                                  hyp->mutable_feats());
        }
        prev_pos = input.find_first_not_of(SPACE_DELIM, prev_pos);
      }
      prev_pos = input.find(FIELD_DELIM, prev_pos) + 3;
      break;
    case 3:
      prev_pos = AddFeature(input, prev_pos, "fullmodel",
                            hyp->mutable_feats());
      break;
    }
  }
  return true;
}

};
