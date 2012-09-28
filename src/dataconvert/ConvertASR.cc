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
#include "ConvertASR.h"

using namespace std;
const char* SPACE_DELIM = " 	";

// Compute the BLEU loss given the candidate.
bool ConvertASR::ComputeBLEU(const string& lstring,
                             confusion_learning::CandidateMessage* cand) {
  CDEBUG(4, "Processing loss: " << lstring);
  double overlap[4];
  double ref_len = numeric_limits<double>::min();
  stringstream loss_stream(lstring);
  loss_stream >> ws >> overlap[0];
  loss_stream >> ws >> overlap[1];
  loss_stream >> ws >> overlap[2];
  loss_stream >> ws >> overlap[3];
  loss_stream >> ws >> ref_len >> ws;
  if (ref_len == numeric_limits<double>::min()) {
    cerr << "BLEU loss requires 7 fields in input file" << endl;
    return false;
  }
  // Compute the length of the candidate string.
  const string& candidate = cand->raw_data();
  size_t cur_pos = 0;
  int cand_len = 0;
  while (cur_pos != string::npos) {
    cur_pos = candidate.find_first_of(SPACE_DELIM, cur_pos + 1);
    if (cur_pos != string::npos) {
      cur_pos = candidate.find_first_not_of(SPACE_DELIM, cur_pos + 1);
    }
    cand_len++;
  }
  CDEBUG(5, "Reference length: " << ref_len);
  CDEBUG(5, "Candidate length: " << cand_len);
  // Compute the sentence-level BLEU loss
  double smooth = 1.0;
  double loss = 0.0;
  for (int order = 0; order < 4; order++) {
    double ngorder = overlap[order];
    if (ngorder == 0) {  // smooth sentence level bleu
      smooth *= 0.5;
      ngorder = smooth;
    }
    if (cand_len > order) {
      loss += log(ngorder / static_cast<double>(cand_len - order)) ;
    }
  }
  loss /= 4.0;
  CDEBUG(5, "  BLEU loss with smoothing: " << loss);
  if (cand_len < ref_len && cand_len > 0) {
    // brevity penalty
    loss += 1.0 - (ref_len / static_cast<double>(cand_len));
    CDEBUG(5, "  After adding brefity penalty: " << loss);
  }
  loss = -exp(loss);
  confusion_learning::ScoreMessage* score_msg = cand->add_score();
  score_msg->set_type(confusion_learning::ScoreMessage::LOSS);
  score_msg->set_score(loss);
  CDEBUG(4, "     Candidate loss: " << loss);
  return true;
}

int ConvertASR::SplitLoss(const string& lstring, int prev_index, bool add,
                          confusion_learning::CandidateSetMessage* set) {
  int cur_index;
  int hypix;
  double loss = 0.0;
  if (lstring.empty()) {
    return -1;
  }
  stringstream loss_stream(lstring);
  loss_stream >> ws >> cur_index
              >> ws >> hypix;
  if (loss_stream.eof() || !loss_stream.good()) {
    return -1;
  }
  // Brian indexes from 1.
  cur_index--;
  hypix--;
  // Split the string into the index and score.
  if (prev_index != -1 && prev_index != cur_index) {
    CDEBUG(5, "     ** Storing loss for next instance");
    nextloss_string_ = lstring;
    return cur_index;
  }
  if (bleuloss_) {
    if (!rawtextdata_) {
      cerr << "BLEU loss requires candidate string" << endl;;
      return -1;
    }
  } else {
    double errors = numeric_limits<double>::min();
    double candidate_len = numeric_limits<double>::min();
    loss_stream >> ws >> errors;
    loss_stream >> ws >> candidate_len >> ws;
    if (candidate_len == numeric_limits<double>::min()) {
      loss = errors;
    } else {
      loss = errors / candidate_len;
    }
    CDEBUG(4, "     Loss index: " << cur_index << " hypix: " << hypix
           << " loss: " << loss);
  }
  confusion_learning::CandidateMessage* cand;
  if (add) {
    cand = set->add_candidate();
  } else {
    if (hypix >= set->candidate_size()) {
      cerr << "Too many candidate for loss score";
      return -1;
    }
    cand = set->mutable_candidate(hypix);
  }
  if (bleuloss_) {
    // Delay BLEU loss computation until we see the candidate string.
    string lstring;
    getline(loss_stream, lstring);
    cand->set_raw_data(lstring);
  } else {
    confusion_learning::ScoreMessage* score_msg = cand->add_score();
    score_msg->set_type(confusion_learning::ScoreMessage::LOSS);
    score_msg->set_score(loss);
  }
  return cur_index;
}

bool ConvertASR::AddUpdateLoss(confusion_learning::CandidateSetMessage* set) {
  bool processed = false;
  int cursize = set->candidate_size();
  int prev_index = -1;
  if (!nextloss_string_.empty()) {
    CDEBUG(5, "Loss from previous pass");
    // Add new candidate and process the string.
    prev_index = SplitLoss(nextloss_string_, prev_index, (cursize == 0), set);
    if (prev_index < 0) {
      return false;
    }
    nextloss_string_.clear();
    processed = true;
  }
  while (lossdata_->good() && !lossdata_->eof()) {
    string loss_string;
    getline(*lossdata_, loss_string);
    CDEBUG(7, "Processing next loss: " << loss_string);
    // Split the string into index and score
    int cur_index = SplitLoss(loss_string, prev_index, (cursize == 0), set);
    if (cur_index < 0) {
      return false;
    }
    if (prev_index != -1 && cur_index != prev_index) {
      break;
    }
    prev_index = cur_index;
    processed = true;
  }
  return processed;
}

bool ConvertASR::AddBaseline(const string& bstring,
                             confusion_learning::CandidateMessage* cand) {
  double score;
  stringstream bstream(bstring);
  bstream >> ws >> score;
  CDEBUG(5, "  Adding baseline score: " << score);
  // Split the string into the index and score.
  confusion_learning::ScoreMessage* score_msg = cand->add_score();
  score_msg->set_type(confusion_learning::ScoreMessage::SYSTEM_SCORE);
  score_msg->set_score(score);
  return true;
}

bool ConvertASR::AddRawtext(const string& rstring,
                            confusion_learning::CandidateMessage* cand) {
  // Split the string into the index and score.
  if (bleuloss_) {
    CDEBUG(5, "  Adding raw text & computing BLEU: " << rstring);
    const string loss_string = cand->raw_data();
    cand->set_raw_data(rstring);
    return ComputeBLEU(loss_string, cand);
  }
  CDEBUG(5, "  Adding raw text: " << rstring);
  cand->set_raw_data(rstring);
  return true;
}

bool ConvertASR::AddReference(const string& rstring,
                              confusion_learning::CandidateSetMessage* set) {
  // Split the string into the index and score.
  int epos = 0;

  if (asrref_) {
    epos = rstring.find_first_of(SPACE_DELIM);
    epos = rstring.find_first_of(SPACE_DELIM, epos + 1);
    epos = rstring.find_first_of(SPACE_DELIM, epos + 1);
    epos ++;
    const string& key = rstring.substr(0, epos);
    set->set_source_key(key);
  }
  const string& reference = rstring.substr(epos , rstring.length() - epos);
  set->set_reference_string(reference);
  CDEBUG(5, "Read ref text: " << reference);
  return true;
}

bool ConvertASR::AddFeature(const string& name_prefix,
                            const string& fstring,
                            confusion_learning::CandidateMessage* cand) {
  // Process the data in this stream.
  size_t cur_pos = 0;
  while (cur_pos < fstring.size()) {
    cur_pos = fstring.find_first_not_of(SPACE_DELIM, cur_pos);
    if (cur_pos == string::npos) {
      break;
    }
    size_t end_pos = fstring.find_first_of(SPACE_DELIM, cur_pos);
    if (end_pos == string::npos) {
      end_pos = fstring.length();
    }
    CDEBUG(5, "Processing feature: " << fstring.substr(cur_pos, end_pos - cur_pos));
    size_t epos = fstring.rfind('=', end_pos);
    if (epos == string::npos) {
      epos = end_pos;
    }
    string feature_string = name_prefix + "_" + fstring.substr(cur_pos, epos - cur_pos);
    cur_pos = epos + 1;
    confusion_learning::FeatureMessage* feature = cand->mutable_feats()->add_feature();
    feature->set_name(feature_string);
    if (epos != end_pos && (cur_pos >= fstring.size() || fstring[cur_pos] != ' ')) {
      // Read the value 
      double value = atof(fstring.substr(cur_pos, end_pos - cur_pos).c_str());
      if (fabs(value - 1.0) > 1e-10) {
        feature->set_value(value);
      }
    }
    CDEBUG(5, "      Adding feature: " << feature_string);
    cur_pos = end_pos + 1;
  }
  return true;
}

bool ConvertASR::AddData(confusion_learning::CandidateSetMessage* set) {
  // Add reference for this example.
  bool processed = false;
  CDEBUG(5, "Adding data...");
  if (refdata_ != NULL && refdata_->good() && !refdata_->eof()) {
    string reference;
    getline(*refdata_, reference);
    CDEBUG(5, "Reading next reference: " << reference);
    processed = AddReference(reference, set);
  }
  if (lossdata_ != NULL) {
    CDEBUG(5, "Reading loss and updating");
    processed = AddUpdateLoss(set);
  }
  for (int cur_index = 0; cur_index < set->candidate_size(); ++cur_index) {
    CDEBUG(5, "Adding data to candidate: " << cur_index);
    confusion_learning::CandidateMessage* cand = set->mutable_candidate(cur_index);
    if (baselinedata_ && baselinedata_->good() && !baselinedata_->eof()) {
      string baseline_string;
      getline(*baselinedata_, baseline_string);
      processed = AddBaseline(baseline_string, cand);
    }
    if (rawtextdata_ && rawtextdata_->good() && !rawtextdata_->eof()) {
      string rawtext_string;
      getline(*rawtextdata_, rawtext_string);
      processed = AddRawtext(rawtext_string, cand);
    }
    // Process all features
    for (size_t curfn = 0; curfn < featdata_.size(); ++curfn) {
      CDEBUG(5, "Reading feature type: " << featnames_[curfn]);
      if (featdata_[curfn] && featdata_[curfn]->good() &&
          !featdata_[curfn]->eof()) {
        string fstring;
        getline(*featdata_[curfn], fstring);
        processed = AddFeature(featnames_[curfn], fstring, cand);
      }
    }
  }
  return processed;
}
