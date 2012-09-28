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
/// Provides the implementation of the reranker::NgramFeatureExtractor class.
/// \author dbikel@google.com (Dan Bikel)

#include "ngram-feature-extractor.H"

namespace reranker {

using std::stringstream;

REGISTER_FEATURE_EXTRACTOR(NgramFeatureExtractor)

void
NgramExtractor::Extract(const vector<string> &tokens,
                        const int n,
                        const string &prefix,
                        FeatureVector<string,double> &symbolic_features) const {
  int tokens_len = (int)tokens.size();
  int last_token_index = tokens_len - 1;
  for (int i = 1; i < tokens_len; ++i) {
    int max_prev = i;
    for (int prev_index = ((i - n + 1) < 0) ? 0 : (i - n + 1);
         prev_index <= max_prev;
         ++prev_index) {
      // No need to output a feature consisting solely of "</s>" token.
      if (max_prev == last_token_index && max_prev == prev_index) {
        break;
      }
      stringstream symbol_ss;
      if (prefix.empty()) {
        symbol_ss << n << "g_ng{";
      } else {
        symbol_ss << prefix << "{";
      }
      for (int j = prev_index; j <= max_prev; ++j) {
        symbol_ss << tokens[j] << ((j < max_prev) ? "," : "}");
      }
      symbolic_features.IncrementWeight(symbol_ss.str(), 1.0);
    }
  }
}

void
NgramFeatureExtractor::ExtractSymbolic(Candidate &candidate,
                                       FeatureVector<string,double> &
                                       symbolic_features) {
  vector<string> tokens;
  tokens.push_back("<s>");
  tokenizer_.Tokenize(candidate.raw_data(), tokens);
  tokens.push_back("</s>");
  ngram_extractor_.Extract(tokens, n_, prefix_, symbolic_features);
}


}  // namespace reranker
