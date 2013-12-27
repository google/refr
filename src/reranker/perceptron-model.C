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
/// \file perceptron-model.C
/// Provides the reranker::PerceptronModel class implementation.
/// \author dbikel@google.com (Dan Bikel)

#define DEBUG 1

#include <iostream>
#include <vector>
#include <unordered_set>

#include "candidate-set.H"
#include "training-time.H"

#include "perceptron-model.H"

using std::cerr;
using std::endl;
using std::vector;
using std::unordered_set;

namespace reranker {

REGISTER_MODEL(PerceptronModel)

REGISTER_NAMED_MODEL_UPDATE_PREDICATE(PerceptronModel::DefaultUpdatePredicate,
                                      PerceptronModelDefaultUpdatePredicate)

REGISTER_NAMED_MODEL_UPDATER(PerceptronModel::DefaultUpdater,
                             PerceptronModelDefaultUpdater)

string
PerceptronModel::proto_reader_spec_("PerceptronModelProtoReader()");

string
PerceptronModel::proto_writer_spec_("PerceptronModelProtoWriter()");


void
PerceptronModel::RegisterInitializers(Initializers &initializers) {
  bool required = true;
  initializers.Add("name", &name_, required);
  initializers.Add("score_comparator", &score_comparator_);
  initializers.Add("gold_comparator", &gold_comparator_);
  initializers.Add("candidate_set_scorer", &candidate_set_scorer_);
  initializers.Add("update_predicate", &update_predicate_);
  initializers.Add("updater", &updater_);
  initializers.Add("step_size", &step_size_);
}

void
PerceptronModel::Init(const Environment *env, const string &arg) {
  model_spec_.clear();
  model_spec_.append(arg);
}

void
PerceptronModel::Train(CandidateSetIterator &examples,
                       CandidateSetIterator &development_test) {
  while (NeedToKeepTraining()) {
    NewEpoch();
    TrainOneEpoch(examples);
    Evaluate(development_test);
    // TODO(dbikel,kbhall): Iterative parameter mixing goes here.
    //                      Keith: Please note that FeatureVector has
    //                      an AddScaledVector method which is
    //                      probably useful here.
  }
  if (DEBUG) {
    cerr << "Best model epoch: " << best_model_epoch_ << endl;
    cerr << "Total elpased time: " << time_.absolute_seconds() << " seconds."
         << endl;
  }
  if (DEBUG >= 2) {
    cerr << "Final raw model: " << models_.GetModel(true) << endl
         << "Final averaged model: " << models_.GetModel(false) << endl;
    cerr << "Final best raw model: "
         << best_models_.GetModel(true) << endl
         << "Final best averaged model: "
         << best_models_.GetModel(false) << endl;
  }
}

bool
PerceptronModel::NeedToKeepTraining() {
  int num_epochs = time().epoch() + 1;

  if (DEBUG) {
    if (max_epochs() > 0) {
      if (num_epochs < max_epochs()) {
        cerr << "Training because we have trained only " << num_epochs
             << " epochs but max epochs is " << max_epochs() << "." << endl;
      }
      else {
        cerr << "Stopping training because we have trained "
             << num_epochs << " epochs and max epochs is "
             << max_epochs() << "." << endl;
      }
    }
  }

  if (max_epochs() > 0 && num_epochs >= max_epochs()) {
    return false;
  }

  if (DEBUG) {
    if (min_epochs() > 0 && num_epochs < min_epochs()) {
        cerr << "Training because we have trained " << num_epochs
             << " epochs but min epochs is " << min_epochs() << "." << endl;
    }
  }

  if (min_epochs() > 0 && num_epochs < min_epochs()) {
    return true;
  }

  if (DEBUG) {
    if (num_epochs_in_decline_ < max_epochs_in_decline_) {
      cerr << "Training because num epochs in decline is "
           << num_epochs_in_decline_ << " which is less than "
           << max_epochs_in_decline_ << "." << endl;
    } else {
      cerr << "Stopping training because num epochs in decline is "
           << num_epochs_in_decline_ << " which is greater than or equal to "
           << max_epochs_in_decline_ << "." << endl;
    }
  }

  return num_epochs_in_decline_ < max_epochs_in_decline_;
}

void
PerceptronModel::NewEpoch() {
  if (DEBUG) {
    if (time_.epoch() > 0) {
      cerr << "Epoch " << time_.epoch() << ": "
           << time_.seconds_since_last_epoch() << " seconds." << endl;
    }
  }
  time_.NewEpoch();
  num_training_errors_per_epoch_.push_back(0);
}


void
PerceptronModel::TrainOneEpoch(CandidateSetIterator &examples) {
  examples.Reset();
  while (examples.HasNext()) {
    TrainOnExample(examples.Next());
  }
  EndOfEpoch();
}

void
PerceptronModel::EndOfEpoch() {
  models_.UpdateAllFeatureAverages(time_);
  if (end_of_epoch_hook_ != NULL) {
    end_of_epoch_hook_->Do(this);
  }
  if (DEBUG) {
    int num_training_errors_this_epoch =
        *num_training_errors_per_epoch_.rbegin();
    double percent_training_errors_this_epoch =
        ((double)num_training_errors_this_epoch / time_.index()) * 100.0;
    cerr << "Epoch " << time_.epoch() << ": number of training errors: "
         << num_training_errors_this_epoch << " ("
         << percent_training_errors_this_epoch << "%)" << endl;
  }
}


void
PerceptronModel::TrainOnExample(CandidateSet &example) {
  time_.Tick();

  if (symbols_ != NULL) {
    example.CompileFeatures(symbols_);
  }

  bool training = true;
  ScoreCandidates(example, training);

  if (NeedToUpdate(example)) {
    if (DEBUG >= 2) {
      cerr << "Time:" << time_.to_string() << ": need to update because "
           << "best scoring index " << example.best_scoring_index()
           << " is not equal to gold index " << example.gold_index() << endl;
    }
    ++(*num_training_errors_per_epoch_.rbegin());
    ++num_training_errors_;
    Update(example);
  }
}

bool
PerceptronModel::NeedToUpdate(CandidateSet &example) {
  return update_predicate_->NeedToUpdate(this, example);
}

bool
PerceptronModel::DefaultUpdatePredicate::NeedToUpdate(Model *model,
                                                      CandidateSet &example) {
  return example.best_scoring_index() != example.gold_index();
}

void
PerceptronModel::Update(CandidateSet &example) {
  updater_->Update(this, example);
}

void
PerceptronModel::DefaultUpdater::Update(Model *m, CandidateSet &example) {
  PerceptronModel *model = dynamic_cast<PerceptronModel *>(m);
  ++(model->num_updates_);
  unordered_set<int> gold_features;
  unordered_set<int> best_scoring_features;
  model->ComputeFeaturesToUpdate(example, gold_features, best_scoring_features);

  model->models_.UpdateGoldAndCandidateFeatureAverages(model->time_,
                                                       gold_features,
                                                       best_scoring_features);
  double step_size =
      model->ComputeStepSize(gold_features, best_scoring_features, example);

  // Finally, update perceptrons.

  if (DEBUG >= 2) {
    cerr << "Updating weights for gold features [";
    for (unordered_set<int>::const_iterator it = gold_features.begin();
         it != gold_features.end(); ++it) {
      cerr << " " << *it;
    }
    cerr << "] from\n\t" << example.GetGold() << endl;

    cerr << "Updating weights for best scoring features [";
    for (unordered_set<int>::const_iterator it = best_scoring_features.begin();
         it != best_scoring_features.end(); ++it) {
      cerr << " " << *it;
    }
    cerr << "] from\n\t" << example.GetBestScoring() << endl;

  }

  double positive_step = step_size;
  model->models_.UpdateWeights(model->time_, gold_features,
                               example.GetGold().features(), positive_step);
  double negative_step = -step_size;
  model->models_.UpdateWeights(model->time_, best_scoring_features,
                        example.GetBestScoring().features(), negative_step);

  if (DEBUG >=2) {
    cerr << "Raw model: " << model->models_.GetModel(true) << endl;
    cerr << "Avg model: " << model->models_.GetModel(false) << endl;
  }
}

double
PerceptronModel::Evaluate(CandidateSetIterator &development_test) {
  double total_weight = 0.0;
  double total_weighted_loss = 0.0;
  double total_oracle_loss = 0.0;
  double total_baseline_loss = 0.0;
  num_testing_errors_per_epoch_.push_back(0);

  bool not_training = false;
  size_t development_test_size = 0;
  development_test.Reset();
  while (development_test.HasNext()) {
    ++development_test_size;
    CandidateSet &candidate_set = development_test.Next();
    if (symbols_ != NULL) {
      candidate_set.CompileFeatures(symbols_);
    }
    ScoreCandidates(candidate_set, not_training);
    double loss_weight =
        use_weighted_loss() ? candidate_set.loss_weight() : 1.0;
    total_weight += loss_weight;
    total_weighted_loss += loss_weight * candidate_set.GetBestScoring().loss();
    total_oracle_loss += loss_weight * candidate_set.GetGold().loss();

    // For now, assume that the candidate sets are sorted by the baseline score.
    total_baseline_loss += loss_weight * candidate_set.Get(0).loss();
    if (candidate_set.best_scoring_index() != candidate_set.gold_index()) {
      ++(*num_testing_errors_per_epoch_.rbegin());
    }
  }

  double loss_this_epoch = total_weighted_loss / total_weight;
  loss_per_epoch_.push_back(loss_this_epoch);

  int num_testing_errors_this_epoch =
      num_testing_errors_per_epoch_[time_.epoch()];
  double percent_testing_errors_this_epoch =
      ((double)num_testing_errors_this_epoch / development_test_size) * 100.0;
  double oracle_loss = total_oracle_loss / total_weight;
  double baseline_loss = total_baseline_loss / total_weight;
  cerr << "Epoch " << time_.epoch() << ": oracle loss: " << oracle_loss << endl;
  cerr << "Epoch " << time_.epoch() << ": baseline loss: " << baseline_loss << endl;
  cerr << "Epoch " << time_.epoch() << ": average devtest loss: "
       << loss_this_epoch << endl;
  cerr << "Epoch " << time_.epoch() << ": number of testing errors: "
       << num_testing_errors_this_epoch << " ("
       << percent_testing_errors_this_epoch << "%)" << endl;

  if (time_.epoch() == 0 ||
      loss_this_epoch < loss_per_epoch_[best_model_epoch_]) {
    best_models_ = models_;
    best_model_epoch_ = time_.epoch();
  }

  if (time_.epoch() > 0 &&
      time_.epoch() != best_model_epoch_ &&
      loss_this_epoch >= loss_per_epoch_[best_model_epoch_]) {
    ++num_epochs_in_decline_;
  } else {
    // We're in the first epoch, or we've made strictly fewer errors
    // than the previous epoch.
    num_epochs_in_decline_ = 0;
  }
  return loss_this_epoch;
}

void
PerceptronModel::ScoreCandidates(CandidateSet &candidates, bool training) {
  candidate_set_scorer_->Score(this, candidates, training);
}

double
PerceptronModel::ScoreCandidate(Candidate &candidate, bool training) {
  bool use_raw = training;
  const FeatureVector<int,double> &model = models_.GetModel(use_raw);
  double score = kernel_fn_->Apply(model, candidate.features());
  if (DEBUG >= 2) {
    cerr << "Time:" << time_.to_string() << ": scoring candidate "
         << candidate << " with " << (use_raw ? "raw" : "avg")
         << " model: " << model << endl
         << "\tscore: " << score << endl;
  }
  candidate.set_score(score);
  return score;
}

void
PerceptronModel::CompactifyFeatureUids() {
  // First, produce mapping for uid's of current non-zero features to dense
  // interval [0,n-1] (where there are n non-zero features).
  unordered_set<int> old_uids;
  models_.weights().GetNonZeroFeatures(old_uids);
  models_.average_weights().GetNonZeroFeatures(old_uids);
  unordered_map<int, int> old_to_new_uids;
  int new_uid = 0;
  for (unordered_set<int>::const_iterator it = old_uids.begin();
       it != old_uids.end();
       ++it) {
    old_to_new_uids[*it] = new_uid++;
  }
  models_.RemapFeatureUids(old_to_new_uids);
  best_models_.RemapFeatureUids(old_to_new_uids);

  if (symbols_ != NULL) {
    Symbols *old_symbols = symbols_->Clone();
    symbols_->Clear();
    for (Symbols::const_iterator it = old_symbols->begin();
         it != old_symbols->end();
         ++it) {
      unordered_map<int, int>::const_iterator old_to_new_uid_it =
          old_to_new_uids.find(it->second);
      if (old_to_new_uid_it != old_to_new_uids.end()) {
        int new_uid = old_to_new_uid_it->second;
        const string &symbol = it->first;
        symbols_->SetIndex(symbol, new_uid);
      }
    }
    delete old_symbols;
  }
}

void
PerceptronModel::ComputeFeaturesToUpdate(const CandidateSet &example,
                                         unordered_set<int> &
                                         gold_features_to_update,
                                         unordered_set<int> &
                                         best_scoring_features_to_update)
    const {
  // Collect gold features that are not in best-scoring candidate.
  const FeatureVector<int,double> &gold_features =
      example.GetGold().features();
  const FeatureVector<int,double> &best_scoring_features =
      example.GetBestScoring().features();

  if (DEBUG >= 2) {
    cerr << "Gold index: " << example.gold_index()
         << "; best scoring index: " << example.best_scoring_index()
         << endl;
    cerr << "Original gold features: " << gold_features << endl
         << "Original best scoring features: " << best_scoring_features << endl;
  }

  gold_features.GetNonZeroFeatures(gold_features_to_update);
  best_scoring_features.RemoveEqualFeatures(gold_features,
                                            gold_features_to_update);

  if (DEBUG >= 2) {
    cerr << "Time:" << time_.to_string() << ": new gold features: [";
    for (unordered_set<int>::const_iterator it =
             gold_features_to_update.begin();
         it != gold_features_to_update.end();
         ++it) {
      cerr << " " << *it;
    }
    cerr << "]" << endl;
  }

  best_scoring_features.GetNonZeroFeatures(best_scoring_features_to_update);
  gold_features.RemoveEqualFeatures(best_scoring_features,
                                    best_scoring_features_to_update);
  if (DEBUG >= 2) {
    cerr << "Time:" << time_.to_string() << ": new best scoring features: [";
    for (unordered_set<int>::const_iterator it =
             best_scoring_features_to_update.begin();
         it != best_scoring_features_to_update.end();
         ++it) {
      cerr << " " << *it;
    }
    cerr << "]" << endl;
  }

}

}  // namespace reranker
