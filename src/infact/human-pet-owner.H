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
/// Provides the HumanPetOwner class.

#ifndef INFACT_HUMAN_PET_OWNER_H_
#define INFACT_HUMAN_PET_OWNER_H_

#include "pet-owner.H"

namespace infact {

/// A concrete type of PetOwner that can be constructed by a \link
/// Factory\endlink\<\link PetOwner \endlink\> instance.
class HumanPetOwner : public PetOwner {
 public:
  /// Constructs a human pet owner.
  HumanPetOwner() : PetOwner() { }

  // Destroys a human pet owner.
  virtual ~HumanPetOwner() { }

  virtual void RegisterInitializers(Initializers &initializers) {
    initializers.Add("pets", &pets_, true);
  }

  /// Returns the number of pets owned by this pet owner.
  virtual int GetNumberOfPets() {
    return pets_.size();
  }

  /// Gets the pet with the specified index owned by this pet owner.
  virtual shared_ptr<Animal> GetPet(int i) {
    return pets_.at(i);
  }

 private:
  vector<shared_ptr<Animal> > pets_;
};

}  // namespace infact

#endif
