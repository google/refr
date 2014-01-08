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
/// Provides various example class headers.
/// \author dbikel@google.com (Dan Bikel)

#ifndef INFACT_EXAMPLE_H_
#define INFACT_EXAMPLE_H_

#include <string>

#include "factory.H"

namespace infact {

/// A very simple class to represent an animal.
class Animal : public FactoryConstructible {
 public:
  /// Constructs a generic animal.
  Animal() {}
  /// Destroys this animal.
  virtual ~Animal() {}

  /// Returns the name of this animal.
  virtual string GetName() = 0;
};

/// Registers the \link infact::Animal Animal \endlink with the
/// specified subtype <tt>TYPE</tt> and <tt>NAME</tt> with the \link
/// infact::Animal Animal \endlink \link infact::Factory
/// Factory\endlink.
#define REGISTER_ANIMAL(TYPE) \
  REGISTER_NAMED(TYPE,TYPE,Animal)

/// A class to represent a cow.
class Cow : public Animal {
public:
  /// Constructs a cow.
  Cow() : Animal() { }

  // Destroys this instance.
  virtual ~Cow() { }

  /// Registers the two variables that may be initialized when this object
  /// is constructed via \link Factory::CreateOrDie\endlink.
  /// <table>
  /// <tr>
  ///   <th>Variable name</th>
  ///   <th>Type</th>
  ///   <th>Required</th>
  ///   <th>Description</th>
  ///   <th>Default value</th>
  /// </tr>
  /// <tr>
  ///   <td><tt>name</tt></td>
  ///   <td><tt>string</tt></td>
  ///   <td>Yes</td>
  ///   <td>The name of this model animal.</td>
  ///   <td>n/a</td>
  /// </tr>
  /// <tr>
  ///   <td><tt>age</tt></td>
  ///   <td><tt>int</tt></td>
  ///   <td>No</td>
  ///   <td>The age of this animal.</td>
  ///   <td>0</td>
  /// </tr>
  /// </table>
  virtual void RegisterInitializers(Initializers &initializers) {
    initializers.Add("name", &name_, true);
    initializers.Add("age", &age_);
  }

  /// Returns the name of this animal.
  virtual string GetName() {
    return name_;
  }

private:
  string name_;
  int age_;
};

/// An owner of a pet.
class PetOwner : public FactoryConstructible {
 public:
  /// Constructs a generic pet owner.
  PetOwner() {}

  /// Destroys this pet owner.
  virtual ~PetOwner() {}

  /// Returns the number of pets owned by this pet owner.
  virtual int GetNumberOfPets() = 0;

  /// Gets the pet with the specified index owned by this pet owner.
  virtual shared_ptr<Animal> GetPet(int i) = 0;
};

#define REGISTER_PET_OWNER(TYPE) \
  REGISTER_NAMED(TYPE,TYPE,PetOwner)

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
