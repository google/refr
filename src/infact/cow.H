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
/// Provides the Cow class.

#ifndef INFACT_COW_H_
#define INFACT_COW_H_

#include "animal.H"

namespace infact {

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

}  // namespace reranker

#endif
