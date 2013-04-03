/// \file
/// Test driver for the Environment class.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <tr1/memory>

#include "environment.H"

using namespace std;
using namespace std::tr1;
using namespace reranker;

int
main(int argc, char **argv) {
  bool test = true;
  Environment env(test);
}
