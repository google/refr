/// \file
/// Test driver for the Environment class.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <tr1/memory>

#include "environment-impl.H"

using namespace std;
using namespace std::tr1;
using namespace reranker;

int
main(int argc, char **argv) {
  int debug = 1;
  Environment *env = new EnvironmentImpl(debug);
  delete env;
}
