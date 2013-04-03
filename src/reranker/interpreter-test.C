/// \file
/// Test driver for the Interpreter class.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <tr1/memory>

#include "interpreter.H"

using namespace std;
using namespace std::tr1;
using namespace reranker;

int
main(int argc, char **argv) {
  cout << "Hello!  Please type assignment statements." << endl;

  Interpreter interpreter;
  interpreter.Eval(cin);

  cout << "Have a nice day!" << endl;
}
