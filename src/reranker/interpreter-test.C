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
  cout << "\nHello!  Please type assignment statements.\n" << endl;

  int debug = 1;
  Interpreter interpreter(debug);
  if (argc >= 2) {
    interpreter.Eval(argv[1]);
  } else {
    interpreter.Eval(cin);
  }
  bool value_for_f;
  bool success = interpreter.Get("f", &value_for_f);
  if (success) {
    cout << "Success! f=" << (value_for_f ? "true" : "false") << endl;
  } else {
    cout << ":( ... no value for variable f" << endl;
  }

  cout << "\n\nEnvironment: " << endl;
  interpreter.PrintEnv(cout);

  cout << "\nHave a nice day!\n" << endl;
}
