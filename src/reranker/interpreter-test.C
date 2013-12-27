/// \file
/// Test driver for the Interpreter class.
/// \author dbikel@google.com (Dan Bikel)

#include <iostream>
#include <memory>

#include "interpreter.H"

using namespace std;
using namespace reranker;

int
main(int argc, char **argv) {
  cout << "Here is a list of abstract types and the concrete implementations\n"
       << "you can construct:" << endl;
  int debug = 1;
  Interpreter interpreter(debug);

  cout << endl;
  interpreter.PrintFactories(cout);

  cout << "\nHello!  Please type assignment statements.\n" << endl;

  if (argc >= 2) {
    interpreter.Eval(argv[1]);
  } else {
    interpreter.Eval(cin);
  }

  cout << "\nNow doing some hard-coded testing, looking to see if you\n"
       << "set a variable named \"f\" to have a boolean value." << endl;

  bool value_for_f;
  bool success = interpreter.Get("f", &value_for_f);
  if (success) {
    cout << "Success! f=" << (value_for_f ? "true" : "false") << endl;
  } else {
    cout << ":( ... no boolean value for variable f" << endl;
  }

  cout << "\n\nEnvironment: " << endl;
  interpreter.PrintEnv(cout);

  cout << "\nHave a nice day!\n" << endl;
}
