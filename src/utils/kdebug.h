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
// Description:
//  Basic macros to help with debugging.
//
//----------------------------------------------------------
#if !defined(KDEBUG_OFF)
  extern int kDEBUG;
  #define INITDEBUG(val) int kDEBUG=val;
  #define SETDEBUG(val) kDEBUG=val;
//    #if KDEBUG > 5
//        #error "Max Debug Level is 5"
//    #endif

//    #if KDEBUG < 0
//        #error "Min Debug Level is 0"
//    #endif

  #ifdef CDEBUG
    #undef CDEBUG
  #endif

  #ifdef FUNC_DEBUG
    #undef FUNC_DEBUG
  #endif
  #ifdef DIEDEBUG
    #undef DIEDEBUG
  #endif

  #define CDEBUG(errVal, a) {if (errVal <= kDEBUG) {std::cerr << a << std::endl;}}
  #define DIEDEBUG(exitVal, a) {std::cerr << a << std::endl; exit(exitVal);}
  #define FUNC_DEBUG(errVal, a) {if (errVal <= kDEBUG) {std::cerr << a << std::endl;}}
#else
  #ifdef CDEBUG
    #undef CDEBUG
  #endif

  #ifdef FUNC_DEBUG
    #undef FUNC_DEBUG
  #endif

  #ifdef DIEDEBUG
    #undef DIEDEBUG
  #endif
  #define CDEBUG(errVal, a) {}
  #define FUNC_DEBUG(errVal, a) {}
  #define DIEDEBUG(exitVal, a) {}
#endif
