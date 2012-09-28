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
// Convert Philipp's MT n-best list format to a DataSet protocol buffer.
// Author: kbhall@google.com (Keith Hall)

#ifndef PROTO_DATAIO_H_
#define PROTO_DATAIO_H_

#include <fcntl.h>
#include <google/protobuf/message.h>
#include <string>
#include "../libb64/include/b64/decode.h"
#include "../libb64/include/b64/encode.h"
#include "../utils/kdebug.h"

using namespace std;
using google::protobuf::Message;

class ConfusionProtoIO {
 public:
  enum Mode {
    READ,
    WRITE,
    READSTD,
    WRITESTD
  };
  ConfusionProtoIO();
  ConfusionProtoIO(const string& file_name, Mode iomode,
                   bool compressed = true, bool base64 = true);
  ~ConfusionProtoIO();

  void Close();

  bool Read(Message* message) {
    if (base64_) {
      return ReadBase64(message);
    } else {
      return ReadRaw(message);
    }
  }
  bool Write(const Message& message) {
    if (base64_) {
      return WriteBase64(message);
    } else {
      return WriteRaw(message);
    }
  }
  bool DecodeBase64(const string& encodedmsg, Message* message);
  int EncodeBase64(const Message& message, string* encodedmsg);

  istream* inputstream(void) { return inputs_; }
  ostream* outputstream(void) { return outputs_; }

 protected:
  bool ResizeBuffer(int num_bytes, bool b64);
  bool ReadRaw(Message* message);
  bool WriteRaw(const Message& message);
  bool ReadBase64(Message* message);
  bool WriteBase64(const Message& message);

 private:
  bool base64_;
  bool is_compressed_;
  istream* inputs_;
  ostream* outputs_;
  char* iobuffer_;
  char* b64buffer_;;
  int bufsize_;
  string b64obuffer_;
  base64::encoder encode_obj_;
  base64::decoder decode_obj_;
};

#endif
