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

#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include "dataio.h"
#include "../gzstream/gzstream.h"
#include "../utils/kdebug.h"
INITDEBUG(0);

using namespace std;

const int kBufferSize = 1 >> 20;

ConfusionProtoIO::ConfusionProtoIO()
    : base64_(true), is_compressed_(false),
      inputs_(NULL), outputs_(NULL),
      iobuffer_(NULL), b64buffer_(NULL), bufsize_(0),
      encode_obj_(kBufferSize), decode_obj_(kBufferSize) {
}

ConfusionProtoIO::ConfusionProtoIO(const string& file_name,
                                   Mode iomode,
                                   bool compressed,
                                   bool base64)
    : base64_(base64), is_compressed_(compressed),
      inputs_(NULL), outputs_(NULL),
      iobuffer_(NULL), b64buffer_(NULL), bufsize_(0),
      encode_obj_(kBufferSize), decode_obj_(kBufferSize) {
  // Open the raw file:
  switch (iomode) {
   case READSTD:
     if (compressed) {
       cerr << "Compression on stdin not supported";
     }
     inputs_ = &cin;
     break;
   case READ:
     if (compressed) {
       inputs_ = new igzstream(file_name.c_str());
     } else {
       inputs_ = new ifstream(file_name.c_str());
     }
    break;
   case WRITESTD:
     if (compressed) {
       cerr << "Compression on stdout not supported";
     }
     outputs_ = &cout;
     break;
   case WRITE:
     if (compressed) {
       outputs_ = new ogzstream(file_name.c_str());
     } else {
       outputs_ = new ofstream(file_name.c_str());
     }
     break;
  };
}

ConfusionProtoIO::~ConfusionProtoIO() {
  Close();
}

void ConfusionProtoIO::Close(void) {
  if (inputs_ != &cin) {
    delete inputs_;
  }
  inputs_ = NULL;
  if (outputs_ != &cout) {
    delete outputs_;
  }
  outputs_ = NULL;
  if (iobuffer_ != NULL) {
    delete [] iobuffer_;
    iobuffer_ = NULL;
    delete [] b64buffer_;
    b64buffer_ = NULL;
    bufsize_ = 0;
  }
}

bool ConfusionProtoIO::ResizeBuffer(int newsize, bool b64) {
  if (bufsize_ < newsize) {
    if (iobuffer_) {
      delete [] iobuffer_;
      if (b64) {
        delete [] b64buffer_;
      }
    }
    bufsize_ = newsize * 2;
    iobuffer_ = new char[bufsize_];
    if (b64) {
      b64buffer_ = new char[bufsize_ * 2];
    }
  }
  return true;
}

bool ConfusionProtoIO::WriteRaw(const google::protobuf::Message& message) {
  if (outputs_ == NULL) {
    cerr << "No output stream specified" << endl;
    return false;
  }
  CDEBUG(5, "Writing message of " << message.ByteSize() << " bytes");
  int num_bytes = message.ByteSize();
  *outputs_ << num_bytes;
  ResizeBuffer(num_bytes, false);
  if (!message.SerializeToArray(iobuffer_, num_bytes)) {
    cerr << "Unable to write message to stream" << endl;
    return false;
  }
  outputs_->write(iobuffer_, num_bytes);
  return true;
}

int ConfusionProtoIO::EncodeBase64(const google::protobuf::Message& message,
                                   string* encodedmsg) {
  base64_init_encodestate(&(encode_obj_._state));

  int num_bytes = message.ByteSize();
  CDEBUG(5, "About to resize buffer");
  ResizeBuffer(num_bytes, true);
  CDEBUG(5, "About to serialize the message of " << num_bytes << " bytes");
  if (!message.SerializeToArray(iobuffer_, num_bytes)) {
    cerr << "Unable to write message to stream" << endl;
    return -1;
  }
  CDEBUG(5, "Encoding the message into internal buffer");
  b64buffer_[0] = '\n';
  int total_codelength = 0;
  int codelength = encode_obj_.encode(iobuffer_, num_bytes, b64buffer_);
  if (codelength > 0) {
    CDEBUG(5, "Wrote " << codelength << " bytes of encoded message");
    total_codelength += codelength;
    encodedmsg->append(b64buffer_, codelength);
  }
  CDEBUG(5, "Encoding end of message into internal buffer");
  codelength = encode_obj_.encode_end(b64buffer_);
  if (codelength > 0) {
    total_codelength += codelength;
    encodedmsg->append(b64buffer_, codelength);
  }
  if (total_codelength < 1) {
    cerr << "Unable to base64 encode bytes" << endl;
    cerr << "Attempted to write string of length: " << num_bytes << endl;
    return -1;
  }
  return total_codelength;
}

bool ConfusionProtoIO::WriteBase64(const google::protobuf::Message& message) {
  CDEBUG(5, "Writing message of " << message.ByteSize() << " bytes");
  b64obuffer_.clear();
  if (outputs_ == NULL) {
    cerr << "No output stream specified" << endl;
    return false;
  }
  int codelength = -1;
  if ((codelength = EncodeBase64(message, &b64obuffer_)) < 0) {
    return false;
  }
  if (codelength > 0) {
    // Base64 Encode the string.
    outputs_->write(b64obuffer_.c_str(), codelength);
    CDEBUG(5, "Wrote message of " << codelength << " base64 encoded bytes");
  }
  b64obuffer_.clear();
  return true;
}

bool ConfusionProtoIO::ReadRaw(google::protobuf::Message* message) {
  if (inputs_ == NULL) {
    cerr << "No input stream specified" << endl;
    return false;
  }
  int num_bytes = -1;
  *inputs_ >> num_bytes;
  if (num_bytes == -1) {
    CDEBUG(5, "Found end of file");
    return false;
  }
  ResizeBuffer(num_bytes, false);
  inputs_->read(iobuffer_, num_bytes);
  CDEBUG(5, "Read " << num_bytes << " as the size of the message");
  if (!message->ParseFromArray(iobuffer_, num_bytes)) {
    cerr << "Unable to read message" << endl;
    return false;
  }
  return true;
}

bool ConfusionProtoIO::DecodeBase64(const string& encodedmsg,
                                    google::protobuf::Message* message) {
  // Assume that the protos were short enough to fit into a single base64 block.
  // Otherwise this will break.
  base64_init_decodestate(&(decode_obj_._state));
  if (encodedmsg.empty()) {
    return false;
  }
  ResizeBuffer(encodedmsg.length(), true);
  // Base64 decode the string.
  int datalen = decode_obj_.decode(encodedmsg.c_str(), encodedmsg.length(), iobuffer_);
  CDEBUG(5, "Decoded an object of length: " << datalen);
  if (!message->ParseFromArray(iobuffer_, datalen)) {
    cerr << "Unable to read message" << endl;
    return false;
  }
  return true;
}

bool ConfusionProtoIO::ReadBase64(google::protobuf::Message* message) {
  if (inputs_ == NULL) {
    cerr << "No input stream specified" << endl;
    return false;
  }
  getline(*inputs_, b64obuffer_);
  if (!DecodeBase64(b64obuffer_, message)) {
    return false;
  }
  CDEBUG(5, "Read an input of record length: " << b64obuffer_.length());
  // Resize for later calls.
  if (b64obuffer_.capacity() < b64obuffer_.length() * 2) {
    b64obuffer_.reserve(b64obuffer_.length() * 2);
  }
  return true;
}


