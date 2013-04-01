AM_CPPFLAGS = `pkg-config --cflags protobuf`
AM_LDFLAGS = -L../proto -L../gzstream -L../libb64/src \
        `pkg-config --cflags --libs protobuf`

LDADD = -lproto_lib -lprotobuf -lpthread -ldataio -lgzstream -lb64

bin_PROGRAMS = mt_nbest_proto asr_nbest_proto

mt_nbest_proto_SOURCES = mt_nbest_proto.cc ConvertMT.h ConvertMT.cc \
	../utils/kdebug.h

asr_nbest_proto_SOURCES = asr_nbest_proto.cc ConvertASR.h ConvertASR.cc \
	../utils/kdebug.h
