AM_CPPFLAGS = `pkg-config --cflags protobuf`

AM_LDFLAGS = -L../gzstream -L../libb64/src \
		`pkg-config --cflags --libs protobuf`

bin_PROGRAMS = bin/protoview bin/featconvert bin/featcount

bin_protoview_SOURCES = protoview.cc
bin_protoview_LDADD = libdataio.a libproto_lib.a -lprotobuf -lpthread -lgzstream -lb64

bin_featconvert_SOURCES = featconvert.cc
bin_featconvert_LDADD = libdataio.a libproto_lib.a -lprotobuf -lpthread -lgzstream -lb64

bin_featcount_SOURCES = featcount.cc
bin_featcount_LDFLAGS = -lm -lz

lib_LIBRARIES = libproto_lib.a libdataio.a
libproto_lib_a_SOURCES =
libproto_lib_a_LIBADD = model.pb.o data.pb.o

libdataio_a_SOURCES = dataio.cc dataio.h
libdataio_a_LIBADD = libproto_lib.a

PROTO_SRCS = model.pb.h model.pb.cc data.pb.h data.pb.cc
BUILT_SOURCES = ${PROTO_SRCS}
CLEANFILES = ${PROTO_SRCS}

model.pb.cc model.pb.h: model.proto
	protoc --cpp_out=. $<

model_python: model.proto
	protoc --python_out=. $<

model_java: model.proto
	protoc --java_out=. $<

data.pb.cc data.pb.h: data.proto
	protoc --cpp_out=. $<

data_python: data.proto
	protoc --python_out=. $<

data_java: data.proto
	protoc --java_out=. $<
