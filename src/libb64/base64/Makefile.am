BUFFERSIZE = 16777216
AM_CPPFLAGS = -DBUFFERSIZE=$(BUFFERSIZE) -I$(srcdir)/../include

bin_PROGRAMS = bin/base64
bin_base64_SOURCES = base64.cc
bin_base64_LDADD = ../src/libb64.a