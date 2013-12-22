#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.65])
AC_INIT([Reranker Framework (ReFr)], [1.0], [dbikel@google.com])
AM_INIT_AUTOMAKE([foreign nostdinc -Wall -Werror])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AC_PROG_SED
AM_PROG_AR

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_SRCDIR([src/dataconvert/asr_nbest_proto.cc])
AC_CONFIG_FILES([Makefile
                 src/scripts/python_wrapper
                 src/Makefile
                 src/dataconvert/Makefile
                 src/gzstream/Makefile
                 src/libb64/Makefile
                 src/libb64/base64/Makefile
                 src/libb64/src/Makefile
                 src/proto/Makefile
                 src/reranker/Makefile])

# Checks for libraries.
AC_CHECK_LIB([m], [pow])
AC_CHECK_LIB([z], [gzread])

# find Protocol Buffers
#PKG_CHECK_MODULES(PROTOBUF, protobuf >= 2.0)
AC_PATH_PROG(PROTOC, protoc, [no])
if test "$PROTOC" == "no"; then
    AC_MSG_FAILURE([Protocol Buffers compiler 'protoc' is required to build.])
fi

# Check for the google sparse hash
AC_ARG_WITH(sparsehash, AS_HELP_STRING([--with-sparsehash=PATH],
            [specify directory containing the google sparsehash headers http://code.google.com/p/google-sparsehash/)]))

if test "$with_sparsehash" -a -d "$with_sparsehash"; then
  sparsehash_include="-I$with_sparsehash/include"
fi

AC_SUBST(CPPFLAGS, "$CPPFLAGS $sparsehash_include")

AC_LANG_PUSH(C++)
# Check that sparsehash is available.
AC_CHECK_HEADERS([sparsehash/sparse_hash_set],,\
                 [AC_MSG_ERROR([google sparse hash library is required: http://code.google.com/p/google-sparsehash/])])
AC_LANG_POP(C++)

AM_PATH_PYTHON([2.4])

AC_PATH_PROG(HADOOPBIN, hadoop, [no])
if test "$HADOOPBIN" == "no"; then
   AC_MSG_RESULT([No hadoop binary ... continuing anyway])
fi

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([floor memset])

AC_OUTPUT
