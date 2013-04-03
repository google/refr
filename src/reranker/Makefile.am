AM_CPPFLAGS = `pkg-config --cflags protobuf` -I. -Wall
AM_LDFLAGS = -L../proto -L../gzstream -L../libb64/src \
        `pkg-config --cflags --libs protobuf`

LDADD = -lproto_lib -lprotobuf -lpthread -ldataio -lgzstream -lb64

bin_PROGRAMS = bin/run-model \
		bin/compile-features \
		bin/extract-features \
		bin/model-merge-reducer \
		bin/model-combine-shards \
		bin/model-combine-symbols \
		bin/piped-model-evaluator \
		bin/symbolize-model

testdir=${exec_prefix}/test-bin
test_PROGRAMS = bin/feature-vector-test \
		bin/feature-extractor-test \
		bin/candidate-set-proto-reader-test \
		bin/candidate-set-proto-writer-test \
		bin/perceptron-model-test \
		bin/executive-feature-extractor-test \
		bin/stream-tokenizer-test \
		bin/environment-test \
		bin/interpreter-test

PYTHON_TOOLS = bin/hadoop-run
${PYTHON_TOOLS}: ../scripts/python_wrapper bin/$(am__dirstamp)
	cp $< $@

BUILT_SOURCES = ${PYTHON_TOOLS} defs.py
CLEANFILES = ${PYTHON_TOOLS} defs.py
bin_SCRIPTS = ${PYTHON_TOOLS}

pythonlibdir=$(libdir)/python

pythonlib_SCRIPTS = hadoop-run.py
pythonlib_DATA = hadooputil.py pyutil.py defs.py

defs.py: defs.py.in
	if [ -f $@ ]; then rm -f $@ ; fi
	$(SED) -e "s:[@]bindir@:$(bindir):" \
	       -e "s:[@]hadoopbin@:$(HADOOPBIN):" $< > $@

SRCS =  feature-extractor.C executive-feature-extractor.C perceptron-model.C \
	symbol-table.C file-backed-loss-setter.C candidate.C candidate-set.C \
	model.C rank-feature-extractor.C basic-file-backed-feature-extractor.C \
	file-backed-ngram-feature-extractor.C \
	mira-style-model.C ngram-feature-extractor.C \
	string-canonicalizer.C factory.C \
	model-proto-reader.C model-proto-writer.C model-merge-reducer.C \
	stream-tokenizer.C interpreter.C

PROTO_DEP_SRCS = candidate-set-proto-reader.C candidate-set-proto-writer.C \
		 perceptron-model-proto-reader.C perceptron-model-proto-writer.C

lib_LIBRARIES = lib/libreranker.a
lib_libreranker_a_SOURCES = $(SRCS) $(PROTO_DEP_SRCS)

bin_run_model_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) run-model.C
bin_extract_features_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) extract-features.C
bin_compile_features_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) compile-features.C
bin_model_merge_reducer_SOURCES = model-merge-reducer.C model-merge-reducer-main.C
bin_model_combine_shards_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	model-combine-shards.C
bin_model_combine_symbols_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	model-combine-symbols.C
bin_piped_model_evaluator_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	piped-model-evaluator.C
bin_symbolize_model_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) symbolize-model.C

# The following are test executables (similar to unit tests).
bin_feature_extractor_test_SOURCES = $(SRCS) feature-extractor-test.C
bin_feature_vector_test_SOURCES = $(SRCS) feature-vector-test.C
bin_candidate_set_proto_reader_test_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	candidate-set-proto-reader-test.C
bin_candidate_set_proto_writer_test_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	candidate-set-proto-writer-test.C
bin_perceptron_model_test_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	perceptron-model-test.C
bin_stream_tokenizer_test_SOURCES = $(SRCS) stream-tokenizer-test.C
bin_executive_feature_extractor_test_SOURCES = $(SRCS) $(PROTO_DEP_SRCS) \
	executive-feature-extractor-test.C
bin_environment_test_SOURCES = $(SRCS) environment-test.C
bin_interpreter_test_SOURCES = $(SRCS) interpreter-test.C
