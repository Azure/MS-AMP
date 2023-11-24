# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CPPSOURCES := $(shell find $(CURDIR) -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cc|cxx|cu) -not -path '$(CURDIR)/third_party/*'')
MDSOURCES := $(shell find $(CURDIR) -regextype posix-extended -regex '.*\.md' -not -path '$(CURDIR)/.github/*' -not -path '$(CURDIR)/third_party/*' -prune)

.PHONY: format lint postinstall

cppformat:
	if [ "$(CPPSOURCES)" != "" ]; then clang-format --verbose -i $(CPPSOURCES); fi

cpplint:
	if [ "$(CPPSOURCES)" != "" ]; then clang-format --verbose --dry-run --Werror $(CPPSOURCES); fi

mdlint:
	if [ "$(MDSOURCES)" != "" ]; then mdl $(MDSOURCES) -g -c .mdlrc; fi

format: cppformat
	python3 setup.py format

lint: cpplint mdlint
	python3 setup.py lint

postinstall:
	cd msamp/operators/dist_op && bash build.sh && cd -
	cd msamp/operators/arithmetic && pip install -v . && cd -
	cd msamp/optim && pip install -v . && cd -
