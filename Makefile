# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CPPSOURCES := $(shell find $(CURDIR) -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cc|cxx|cu)')

.PHONY: format lint postinstall

cppformat:
	if [ "$(CPPSOURCES)" != "" ]; then clang-format --verbose -i $(CPPSOURCES); fi

cpplint:
	if [ "$(CPPSOURCES)" != "" ]; then clang-format --verbose --dry-run --Werror $(CPPSOURCES); fi

format: cppformat
	python3 setup.py format

lint: cpplint
	python3 setup.py lint

postinstall:
	cd msamp/operators/dist_op && pip install -v -e . && cd -
	cd msamp/optim && pip install -v -e . && cd -
