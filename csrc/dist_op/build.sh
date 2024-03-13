#!/bin/bash

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

BUILD_ROOT=build
mkdir -p $BUILD_ROOT
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -B $BUILD_ROOT
cmake --build $BUILD_ROOT
cmake --install $BUILD_ROOT
