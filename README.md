[![Build Status](https://travis-ci.org/ickc/TAIL.svg?branch=master)](https://travis-ci.org/ickc/TAIL)

# Introduction

Experimental pipeline as an alternative to POLARBEAR's AnalysisBackend — refactorize, modernize, cythonize, parallelize.

The name TAIL is a word play on POLARBEAR's AnalysisBackend — the "backend of polarbear", which has the shortest tails among living bears. Symbolically, TAIL should be like a tail of a polarbear — you never notice it in front of this powerful polarbear.

# Instructions

## Installing dependencies

First, it is highly recommend to use the Intel's Distribution for Python and Intel's icc, icpc compilers.

You can install the dependencies for the modules and tests by

```bash
# modules
pip install -r requirements.txt
# Tests
pip install -r tests/requirements.txt
```

## Building from source

Currently, you can either use Intel or GNU compiler to compile the source:

```bash
# Intel
make CXX=icpc
# GNU
make CXX=g++
```

Caveat: for some reason, on some systems you need to specify `CC` instead:

```bash
# Intel
make CC=icpc
# GNU
make CC=g++
```

## Running Tests

```bash
make test
```
