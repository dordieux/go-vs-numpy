CXX ?= clang++
ARCH := $(shell uname -m)
CXXFLAGS := -O3 -std=c++17
ifeq ($(ARCH),x86_64)
CXXFLAGS += -march=native
endif

bench: bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f bench

.PHONY: clean
