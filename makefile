#!/bin/bash

all:
	gcc -Wall storkey9_deme_blockfit.c  -lgsl -lgslcblas -lm -O3 -o aann
