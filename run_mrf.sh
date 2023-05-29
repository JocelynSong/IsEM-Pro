#!/bin/bash

protein=$1


python3 fairseq_cli/mrf.py --protein ${protein}
