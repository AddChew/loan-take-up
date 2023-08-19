#!/bin/bash

conda create -n loan-conversion python=3.10 -y
source activate loan-conversion
pip install -r requirements.txt