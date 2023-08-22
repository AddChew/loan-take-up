#!/bin/bash

conda create -n loan-take-up python=3.10 -y
source activate loan-take-up
pip install -r requirements.txt