#!/bin/bash
python2 RFtest.py > $1
sed -i 's/\[//g' $1
sed -i 's/\]//g' $1
sed -i 's/,//g' $1
