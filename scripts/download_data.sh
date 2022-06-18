#!/bin/bash

curl -L https://www.dropbox.com/s/980xn344luntr4s/dataset.zip?dl=0 >> data.zip
unzip data.zip
rm -r data.zip
mv data ..
