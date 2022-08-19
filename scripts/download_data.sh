#!/bin/bash

curl -L https://www.dropbox.com/s/luu2t6c6d24xvrb/dataset.zip?dl=0 >> data.zip
unzip data.zip
rm -r data.zip
mv data ..
