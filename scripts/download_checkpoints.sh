#!/bin/bash

# download
curl -L https://www.dropbox.com/s/se48hqjjaydq70t/checkpoints.zip?dl=0 >> check.zip
unzip check.zip
rm check.zip

# create folder structure
mkdir -p output/sdd/SAR/saved_models/
mkdir -p output/sdd/Goal_SAR/saved_models/

# move checkpoints at correct place
mv checkpoints/SAR_best_model.pt output/sdd/SAR/saved_models/SAR_best_model.pt
mv checkpoints/Goal_SAR_best_model.pt output/sdd/Goal_SAR/saved_models/Goal_SAR_best_model.pt
rm -r checkpoints
