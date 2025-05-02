#!/bin/bash

# Clear previous results to avoid confusion
rm -rf stage1_results
mkdir -p stage1_results

# Run for tourists 1-100
for id in {1..100}; do
  echo "Processing Tourist $id for Stage 1..."
  python alns_main_stage1.py AttractionProfile.csv TouristProfile.csv 123 --tourist_id $id --mode single --save_initial
done

echo "Stage 1 processing complete for 100 tourists"