#!/bin/bash
# Clear previous results to avoid confusion
rm -rf stage2_results
mkdir -p stage2_results
# Run for tourists 1-100
for id in {1..100}; do
    echo "Processing Tourist $id for Stage 2..."
    python disruption_main_stage2.py AttractionProfile.csv TouristProfile.csv --tourist_id $id --mode single --seed 123
done
echo "Stage 2 processing complete for 100 tourists"