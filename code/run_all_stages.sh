#!/bin/bash

# Script to run Stage 1 and Stage 2 for multiple tourists

# Set the number of tourists to process
NUM_TOURISTS=100

# Create output log files with timestamps
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGE1_LOG="stage1_${TIMESTAMP}.log"
STAGE2_LOG="stage2_${TIMESTAMP}.log"

echo "Starting automated processing for ${NUM_TOURISTS} tourists at $(date)"
echo "Stage 1 logs will be saved to ${STAGE1_LOG}"
echo "Stage 2 logs will be saved to ${STAGE2_LOG}"

# Create or clear directories
echo "Preparing directories..."
rm -rf stage1_results
mkdir -p stage1_results
rm -rf stage2_results
mkdir -p stage2_results
rm -rf disruption_routes
mkdir -p disruption_routes

# Process Stage 1 first - create optimal itineraries
echo "Starting Stage 1 processing at $(date)"
echo "==== STAGE 1 PROCESSING STARTED ====" > $STAGE1_LOG

for id in $(seq 1 $NUM_TOURISTS); do
  echo "Processing Tourist $id for Stage 1..."
  echo "==== Processing Tourist $id ====" >> $STAGE1_LOG
  python alns_main_stage1.py AttractionProfile.csv TouristProfile.csv 123 --tourist_id $id --mode single --save_initial >> $STAGE1_LOG 2>&1
  echo "Tourist $id Stage 1 complete at $(date)"
done

echo "Stage 1 processing complete at $(date)"

# Run Stage 1 validation
echo "Validating Stage 1 solutions..."
echo "==== STAGE 1 VALIDATION ====" >> $STAGE1_LOG
bash run_check_stage1.sh >> $STAGE1_LOG 2>&1

# Process Stage 2 - test under disruptions
echo "Starting Stage 2 processing at $(date)"
echo "==== STAGE 2 PROCESSING STARTED ====" > $STAGE2_LOG

for id in $(seq 1 $NUM_TOURISTS); do
  echo "Processing Tourist $id for Stage 2..."
  echo "==== Processing Tourist $id ====" >> $STAGE2_LOG
  python disruption_main_stage2.py AttractionProfile.csv TouristProfile.csv --tourist_id $id --mode single --seed 123 >> $STAGE2_LOG 2>&1
  echo "Tourist $id Stage 2 complete at $(date)"
done

echo "Stage 2 processing complete at $(date)"

# Run Stage 2 validation
echo "Validating Stage 2 solutions..."
echo "==== STAGE 2 VALIDATION ====" >> $STAGE2_LOG
bash run_check_stage2.sh >> $STAGE2_LOG 2>&1

# Generate summary of results
echo "Generating summary report..."
echo "==== SUMMARY REPORT ====" > summary_${TIMESTAMP}.txt
echo "Processing completed at $(date)" >> summary_${TIMESTAMP}.txt
echo "" >> summary_${TIMESTAMP}.txt

echo "Stage 1 Results:" >> summary_${TIMESTAMP}.txt
echo "Total optimized itineraries: $(ls stage1_results/*optimized_solution.json 2>/dev/null | wc -l)" >> summary_${TIMESTAMP}.txt
echo "" >> summary_${TIMESTAMP}.txt

echo "Stage 2 Results:" >> summary_${TIMESTAMP}.txt
echo "Total quick repair solutions: $(ls disruption_routes/*quickrepair.json 2>/dev/null | wc -l)" >> summary_${TIMESTAMP}.txt
echo "Total ALNS replanned solutions: $(ls disruption_routes/*replanned.json 2>/dev/null | wc -l)" >> summary_${TIMESTAMP}.txt

echo "Process complete! Summary saved to summary_${TIMESTAMP}.txt"