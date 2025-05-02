#!/bin/bash

for file in stage1_results/*.json; do
    if [ -f "$file" ]; then
        python check_stage1.py "$file" AttractionProfile.csv >> output1_feasibility.log 2>&1
    fi
done