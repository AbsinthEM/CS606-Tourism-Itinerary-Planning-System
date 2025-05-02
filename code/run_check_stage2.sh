#!/bin/bash

for file in disruption_routes/*.json; do
    if [ -f "$file" ]; then
        python check_stage2.py "$file" AttractionProfile.csv >> output.log 2>&1
    fi
done