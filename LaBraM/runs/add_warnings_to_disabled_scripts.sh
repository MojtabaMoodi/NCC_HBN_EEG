#!/bin/bash
# Script to add warning comments to disabled CV and cross-task run scripts

cd "$(dirname "$0")"

# Find all CV and cross-task scripts
for file in run_*_cv_*.sh run_*_cross_task*.sh; do
    if [ -f "$file" ]; then
        # Check if warning already exists
        if ! grep -q "WARNING.*DISABLED" "$file"; then
            # Determine warning type
            if [[ "$file" == *"_cv_"* ]]; then
                warning="# ⚠️  WARNING: Cross-validation experiments are currently DISABLED
# Cross-validation functionality has been commented out in labram_dataset.py
# This script will fail until cross-validation is re-enabled for HDF5 format
# See labram_dataset.py for details"
            elif [[ "$file" == *"_cross_task_"* ]]; then
                warning="# ⚠️  WARNING: Cross-task experiments are currently DISABLED
# Cross-task functionality has been commented out in labram_dataset.py
# This script will fail until cross-task is re-enabled for HDF5 format
# See labram_dataset.py for details"
            else
                continue
            fi
            
            # Insert warning after the "Generated automatically" comment
            # Use a temporary file to avoid issues with sed on different systems
            awk -v warning="$warning" '
                /Generated automatically/ {
                    print
                    print ""
                    print warning
                    next
                }
                { print }
            ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
            
            echo "Added warning to: $file"
        else
            echo "Warning already exists in: $file"
        fi
    fi
done

echo "Done!"

