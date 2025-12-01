#!/bin/bash
# Script to monitor SLURM job output
# Usage: ./monitor_jobs.sh [job_id] [job_name]

if [ -z "$1" ]; then
    # No arguments - show all running jobs
    echo "=========================================="
    echo "Your Running SLURM Jobs"
    echo "=========================================="
    squeue -u $USER -o "%.18i %.20j %.8u %.2t %.10M %.6D %R"
    echo ""
    echo "Recent output files:"
    ls -lth ../logs/slurm/*.out 2>/dev/null | head -10
    echo ""
    echo "Usage:"
    echo "  ./monitor_jobs.sh <job_id>           # Follow specific job output"
    echo "  ./monitor_jobs.sh <job_name>         # Follow job by name pattern"
    echo "  ./monitor_jobs.sh                    # List all jobs"
    exit 0
fi

JOB_ID="$1"
LOG_DIR="../logs/slurm"

# If argument is a number, treat as job ID
if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    # Find output file by job ID
    OUTPUT_FILE=$(find "$LOG_DIR" -name "*-${JOB_ID}.out" 2>/dev/null | head -1)
    ERROR_FILE=$(find "$LOG_DIR" -name "*-${JOB_ID}.err" 2>/dev/null | head -1)
    
    if [ -z "$OUTPUT_FILE" ]; then
        echo "No output file found for job ID: $JOB_ID"
        echo "Checking job status..."
        scontrol show job "$JOB_ID"
        exit 1
    fi
else
    # Treat as job name pattern
    OUTPUT_FILE=$(find "$LOG_DIR" -name "*${JOB_ID}*.out" 2>/dev/null | head -1)
    ERROR_FILE=$(find "$LOG_DIR" -name "*${JOB_ID}*.err" 2>/dev/null | head -1)
    
    if [ -z "$OUTPUT_FILE" ]; then
        echo "No output file found matching: $JOB_ID"
        echo "Available files:"
        ls -1 "$LOG_DIR"/*.out 2>/dev/null | head -10
        exit 1
    fi
fi

echo "=========================================="
echo "Monitoring Job Output"
echo "=========================================="
echo "Output file: $OUTPUT_FILE"
if [ -n "$ERROR_FILE" ]; then
    echo "Error file: $ERROR_FILE"
fi
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Show last 50 lines, then follow
tail -n 50 "$OUTPUT_FILE"
echo ""
echo "--- Following output (new lines will appear below) ---"
echo ""
tail -f "$OUTPUT_FILE"

