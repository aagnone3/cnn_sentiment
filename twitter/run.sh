#!/usr/bin/env bash
while [[ True ]]
do
    python run.py
    echo "Program finished/killed at $(date +%Y%m%d-%H%M%S). Waiting 30 mins"
    sleep 1800
done
