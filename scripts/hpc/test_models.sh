#!/bin/sh

while read line; do
    if [[ "$line" != "" && "${line::1}" != "#" ]]; then
        echo $line && sbatch -J "test_${line/\//-}" test_model.slurm "$line"

    fi
done < $1
