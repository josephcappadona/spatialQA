#!/bin/sh

while read line; do
    [[ -nz "$line" ]] \
        && [[ "${line::1}" != "#" ]] \
        && echo $line
        && python test_model.slurm $line
done < $1