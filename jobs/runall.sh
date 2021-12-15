#!/bin/bash
for file in ./job*
do 
sbatch $file
done
