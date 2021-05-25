#!/usr/bin/env bash
# To run jobs on a single machine, do
# ./local-analyses.sh <initial seed> <number of runs> <total number of events>

init_seed=$1
num_runs=$2
end_seed=$(($init_seed + $num_runs - 1))
total_events=$3
run_events=$(($total_events / $num_runs))

for rseed in $(seq -w ${init_seed} ${end_seed}); do
    RIVET_ANALYSIS_PATH=../../analysis/diphoton-1tev/ Sherpa -f Run.dat -R $rseed -e $run_events -A Analysis.$rseed >${rseed}.analysis.log 2>${rseed}.analysis.err &
done
