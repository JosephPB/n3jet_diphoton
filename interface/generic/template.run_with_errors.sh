#!/usr/bin/env bash

# this is for illustrative purposes only!

rseed=1
run_events=100

for ((nnrun = 0; nnrun < 20; nnrun++)); do
    dir=run-${nnrun}
    mkdir ${dir}

    perl -p -e "s|AAAAA|${nnrun}|g;" ../template.Makefile >Makefile

    make -j

    perl -p -e "s|LLLLL|21|g; s|GGGGG|Interface${nnrun}|g;" ../template.Run.dat >Run.${nnrun}.dat

    name=${nnrun}.${rseed}.analysis
    RIVET_ANALYSIS_PATH=? Sherpa -f Run.${nnrun}.dat -R ${rseed} -e ${run_events} -A ${name} >${name}.out 2>${name}.err &

    cd ..
done
