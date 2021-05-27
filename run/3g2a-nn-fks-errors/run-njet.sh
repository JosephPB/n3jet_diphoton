#!/usr/bin/env bash

rseed=1
run_events=100

for ((nnrun = 0; nnrun < 20; nnrun++)); do
    dir=run-${nnrun}
    mkdir -p ${dir}
    mkdir -p ${dir}/interface
    mkdir -p ${dir}/run

    perl -p -e "s|LLLLL|21|g; s|GGGGG|Interface${nnrun}|g; s|JJJJJ|1|g;" template.Run.dat >${dir}/run/Run.${nnrun}.dat

    perl -p -e "s|AAAAA|${nnrun}|g;" template.Makefile.run >${dir}/run/Makefile

    perl -p -e "s|AAAAA|${nnrun}|g;" template.Makefile.interface >${dir}/interface/Makefile

    cd ${dir}/interface

    ln -sf ../../../../interface/generic/model_fns.hpp .
    ln -sf ../../../../interface/generic/interface.cpp .
    ln -sf ../../../../interface/generic/interface.hpp .

    make -j

    cd ../run

    ln -sf ../interface/libInterface${nnrun}.so .
    ln -sf ../../njet-3M/Results.db
    ln -sf ../../njet-3M/Results.db.bak

    name=${nnrun}.${rseed}.analysis
    RIVET_ANALYSIS_PATH=../../../../analysis/diphoton-1tev Sherpa -f Run.${nnrun}.dat -R ${rseed} -e ${run_events} -A ${name} >${name}.out 2>${name}.err &

    cd ../..
done
