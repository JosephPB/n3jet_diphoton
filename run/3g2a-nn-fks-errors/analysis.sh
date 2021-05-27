#!/usr/bin/env bash

rseed=1

for ((nnrun = 0; nnrun < 20; nnrun++)); do
    dir=run-${nnrun}

    cd ${dir}/run
    cp ${nnrun}.${rseed}.analysis.yoda Analysis-nn-${nnrun}.yoda

    RIVET_ANALYSIS_PATH=analysis-diphoton rivet-mkhtml ../../Analysis-njet.yoda Analysis-nn-${nnrun}.yoda --errs -t "NJet vs NN for run ${rseed}"

    cd ../..

done
