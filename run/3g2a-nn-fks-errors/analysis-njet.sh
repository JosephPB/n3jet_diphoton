#!/usr/bin/env bash

rseed=1

for ((nnrun = 0; nnrun < 20; nnrun++)); do
    dir=run-${nnrun}

    cd ${dir}/run
    cp ${nnrun}.${rseed}.analysis.yoda Analysis-nn-${nnrun}.yoda

    RIVET_ANALYSIS_PATH=../../../../analysis/diphoton-1tev/ rivet-mkhtml ../../njet-3M/Analysis-njet.yoda Analysis-nn-${nnrun}.yoda --errs -t "NJet vs NN for run ${rseed}"

    cp -r rivet-plots/ ../../njet-3M/rivet-plots-${nnrun}/

    cd ../..

done
