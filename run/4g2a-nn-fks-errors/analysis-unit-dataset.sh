#!/usr/bin/env bash

rseed=$1

for ((nnrun = 0; nnrun < 20; nnrun++)); do
    dir=res-${nnrun}

    cd ${dir}/run
    cp ${nnrun}.${rseed}.analysis.yoda Analysis-nn-${nnrun}.yoda

    RIVET_ANALYSIS_PATH=../../../../analysis/diphoton-1tev/ rivet-mkhtml ../../unit-3M/Analysis-unit.yoda Analysis-nn-${nnrun}.yoda --errs -t "NJet vs NN for run ${rseed}"

    cp -r rivet-plots/ ../../unit-3M/rivet-plots-dataset-${nnrun}/

    cd ../..

done
