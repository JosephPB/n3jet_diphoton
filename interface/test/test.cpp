#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "model_fns.h"

int main()
{
    std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
    std::cout.precision(16);

    std::cout << '\n'
              << "n3jet: simple example of calling a pretrained neural network for inference in a C++ interface" << '\n'
              << '\n';

    const int legs = 5;
    const int pspoints = 2;
    const int pairs = 9;
    const int training_reruns = 20;
    const double delta = 0.02;

    //raw momenta input

    double Momenta[pspoints][legs][4] = {
        { { 80.80323390148537, 0.0, 0.0, 80.8032339014854 },
            { 53.57113343015938, 0.0, 0.0, -53.571133430159385 },
            { 42.47353254764989, 26.671855251549694, 29.511816381880454, 14.88844512898612 },
            { 52.68325739804988, -51.14726465398362, 1.1299491037855054, 12.578002365534454 },
            { 39.21757738594496, 24.475409402433904, -30.641765485665946, -0.2343470231945908 } },
        { { 88.48800215647898, 0.0, 0.0, 88.48800215647898 },
            { 30.862002335592713, 0.0, 0.0, -30.862002335592702 },
            { 37.52777986769674, -20.535557255118455, 24.668353258793047, 19.444729299207314 },
            { 58.90087277950353, 24.239611461766692, -45.272650711571636, 28.846856811754705 },
            { 22.921351844871374, -3.704054206648225, 20.604297452778603, 9.334413709924226 } }
    };

    nn::FKSNetworks networks(legs, training_reruns, "../../models/3g2a/RAMBO/events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/");

    double python_outputs[2] = { 2.2266408e-07, 1.430258598666967e-06 };

    for (int i { 0 }; i < pspoints; ++i) {
        std::cout << "==================== Test point " << i + 1 << " ====================" << '\n';

        // standardise momenta
        std::vector<std::vector<std::vector<double>>> moms(training_reruns, std::vector<std::vector<double>>(pairs + 1, std::vector<double>(legs * 4)));

        // flatten momenta
        for (int p = 0; p < legs; p++) {
            for (int mu = 0; mu < 4; mu++) {
                // standardise input
                for (int k = 0; k < training_reruns; k++) {
                    for (int j = 0; j < pairs; j++) {
                        moms[k][j][p * 4 + mu] = nn::standardise(Momenta[i][p][mu], networks.metadatas[k][j][mu], networks.metadatas[k][j][4 + mu]);
                    }
                    moms[k][pairs][p * 4 + mu] = nn::standardise(Momenta[i][p][mu], networks.metadatas[k][pairs][mu], networks.metadatas[k][pairs][4 + mu]);
                }
            }
        }

        double s_com = Momenta[i][0][0] * Momenta[i][1][0] - (Momenta[i][0][1] * Momenta[i][1][1] + Momenta[i][0][2] * Momenta[i][1][2] + Momenta[i][0][3] * Momenta[i][1][3]);

        // cut/near check
        int cut_near = 0;
        for (int j = 0; j < legs - 1; j++) {
            for (int k = j + 1; k < legs; k++) {
                double prod = Momenta[i][j][0] * Momenta[i][k][0] - (Momenta[i][j][1] * Momenta[i][k][1] + Momenta[i][j][2] * Momenta[i][k][2] + Momenta[i][j][3] * Momenta[i][k][3]);
                double dist = prod / s_com;
                if (dist < delta) {
                    cut_near += 1;
                }
            }
        }

        // inference
        double results_sum = 0;
        for (int j = 0; j < training_reruns; j++) {
            if (cut_near >= 1) {
                // infer over all pairs
                for (int k = 0; k < pairs; k++) {
                    std::vector<double> result = networks.kerasModels[j][k].compute_output(moms[j][k]);
                    double output = nn::destandardise(result[0], networks.metadatas[j][k][8], networks.metadatas[j][k][9]);
                    results_sum += output;
                }
            } else {
                std::vector<double> result = networks.kerasModels[j][pairs].compute_output(moms[j][pairs]);

                double output = nn::destandardise(result[0], networks.metadatas[j][pairs][8], networks.metadatas[j][pairs][9]);

                results_sum += output;
            }
        }

        double average_output = results_sum / training_reruns;

        std::cout << "Python Loop( 0) = " << python_outputs[i] << '\n';
        std::cout << "C++    Loop( 0) = " << average_output << '\n';
    }
}
