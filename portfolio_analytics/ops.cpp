/*
 *
 * Finance c++
 */

#include <cmath>
#include <iostream>
#include <statistics_calculator.h>
extern "C" {

    int add_integers(int a, int b) {
        return a + b;
    }
    double compute_variability(const double stock_prices[], double variabilities[], int array_length) {
        const auto calculator = new StatisticsCalculator();
        calculator->setDataFromArray(stock_prices, array_length);
        calculator->convertToRelativeChanges();

        const int rollingWindowSize = 20;
        int firstVariabilityIndex = 0;
        for(int i = 0; i < array_length; i++) {
            if (i < rollingWindowSize) {
                variabilities[i] = 0;
                firstVariabilityIndex++;
                continue;
            } 

            calculator->setRollingWindow(i - rollingWindowSize, rollingWindowSize);

            const double currentSigma = calculator->calculateRollingStandardDeviation();

            variabilities[i] = currentSigma * std::sqrt(255);
        }
        delete calculator;
        double variabilitiesMean = 0;
        for (int i = firstVariabilityIndex; i < array_length; i++) {
            std::cout<<"compute_variability: variabilities["<<i<<"]="<<variabilities[i]<<std::endl;
            variabilitiesMean += variabilities[i];
        }
        variabilitiesMean /= (array_length-firstVariabilityIndex);
        std::cout<<"operations.compute_variability: mean: "<<variabilitiesMean<<std::endl;

        return variabilitiesMean;
    }
}
