#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>   // To obtain current time time()
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h> // For DBL_MAX

#include "imc/MultilayerPerceptron.h"

int main(int argc, char *argv[])
{
  srand(5);
  imc::MultilayerPerceptron model;
  model.outputFunction = 1;
  int nl = 3;
  int npl[nl];
  npl[0] = 2;
  npl[1] = 1;
  npl[2] = 2;
  model.initialize(nl, npl);
  model.RandomWeights();
  model.seeNeuronalNetwork();
  double inputs[2];
  inputs[0] = 2.5;
  inputs[1] = -3.2;
  model.FeedInput(inputs);
  model.FordwardPropagate();
  double *outputs = new double[2];
  model.GetOutputs(outputs);
  std::cout << "Output: " << outputs[0] << " --- " << outputs[1] << std::endl;
  double target[2];
  target[0] = 0;
  target[1] = 1;
  double error = model.GetError(target, 1);
  std::cout<<"Error: "<<error<<std::endl;
  model.Backpropagate(target,1);
  model.ShowDeltas();
}