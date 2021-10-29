/*********************************************************************
 * File  : MultilayerPerceptron.cpp
 * Date  : 2020
 *********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int randomInt(int Low, int High)
{
	return (rand() % ((High - Low) + 1)) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double randomDouble(double Low, double High)
{
	double f = (double)rand() / RAND_MAX;
	return Low + f * (High - Low);
}

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[])
{
	nOfLayers = nl;
	layers = new Layer[nOfLayers];

	for (int i = 0; i < nOfLayers; i++)
	{
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[layers[i].nOfNeurons];
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			if (i > 0)
			{
				int numWeightsInputs = layers[i - 1].nOfNeurons + 1;
				layers[i].neurons[j].w = new double[numWeightsInputs];
				layers[i].neurons[j].wCopy = new double[numWeightsInputs];
				layers[i].neurons[j].deltaW = new double[numWeightsInputs];
				layers[i].neurons[j].lastDeltaW = new double[numWeightsInputs];
				for (int k = 0; k < numWeightsInputs; k++)
				{
					layers[i].neurons[j].w[k] = 0;
					layers[i].neurons[j].wCopy[k] = 0;
					layers[i].neurons[j].deltaW[k] = 0;
					layers[i].neurons[j].lastDeltaW[k] = 0;
				}
			}
			else
			{
				layers[i].neurons[j].w = nullptr;
				layers[i].neurons[j].wCopy = nullptr;
				layers[i].neurons[j].deltaW = nullptr;
				layers[i].neurons[j].lastDeltaW = nullptr;
			}
			layers[i].neurons[j].out = 0;
			layers[i].neurons[j].delta = 0;
		}
	}

	return 1;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
	freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
	for (int i = 0; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			layers[i].neurons[j].out = 0;
			layers[i].neurons[j].delta = 0;
			delete[] layers[i].neurons[j].w;
			layers[i].neurons[j].w = nullptr;
			delete layers[i].neurons[j].deltaW;
			layers[i].neurons[j].deltaW = nullptr;
			delete layers[i].neurons[j].lastDeltaW;
			layers[i].neurons[j].lastDeltaW = nullptr;
			delete[] layers[i].neurons[j].wCopy;
			layers[i].neurons[j].wCopy = nullptr;
		}
		delete[] layers[i].neurons;
		layers[i].neurons = nullptr;
		layers[i].nOfNeurons = 0;
	}
	delete[] layers;
	layers = nullptr;
	nOfLayers = 0;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
			{
				layers[i].neurons[j].w[k] = randomDouble(-1, 1);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input)
{
	for (int i = 0; i < layers[0].nOfNeurons; i++)
	{
		layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output)
{
	for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
	{
		output[i] = layers[nOfLayers - 1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	// Hidden Layer Propagation
	for (int i = 1; i < nOfLayers - 1; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double sum = 0;
			int k;
			for (k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				sum += layers[i - 1].neurons[k].out * layers[i].neurons[j].w[k];
			}
			sum += layers[i].neurons[j].w[k];
			layers[i].neurons[j].out = 1 / (1 + exp(-sum));
		}
	}
	// Output Layer Propagation
	if (outputFunction == 0)
	{
		int ultimaCapa = nOfLayers - 1;
		for (int j = 0; j < layers[ultimaCapa].nOfNeurons; j++)
		{
			double sum = 0;
			int k;
			for (k = 0; k < layers[ultimaCapa - 1].nOfNeurons; k++)
			{
				sum += layers[ultimaCapa - 1].neurons[k].out * layers[ultimaCapa].neurons[j].w[k];
			}
			sum += layers[ultimaCapa].neurons[j].w[k];
			layers[ultimaCapa].neurons[j].out = 1 / (1 + exp(-sum));
		}
	}

	if (outputFunction == 1)
	{
		int ultimaCapa = nOfLayers - 1;
		double *nets = new double[layers[ultimaCapa].nOfNeurons];
		double sumExpNets = 0;
		for (int j = 0; j < layers[ultimaCapa].nOfNeurons; j++)
		{
			nets[j] = 0;
			int k;
			for (k = 0; k < layers[ultimaCapa - 1].nOfNeurons; k++)
			{
				nets[j] += layers[ultimaCapa].neurons[j].w[k] * layers[ultimaCapa - 1].neurons[k].out;
			}
			nets[j] += layers[ultimaCapa].neurons[j].w[k];
			sumExpNets += exp(nets[j]);
		}
		for (int j = 0; j < layers[ultimaCapa].nOfNeurons; j++)
		{
			layers[ultimaCapa].neurons[j].out = exp(nets[j]) / sumExpNets;
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double *target, int errorFunction)
{
	if (errorFunction == 0)
	{
		int outputSize = this->layers[nOfLayers - 1].nOfNeurons;
		double *outputs = new double[outputSize];
		this->getOutputs(outputs);
		double error = 0;
		for (int i = 0; i < outputSize; i++)
		{
			error += pow(target[i] - outputs[i], 2);
		}
		error /= outputSize;
		return error;
	}
	if (errorFunction == 1)
	{
		int outputSize = layers[nOfLayers - 1].nOfNeurons;
		double error = 0;
		for (int i = 0; i < outputSize; i++)
		{
			error += target[i] * log(layers[nOfLayers - 1].neurons[i].out);
		}
		error /= (double)outputSize;
		return error;
	}
	return -1.0;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double *target, int errorFunction)
{
	if (outputFunction == 0) // Neuronas Sigmoide
	{
		if (errorFunction == 0)
		{
			for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			{
				double output = layers[nOfLayers - 1].neurons[i].out;
				layers[nOfLayers - 1].neurons[i].delta = -(target[i] - output) * output * (1 - output);
			}
		}
		if (errorFunction == 1)
		{
			for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			{
				double output = layers[nOfLayers - 1].neurons[i].out;
				layers[nOfLayers - 1].neurons[i].delta = -(target[i] / output) * output * (1 - output);
			}
		}
	}
	if (outputFunction == 1) // Neuronas Softmax
	{
		// Delta on output layer
		if (errorFunction == 0)
		{
			for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			{
				double sumErrorDelta = 0;
				for (int j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++)
				{
					int factor = i == j ? 1 : 0;
					sumErrorDelta += (target[j] - layers[nOfLayers - 1].neurons[j].out) * layers[nOfLayers - 1].neurons[i].out * (factor - layers[nOfLayers - 1].neurons[j].out);
				}
				layers[nOfLayers - 1].neurons[i].delta = -sumErrorDelta;
			}
		}
		if (errorFunction == 1)
		{
			for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			{
				double sumErrorDelta = 0;
				for (int j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++)
				{
					int factor = i == j ? 1 : 0;
					sumErrorDelta += (target[j] / layers[nOfLayers - 1].neurons[j].out) * layers[nOfLayers - 1].neurons[i].out * (factor - layers[nOfLayers - 1].neurons[j].out);
				}
				layers[nOfLayers - 1].neurons[i].delta = -sumErrorDelta;
			}
		}
	}

	for (int i = nOfLayers - 2; i > 0; i--)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			double sumDeltaWeight = 0;
			double output = this->layers[i].neurons[j].out;
			for (int k = 0; k < this->layers[i + 1].nOfNeurons; k++)
			{
				sumDeltaWeight += this->layers[i + 1].neurons[k].delta * this->layers[i + 1].neurons[k].w[j];
			}
			this->layers[i].neurons[j].delta = sumDeltaWeight * output * (1 - output);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	for (int i = nOfLayers - 1; i > 0; i--)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			int k;
			for (k = 0; k < this->layers[i - 1].nOfNeurons; k++)
			{
				this->layers[i].neurons[j].deltaW[k] = this->layers[i].neurons[j].delta * this->layers[i - 1].neurons[k].out;
			}
			this->layers[i].neurons[j].deltaW[k] = this->layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < this->layers[i - 1].nOfNeurons + 1; k++)
			{
				double learningRate = this->eta * pow(this->decrementFactor, -(this->nOfLayers - i));
				this->layers[i].neurons[j].w[k] -= this->layers[i].neurons[j].deltaW[k] * learningRate - this->mu * this->layers[i].neurons[j].lastDeltaW[k] * learningRate;

				this->layers[i].neurons[j].lastDeltaW[k] = this->layers[i].neurons[j].deltaW[k];
			}
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
	std::cout << "Mostrando red neuronal" << std::endl;
	std::cout << "Tiene " << this->nOfLayers << " capas" << std::endl;
	for (int i = 0; i < this->nOfLayers; i++)
	{
		std::cout << "Mostrando capa: " << i << std::endl;
		std::cout << "Esta capa tiene: " << this->layers[i].nOfNeurons << " neuronas" << std::endl;
		if (i > 0)
		{
			for (int j = 0; j < this->layers[i].nOfNeurons; j++)
			{
				std::cout << "Mostrando pesos de la neurona " << j << " de la capa " << i << std::endl;
				for (int k = 0; k < this->layers[i - 1].nOfNeurons + 1; k++)
				{
					std::cout << this->layers[i].neurons[j].w[k] << std::endl;
				}
			}
		}
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double *input, double *target, int errorFunction)
{
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *MultilayerPerceptron::readData(const char *fileName)
{
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset *trainDataset, int errorFunction)
{
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset *dataset, int errorFunction)
{
}

// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset *dataset)
{
}

// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset *dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *salidas = new double[numSalidas];

	cout << "Id,Category" << endl;

	for (i = 0; i < dataset->nOfPatterns; i++)
	{

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;

		cout << i << "," << maxIndex << endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset *trainDataset, Dataset *testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;

	Dataset *validationDataset = NULL;
	double validationError = 0, previousValidationError = 0;
	int iterWithoutImprovingValidation = 0;

	// Generate validation data
	if (validationRatio > 0 && validationRatio < 1)
	{
		// ....
	}

	// Learning
	do
	{

		train(trainDataset, errorFunction);

		double trainError = test(trainDataset, errorFunction);
		if (countTrain == 0 || trainError < minTrainError)
		{
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if ((trainError - minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if (iterWithoutImproving == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		if (validationDataset != NULL)
		{
			if (previousValidationError == 0)
				previousValidationError = 999999999.9999999999;
			else
				previousValidationError = validationError;
			validationError = test(validationDataset, errorFunction);
			if (validationError < previousValidationError)
				iterWithoutImprovingValidation = 0;
			else if ((validationError - previousValidationError) < 0.00001)
				iterWithoutImprovingValidation = 0;
			else
				iterWithoutImprovingValidation++;
			if (iterWithoutImprovingValidation == 50)
			{
				cout << "We exit because validation is not improving!!" << endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while (countTrain < maxiter);

	if ((iterWithoutImprovingValidation != 50) && (iterWithoutImproving != 50))
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < testDataset->nOfPatterns; i++)
	{
		double *prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;
	}

	*errorTest = test(testDataset, errorFunction);
	;
	*errorTrain = minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);
}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				if (layers[i].neurons[j].w != NULL)
					f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				if (!(outputFunction == 1 && (i == (nOfLayers - 1)) && (k == (layers[i].nOfNeurons - 1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
