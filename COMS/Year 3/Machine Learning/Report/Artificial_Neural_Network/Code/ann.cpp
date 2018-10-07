//Edargorter ANN

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <string>
#include "r_functions.h"

using namespace std;

#define BIAS 1.0

//Training Data
vector<string> classes;
vector<double*> x;
vector<double> y;
string bass_notes[] = {"C", "C#", "Db", "D", "D#", "Eb", "E", "E#", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"};
int bass_index[] = {1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12}; 

class Network{

private:

	//Network attributes
	double*** weights;
	double** net;
	double** errors;
	double** acc_errors;
	double* output;
	double* vector_value; // Placeholder for output classification vector
	double** biases;
	int* layers;

	int no_layers, end_training_points, end_testing_points, div, last_index, one_index;
	double rate, lambda, success_rate, peak_success_rate;

public:

	Network(int* layers, int no_layers, int end_training_points, int end_testing_points, double rate, double lambda){
		this->layers = layers;
		this->no_layers = no_layers;
		this->end_training_points = end_training_points;
		this->end_testing_points = end_testing_points;
		this->rate = rate;
		this->lambda = lambda;
		this->last_index = no_layers - 1;

		initialise_net();
		initialise_weights_biases();
		randomize_weights_set_biases();
	}

	void initialise_net(){
		net = new double*[no_layers];
		for(int i = 0; i < no_layers; i++)
			net[i] = new double[layers[i]];
		output = net[last_index]; // Set "output" to point to network's last layer.
	}

	void initialise_errors(){
		errors = new double*[no_layers];
		for(int i = 0; i < no_layers; i++)
			errors[i] = new double[layers[i]];
	}

	void initialise_acc_errors(){ // For stochastic gradient descent
		acc_errors = new double*[no_layers];
		for(int i = 0; i < no_layers; i++)
			acc_errors[i] = new double[layers[i]];
	}

	void initialise_weights_biases(){ // Initialise weights to random number between 0 and 1, with ...
		weights = new double**[no_layers];
		biases = new double*[no_layers];

		srand(time(NULL));

		for(int i = 1; i < no_layers; i++){ 

			weights[i] = new double*[layers[i]];
			biases[i] = new double[layers[i]];

			for(int j = 0; j < layers[i]; j++)
				weights[i][j] = new double[layers[i - 1]];					
		}
	}

	void randomize_weights_set_biases(){
		int i, j, k;
		for(i = 1; i < no_layers; i++){
			for(j = 0; j < layers[i]; j++){
				biases[i][j] = BIAS;
				for(k = 0; k < layers[i - 1]; k++)
					weights[i][j][k] = (((double)rand() / RAND_MAX)) * sqrt(2.0/layers[i - 1]); // He-et-al weight initialization, keeping in mind the pervious layer's size.
			}
		}
	}

	void print_weights_biases(){
		for(int i = 1; i < no_layers; i++){
			cout << "Layer [" << i << "] -> [" << i + 1 << "] :" << "\n\n";
			for(int j = 0; j < layers[i]; j++){
				for(int k = 0; k < layers[i - 1]; k++)
					cout << " " << weights[i][j][k];
				cout << endl << "BIAS: " << biases[i][j] << endl;
			}
		}
	}

	void print_layers(){
		for(int i = 0; i < no_layers; i++)
			cout << " " << layers[i];
		cout << endl;
	}

	void print_network(){
		for(int i = 0; i < no_layers; i++){
			cout << " [BIAS: " << biases[i] << "]";
			for(int j = 0; j < layers[i]; j++){
				cout << " " << net[i][j];
			}
			cout << endl;
		}
		cout << endl;
	}

	double regularization_term(int n){ // For regularization, keeping the weights small.
		double sum = 0.0, weight;

		for(int i = 1; i < no_layers; i++){
			for(int j = 0; j < layers[i]; j++){
				for(int k = 0; k < layers[i - 1]; k++){
					weight = weights[i][j][k];
					sum += weight*weight;
				}
			}
		}
		return lambda * sum / (2 * n);
	}

	void randomise_training_data(int max_index){ //Basic, Fisher-Yates random shuffle of full data set.
		int j;
		srand(time(NULL));
		for(int i = max_index - 1; i >= 0; i--){
			j = (int)((((double)rand())/RAND_MAX) * i + 1);
			double* x_temp = x[j];
			double y_temp = y[j];
			x[j] = x[i];
			y[j] = y[i];
			x[i] = x_temp;
			y[i] = y_temp;
		}
	}

	void print_vector_value(){
		for(int i = 0; i < layers[last_index]; i++)
			cout << " " << vector_value[i];
		cout << endl;
	}

	void update_vector_value(int index){ // Classifcation vector generator. For instance, the index of 2, given 5 possible classes, would correspond to: {0, 0, 1, 0, 0}
		vector_value[one_index] = 0.0;
		vector_value[index] = 1.0;
		one_index = index;
	}

	double z_value(int layer, int neuron){ // Dot product for inputs to neuron with respective weights, then add the bias.
		int prev_layer = layer - 1;
		double activation = dot(weights[layer][neuron], net[prev_layer], layers[prev_layer]); // in r_functions.h		
		return activation + biases[layer][neuron]; // add bias
	}

	double activate(double value){ // Sigmoid: 1/(1 + e^-z)
		return sigmoid(value);
	}

	void forward_propagate(double* input){ // Propagate input through network. The array variable "output" is the last layer of the network.
		for(int i = 0; i < layers[0]; i++){ 
			net[0][i] = input[i]; // Set first layer to input values.
		}
		for(int i = 1; i < no_layers; i++){
			for(int j = 0; j < layers[i]; j++){
				net[i][j] = activate(z_value(i, j)); //set neuron to activation of z value.
			}
		}
	}

	double validation_accuracy(){
		double highest_value, successes = 0.0;
		int highest_index;

		for(int i = end_testing_points; i < y.size(); i++){
			forward_propagate(x[i]);
			update_vector_value(y[i]);

			highest_value = 0.0;

			for(int j = 0; j < layers[last_index]; j++){
				if(output[j] > highest_value){ // Determine network's "guessed" class for input x[i].
					highest_index = j;
					highest_value = output[j];
				}
			}
			if(highest_index == y[i])
				successes++;
		}
		return successes / (y.size() - end_testing_points);
	}

	void print_validation_accuracy(){
		cout << "Validation Accuracy: " << 100 * validation_accuracy() << "%" << endl;
	}

	double cost(bool regularize){
		double error = 0.0, highest_value, successes = 0.0;
		int highest_index;

		for(int i = end_training_points; i < end_testing_points; i++){
			forward_propagate(x[i]);
			update_vector_value(y[i]); // Generate (Update current) classifcation vector, where the index of correct value is 1, while the other indices correspond to 0.

			highest_value = 0.0;

			for(int j = 0; j < layers[last_index]; j++){
				if(output[j] > highest_value){ // Determine network's "guessed" class for input x[i].
					highest_index = j;
					highest_value = output[j];
				}
				error += quad_reg(output[j], vector_value[j]); // Error function. In this case, the quadratic cost (see "r_functions.h").
			}
			if(highest_index == y[i]) // Update number of correct classifications.
				successes++;
		}
		success_rate = successes / (end_testing_points - end_training_points); // Success rate as fraction of total test data items.
		double cost_value = (error / (2 * end_training_points)); // quadratic_cost function suffices for analysis here; for logistic-regression cost: -1.0 * (error / points) + lambda * sum_squared_weights() / (2 * points);
		
		if(regularize)
			cost_value += regularization_term(end_training_points);

		return cost_value; 
	}

	void regularize_weights(){
		int i, j, k;
		for(i = 1; i < no_layers; i++){
			for(j = 0; j < layers[i]; j++){
				for(k = 0; k < layers[i - 1]; k++)
					weights[i][j][k] += (rate * lambda * weights[i][j][k]) / end_training_points; 
			}
		}
	}

	void update_stochastic_biases(int m){
		int i, j;
		for(i = 1; i < no_layers; i++){
			for(j = 0; j < layers[i]; j++)
				biases[i][j] += rate * acc_errors[i][j] / m;
		}
	}

	void update_stochastic_weights(int m){
		int i, j, k;
		for(i = 1; i < no_layers; i++){
			for(j = 0; j < layers[i]; j++){
				for(k = 0; k < layers[i - 1]; k++)
					weights[i][j][k] += (rate * acc_errors[i][j] * net[i - 1][k]) / m; 
 			}
		}
	}

	void update_biases(){
		int i, j;
		for(i = 1; i < no_layers; i++){
			for(j = 0; j < layers[i]; j++)
				biases[i][j] += rate * errors[i][j];
		}
	}

	void update_weights(){
		int i, j, k;
		for(i = 1; i < no_layers; i++){
			for(j = 0; j < layers[i]; j++){
				for(k = 0; k < layers[i - 1]; k++)
					weights[i][j][k] += rate * errors[i][j] * net[i - 1][k]; 
			}
		}
	}

	void set_acc_errors_zero(){
		for(int i = 0; i < no_layers; i++){
			for(int j = 0; j < layers[i]; j++)
				acc_errors[i][j] = 0.0;
		}
	}

	void accumulate_errors(){
		for(int i = 0; i < no_layers; i++){
			for(int j = 0; j < layers[i]; j++)
				acc_errors[i][j] += errors[i][j];
		}
	}

	void back_propagate(int index){ // Calculate neuron errors, back propagating through the network (Online learning).
		int i = last_index, next_layer;
		double error;

		forward_propagate(x[index]);
		update_vector_value(y[index]);

		for(int j = 0; j < layers[i]; j++)
			errors[i][j] = (output[j] - vector_value[j]) * d_sigmoid(output[j]);

		for(i = i - 1; i > 0; i--){
			error = 0.0;
			next_layer = i + 1;
			for(int j = 0; j < layers[i]; j++){
				for(int k = 0; k < layers[next_layer]; k++)
					error += weights[next_layer][k][j] * errors[next_layer][k];
				errors[i][j] = error * d_sigmoid(net[i][j]);
			}
		}
	}

	void print_info(int epoch, bool regularize){
		cout << "[" << epoch << "] Learning rate: " << rate << " Cost: " << cost(regularize) << " Success Rate: " << 100 * success_rate << "% "<< "Peak: " << 100 * peak_success_rate << "%" << endl;
	}

	void initialise_vector_value(){
		vector_value = new double[layers[last_index]];
		one_index = 0;
	}

	void stochastic_gradient_descent(int epochs, bool regularize, int m){ 
	// Trains network with mini-batches, averaging out each neuron's error over the training data set.
		int epoch = 0;
		peak_success_rate = 0.0;

		initialise_errors();
		initialise_acc_errors();
		initialise_vector_value();

		while(epoch < epochs){
			set_acc_errors_zero();

			for(int i = 0; i < m; i++){
				back_propagate(i);
				accumulate_errors();
			}

			update_stochastic_weights(m);
			update_stochastic_biases(m);

			if(regularize)
				regularize_weights();

			if(success_rate > peak_success_rate)
				peak_success_rate = success_rate;

			print_info(epoch, regularize);

			randomise_training_data(m);

			epoch++;
		}

	}

	void train_online(int epochs, bool regularize, string filename){ // Online learning training algorithm: updating weights and biases for each data point.
		//Run through epochs
		int epoch = 0;
		peak_success_rate = 0.0;

		initialise_errors();
		initialise_vector_value();

		ofstream out_file;
		out_file.open(filename + ".txt");

		while(epoch < epochs){
			for(int i = 0; i < end_training_points; i++){
				back_propagate(i);
				update_weights();
				update_biases();
			}

			if(regularize)
				regularize_weights();

			out_file << to_string(epoch) + "," + to_string(success_rate) + "\n";

			if(success_rate > peak_success_rate)
				peak_success_rate = success_rate;

			print_info(epoch, regularize);

			epoch++;
		}
		out_file.close();
	}
};

// Helpful functions for data import.

void print_array(double* arr, int length){ // Testing/Debugging purposes
	for(int i = 0; i < length; i++){
		cout << " " << arr[i];
	}
	cout << endl;
}

double* random_input(int num, int min, int max){ // Generates random vector input of size "num" with values between "max" and "min". (Neural Network testing purposes.)
	double* inputs = new double[num];
	int range = max - min;
	srand(time(NULL));
	for(int i = 0; i < num; i++){
		inputs[i] = ((double)rand() / RAND_MAX) * max + min;
	}
	return inputs;
}

double get_note_value(string item){ // YES -> 1.0, NO -> 0.0
	return (item == "YES") ? 1.0 : 0.0;
}

double get_chord_value(string chord){
	for(int i = 0; i < classes.size(); i++){
		if(chord.compare(classes[i]) == 0)
			return (double)i;
	}
	return -1.0;
}

double get_note_class(string note){
	for(int i = 0; i < 18; i++){
		if(note == bass_notes[i])
			return bass_index[i];
	}
	return 0.0;
}

double* get_vector_class(int index, int no_classes){
	double* v = new double[no_classes];
	for(int i = 0; i < no_classes; i++)
		v[i] = 0.0;
	v[index] = 1.0;
	return v;
}

void process_training_data(string filename){ // Process training data, obtain vector classes for each input x[i], and obtain chord values for y[i].
	ifstream file(filename);
	string strline;
	vector<string> lines, line;
	int count = 1;
	double* v;

	while(getline(file, strline)){
		line = get_info(strline);
		x.push_back(new double[29]);
		for(int i = 0; i < 12; i++)
			x.back()[i] = get_note_value(line[i + 2]);
		v = get_vector_class(get_note_class(line[14]) - 1, 12);
		for(int i = 0; i < 12; i++)
			x.back()[i + 12] = v[i];
		v = get_vector_class(stoi(line[15]) - 1, 5);
		for(int i = 0; i < 5; i++){
			x.back()[i + 24] = v[i];
		}
		y.push_back(get_chord_value(line[16]));
	}
}

void process_classes(string filename){
	ifstream file(filename);
	string line;
	while(getline(file, line)){
		classes.push_back(line);
	}
}

// Execute in terminal with the following format: ./ann [filename] [number of epochs] [learning rate (negative)] [first layer size] [second layer size] ... [Regularization: 1|0]
int main(int argc, char** argv){ 

	if(argc < 6){
		cout << "No ANN architecture passed..." << endl;
		return 0;
	}

	int epochs, layers_size = argc - 5;
	int layers[layers_size];
	double learning_rate;
	bool regularization = true;

	string filename = argv[1];

	try{
		epochs = stoi(argv[2]);
		learning_rate = stod(argv[3]);
		if(stoi(argv[argc - 1]) < 1)
			regularization = false;
	} catch (exception& e){
		cout << "Type error.";
	}

	for(int i = 4; i < argc - 1; i++){
		try{
			layers[i - 4] = stoi(argv[i]);
		} catch (exception& e) {
			cout << "Type error.";
		}
	}

	cout << "Parameters accepted." << endl;

	//double* input = random_input(layers[0], 0, 1);

	process_classes("chord_index.txt");
	process_training_data(filename);

	srand(time(NULL));

	int end_training_set = 0.6 * y.size();
	int end_testing_set = 0.8 * y.size();

	Network n = Network(&layers[0], layers_size, end_training_set, end_testing_set, learning_rate, 0.1);
	
	/*
	n.print_layers();
	n.randomise_training_data(y.size());
	//n.stochastic_gradient_descent(epochs, regularization, 0.4 * y.size());
	n.train_online(epochs, regularization);
	n.print_validation_accuracy();
	*/
	
	double accumacc = 0.0;

	for(int i = 0; i < 5; i++){
		n.randomise_training_data(y.size());
		n.train_online(epochs, false, "no_reg_" + to_string(i));
		accumacc += n.validation_accuracy();
		n.randomize_weights_set_biases();
	}

	cout << "Average Validation Accuracy ( No regularization ): " << 100 * accumacc / 5 << endl; 

	accumacc = 0.0;

	for(int i = 0; i < 5; i++){
		n.randomise_training_data(y.size());
		n.train_online(epochs, true, "reg_" + to_string(i));
		accumacc += n.validation_accuracy();
		n.randomize_weights_set_biases();
	}

	cout << "Average Validation Accuracy ( Regularization ): " << 100 * accumacc / 5 << endl; 

	return 1;
}