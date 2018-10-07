//Helping functions

#include <iostream>
#include <math.h>
#include <vector>
#include <string>

using namespace std;

vector<string> get_info(string line){
	vector<string> info;
	string word = "";
	bool check = false;
	for(int i = 0; i < line.length(); i++){
		if(line[i] == ' ' || line[i] == ','){
			if(!check){
				check = true;
				info.push_back(word);
				word = "";
			}
		} else {
			word = word + line[i];
			check = false;
			if(i == line.length() - 1){
				info.push_back(word);
			}
		}
	}
	return info;
}

vector<int> random_shuffle(int n){
	vector<int> nums;

	for(int i = 0; i < n; i++)
		nums.push_back(i);

	int j;
	srand(time(NULL));

	for(int i = n - 1; i >= 0; i--){
		j = (int)((((double)rand())/RAND_MAX) * i + 1);
		int temp = nums[j];
		nums[j] = nums[i];
		nums[i] = temp;
	}

	return nums;
}

double dot(double* a, double* b, int n){
	double prod = 0.0;
	for(int i = 0; i < n; i++)
		prod += a[i]*b[i];
	return prod;
}

double sigmoid(double z){
	return 1.0 / (1.0 + exp(-1*z));
}

double d_sigmoid(double z){
	return sigmoid(z)*(1.0 - sigmoid(z));
}

double quad_reg(double a, double b){
	double result = b - a;
	return result*result;
}

double log_reg(double a, double b){
	return b*log(a) + (1.0 - b)*log(1.0 - a);
}
