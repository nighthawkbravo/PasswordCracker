#pragma once
#include <string>
#include "randomString.h"
#include <chrono>

using std::string;

class Algorithm {
protected:
	string passHash;
	int length;
	int* guess;

public:
	void solve(RandomString rs, int b, int t);	

	Algorithm(string p, int len) {
		passHash = p;
		length = len;
	}

	~Algorithm() {
		delete guess;
	}
};