#pragma once

#include <iostream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "sha256.h"

using std::string;

class RandomString {

private:
	string password;
	string passwordHash;
	int length;

	
public:
	RandomString(int start, int end);

	inline string getPasswordHash() { return passwordHash; }
	inline int getLength() { return length; }

	bool checkPassword(string guess);

	string convertToString(char* a, int size);

	inline string getPassword() { return password; }

};