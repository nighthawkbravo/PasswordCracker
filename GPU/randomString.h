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
	int* array;
	int length;

	int RandomLowerCase();
	int RandomUpperCase();
	int RandomNumber();
	int RandomChar();

public:
	RandomString(int len);
	~RandomString();

	inline int getLength() { return length; }

	bool checkPassword(string guess);

	string convertToString(char* a, int size);
	int* convertToIntArr(string s, int size);

	inline string getPassword() { return password; }
	inline string getHashPassword() { return passwordHash; }
};