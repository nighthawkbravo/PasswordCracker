#include "randomString.h"

// ASCII 

using std::cout;
using std::endl;

RandomString::RandomString(int start, int end) {
	srand(time(NULL));

	//length = rand() % 11 + 5 // 5 - 10
	length = 10;
	char* wordChar = new char[10];

	for (int i = 0; i < length; ++i) {
		char c = char(rand() % (end + 1) + start);
		wordChar[i] = c;
		cout << wordChar[i] << " : " << c << endl;
	}

	

	//password = convertToString(wordChar, length);
	passwordHash = sha256(password);
}

bool RandomString::checkPassword(string guess) {
	return password.compare(guess);
}

string convertToString(char* a, int size)
{
	int i;
	string s = "";
	for (i = 0; i < size; i++) {
		s = s + a[i];
	}
	return s;
}