#include "randomString.h"

// ASCII 

using std::cout;
using std::endl;

RandomString::RandomString(int len) {
	srand(time(NULL));

	//length = rand() % 11 + 5 // 5 - 10
	char* wordChar = new char[len];

	for (int i = 0; i < len; ++i) {
		int k = rand() % 3; // 0,1,2
		char c;
		/*switch (k) {
			case 0:
				c = char (RandomLowerCase());
				break;
			case 1:
				c = char(RandomUpperCase());
				break;
			case 2:
				c = char(RandomNumber());
				break;
		}*/
		c = char(RandomChar());

		wordChar[i] = c;
		//cout << wordChar[i] << " : " << c << " : "<< int(c) << endl;
	}

	password = convertToString(wordChar, len);

	//password = convertToString(wordChar, length);
	passwordHash = sha256(password);
}

bool RandomString::checkPassword(string guess) {
	if (passwordHash.compare(guess) == 0) return true;
	return false;
}

string RandomString::convertToString(char* a, int size)
{
	int i;
	string s = "";
	for (i = 0; i < size; i++) {
		s = s + a[i];
	}
	return s;
}

// Letters a-z : 97-122
int RandomString::RandomLowerCase() {
	return rand() % (25 + 1) + 97;
}
// Letters A-Z : 65-90
int RandomString::RandomUpperCase() {
	return rand() % (25 + 1) + 65;
}
// numbers 0-9 : 48-57
int RandomString::RandomNumber() {
	return rand() % (9 + 1) + 48;
}

int RandomString::RandomChar() {
	return rand() % (74 + 1) + 48;
}

