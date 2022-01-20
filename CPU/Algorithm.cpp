#include "Algorithm.h"

void Algorithm::solve(RandomString rs, int b, int t) {

	guess = new int[length];
	char* guessChar = new char[length];
	int diff = t - b;

	for (int i = 0; i < length; ++i) {
		guess[i] = b;
		guessChar[i] = char(b);
	}

	int i = 0;
	int j = 0;

	string s = rs.convertToString(guessChar, length);
	auto start = std::chrono::high_resolution_clock::now();
	while (!rs.checkPassword(sha256(s))) {

		for (int i = 0; i < length; ) {
			if (guess[i] < t) {
				guess[i]++;
				guessChar[i] = char(guess[i]);
				break;
			}
			else {
				guess[i] = b;
				guessChar[i] = char(b);
				i++;
			}
		}
		s = rs.convertToString(guessChar, length);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "Password Cracked: " << s;
	std::cout << " Time: " << duration.count() << "ns (nanoseconds)" << std::endl;

	std::cout << "Password Cracked: " << s;
	std::cout << " Time: " << duration2.count() << " (milliseconds)" << std::endl;

	std::cout << "Password Cracked: " << s;
	std::cout << " Time: " << duration.count() / 1000000000.0 << " (seconds)" << std::endl;
}