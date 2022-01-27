
// sha256 C++ http://www.zedwood.com/article/cpp-sha256-function
#include "sha256.h"
#include <iostream>
#include "RandomString.h"
#include <string>
#include "Algorithm.h"


// ASCII
// numbers 0-9 : 48-57
// Letters A-Z : 65-90
// Letters a-z : 97-122

using std::string;
using std::cout;
using std::endl;

int main()
{
    int len = 4;
    RandomString RS(len);

    cout << endl;
    cout << "Password: " << RS.getPassword() << endl;
    cout << "Password Hash:" << RS.getHashPassword() << endl;

    Algorithm A(RS.getHashPassword(), len);

    A.solve(RS, 48, 122);

    return 0;
}

