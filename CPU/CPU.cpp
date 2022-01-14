
// sha256 C++ http://www.zedwood.com/article/cpp-sha256-function
#include "sha256.h"
#include <iostream>
#include "RandomString.h"

// ASCII
// numbers 0-9 : 48-67
// Letters A-Z : 65-90
// Letters a-z : 97-122

using std::string;
using std::cout;
using std::endl;

int main()
{
    string input = "lucas";
    string output1 = sha256(input);

    RandomString RS(48, 122);

    cout << RS.getPassword() << endl;

    //cout << "sha256('" << input << "'): " << output1 << endl;
    return 0;
}