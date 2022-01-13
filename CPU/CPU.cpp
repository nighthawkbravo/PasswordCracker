
// sha256 C++ http://www.zedwood.com/article/cpp-sha256-function
#include "sha256.h"
#include <iostream>

using std::string;
using std::cout;
using std::endl;

int main()
{
    //std::cout << "Hello World!\n";
    string input = "grape";
    string output1 = sha256(input);

    cout << "sha256('" << input << "'):" << output1 << endl;
    return 0;
}