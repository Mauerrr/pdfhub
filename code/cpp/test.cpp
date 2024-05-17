#include <iostream>
#include <climits>
#define ZERO 0

int main(){
    using namespace std;
    
    char ch = 'M';
    int i = ch;
    cout << "The Ascii code for " << ch << " is " << i << endl;
    cout << "Add one to the character code: " << endl;
    ch = ch + 1;
    i = ch;
    cout << "The Ascii code for " << ch << " is " << i << endl;
    cout << "Displaying char ch using cout.put() ";
    cout.put(ch);
    cout.put('!');


    return 0;
    
}




