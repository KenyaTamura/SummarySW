#include<iostream>
#include<fstream>
#include<string>

#include"Data.h"
#include"SimpleSW.h"

using namespace std;

int main(int argc, char* argv[]) {
/*
	if (argc == 1) {
		cout << "No entry\n";
		return 0;
	}
	else{
		// file read by Data class
		Data data(argv[1]);
	}
*/
	Data base(10000);	
	Data query(20);
	SimpleSW(base,query,10);
	return 0;
}
