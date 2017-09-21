#include<iostream>
#include<fstream>
#include<string>

/*
#include"Data.h"
#include"SimpleSW.h"
#include"Preprocess.h"
#include"PreprocessSW.h"
#include"SeparateSW.h"
#include"PostponeSW.h"
*/

#include"Measure.h"

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
/*
	Data base(10000);	
	Data query(20);
	int threshold = 10;
	cout << "Original = " << query.data() << endl;
	SimpleSW(base, query, threshold);
	cout << endl;
	PreprocessSW(base, query, Preprocess(base, query, threshold), threshold);
	cout << endl;
	SeparateSW(base, query, 8, threshold);
	cout << endl;
	PostponeSW(base, query, threshold);
*/
	
	Measure m("data.txt");	
	// ptn size, threshold, times
	m.exp(8, 4, 10);
	return 0;
}
