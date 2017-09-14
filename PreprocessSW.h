#ifndef SW_CUH
#define SW_CUH

class Data;
class Preprocess;

class SW{
public:
	SW(Data& txt, Data& ptn, Preprocess& prepro, int threshold = 0xffff);
	~SW();
private:
	void call_DP(Data& txt, Data& ptn, Preprocess& prepro);
	void checkScore(const char* direction, const int* score, Data& txt);
	void traceback(const char* direction, const Data& txt, int txt_point, int ptn_point); 
	void show(const char* score, Data& txt, Data& ptn);
};


#endif





