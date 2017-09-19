#ifndef SW_CUH
#define SW_CUH

class Data;

class PostPoneSW{
public:
	SW(Data& txt, Data& ptn, int threshold = 0xffff);
	~SW();
private:
	void call_DP(Data& txt, Data& ptn);
	void checkScore(const int* score, Data& txt, Data& ptn);
	void traceback(const char* direction, const Data& txt, int x, int y, int length); 
	void show(const char* score, Data& txt, Data& ptn, int start, int length);
};


#endif





