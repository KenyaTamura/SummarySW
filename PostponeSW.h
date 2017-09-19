#ifndef POSTPONESW_CUH
#define POSTPONESW_CUH

class Data;

class PostponeSW{
public:
	PostponeSW(const Data& txt, const Data& ptn, int threshold = 0xffff);
	~PostponeSW();
private:
	void call_DP(const Data& txt, const Data& ptn);
	void checkScore(const int* score, const Data& txt, const Data& ptn) const;
	void traceback(const char* direction, const Data& txt, int x, int y, int length) const; 
	void show(const char* score, const Data& txt, const Data& ptn, int start, int length) const;
};


#endif





