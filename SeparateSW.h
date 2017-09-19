#ifndef SW_CUH
#define SW_CUH

class Data;

class SeparateSW{
public:
	SeparateSW(const Data& txt, const Data& ptn, int block_num, int threshold = 0xffff);
	~SeparateSW();
private:
	void call_DP(const Data& txt, const Data& ptn);
	void checkScore(const char* direction, const int* score, const Data& txt, const Data& ptn) const;
	void traceback(const char* direction, const Data& txt, const Data& ptn, int txt_point, int ptn_point, int block) const;
	void show(const char* direction, const Data& txt, const Data& ptn) const;
	
	const int mBlock;
};


#endif





