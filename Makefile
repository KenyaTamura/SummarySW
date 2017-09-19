OBJS = main.o Data.o SimpleSW.o Preprocess.o PreprocessSW.o SeparateSW.o PostponeSW.o Measure.o Timer.o Writer.o
BIN = a.out
NVCC = nvcc -std=c++11 
GPP = g++ -std=c++11 -c -o

$(BIN): $(OBJS)
	$(NVCC) $(OBJS) -o $(BIN)

clean:
	rm *.o

.SUFFIXES: .o .cpp .cu

.cu.o: 
	$(NVCC) -c -o $@ $<

.cpp.o:
	$(GPP) -c -o $@ $<

