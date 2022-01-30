all: passwordCrackerCPU
.PHONY: all

# CC compiler options:
CC=g++

sha256.o: sha256.cpp sha256.h
	$(CC) -c sha256.cpp

randomString.o: randomString.cpp randomString.h
	$(CC) -c randomString.cpp

Algorithm.o: Algorithm.cpp Algorithm.h
	$(CC) -c Algorithm.cpp

CPU.o: CPU.cpp
	$(CC) -c CPU.cpp

passwordCrackerCPU: CPU.o Algorithm.o sha256.o randomString.o
	$(CC) -o randomString.o sha256.o Algorithm.o CPU.o

clean:
	rm passwordCrackerCPU CPU.o sha256.o randomString.o Algorithm.o
	
