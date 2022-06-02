all:
	gcc -c -fPIC c_ldpc.c
	mkdir -p ../bin
	gcc --shared c_ldpc.o -o ../bin/c_ldpc.so
