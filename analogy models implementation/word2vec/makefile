CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec compute-accuracy-addition compute-accuracy-direction compute-accuracy-multiplication word2phrase distance word-analogy compute-accuracy

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh
compute-accuracy-addition : compute-accuracy-addition.c
	$(CC) compute-accuracy-addition.c -o compute-accuracy-addition $(CFLAGS)
	chmod +x *.sh
compute-accuracy-direction : compute-accuracy-direction.c
	$(CC) compute-accuracy-direction.c -o compute-accuracy-direction $(CFLAGS)
	chmod +x *.sh
compute-accuracy-multiplication : compute-accuracy-multiplication.c
	$(CC) compute-accuracy-multiplication.c -o compute-accuracy-multiplication $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec compute-accuracy-addition compute-accuracy-direction compute-accuracy-multiplication word2phrase distance word-analogy compute-accuracy