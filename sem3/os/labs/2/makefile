all:
	nasm --prefix _ -f macho bubble_sort.asm -o bubble_sort.o
	gcc -m32 -c main.c -o main.o
	gcc -m32 main.o bubble_sort.o
	rm *.o
