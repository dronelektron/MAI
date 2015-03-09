//                 lexer.h 2015
#ifndef LEXER_H
#define LEXER_H
#include "baselexer.h"

//********************************************
//*        Developed by baa                  *
//*             (c)  2015                    *
//********************************************

class tLexer : public tBaseLexer
{
public:
	// персональный код разработчика
	std::string Authentication() const
	{
		return "baa";
	}

	//конструктор
	tLexer() : tBaseLexer()
	{
		//создать автоматы:

		// #1 целое
		addstr  (Aint, 0, "+-", 2);
		addstr  (Aint, 0, "0", 1);
		addstr  (Aint, 1, "0", 1);
		addstr  (Aint, 2, "0", 1);
		addrange(Aint, 0,'1','9',3);
		addrange(Aint, 2,'1','9',3);
		addrange(Aint, 3,'0','9',3);
		Aint.final(1);
		Aint.final(3);

		//________________________________________

		// #2 восьмеричное Aoct
		addstr  (Aoct, 0, "0", 1);
		addstr  (Aoct, 0, "+-", 2);
		addstr  (Aoct, 1, "0", 1);
		addrange(Aoct, 1, '1', '7', 3);
		addstr  (Aoct, 2, "0", 1);
		addrange(Aoct, 3, '0', '7', 3);
		Aoct.final(3);

		//________________________________________

		// #3 вещественное  Afloat

		addstr  (Afloat, 0, "+-", 1);
		addrange(Afloat, 1, '0', '9', 2);
		addrange(Afloat, 2, '0', '9', 2);
		addstr  (Afloat, 2, ".", 3);
		addrange(Afloat, 3, '0', '9', 4);
		addrange(Afloat, 4, '0', '9', 4);
		addstr  (Afloat, 4, "eE", 5);
		addstr  (Afloat, 5, "+-", 6);
		addrange(Afloat, 6, '0', '9', 7);
		addrange(Afloat, 7, '0', '9', 7);
		Afloat.final(7);

		//________________________________________

		// #4 идентификатор  Aid

		addstr  (Aid, 0, "!", 1);
		addrange(Aid, 0, 'a', 'z', 1);
		addrange(Aid, 0, 'A', 'Z', 1);
		addstr  (Aid, 1, "!", 1);
		addstr  (Aid, 1, "-", 1);
		addrange(Aid, 1, '0', '9', 1);
		addrange(Aid, 1, 'a', 'z', 1);
		addrange(Aid, 1, 'A', 'Z', 1);
		Aid.final(1);

		//________________________________________

		// #5 идентификатор предиката Aidq

		addstr  (Aidq, 0, "?", 1);
		addrange(Aidq, 0, 'a', 'z', 1);
		addrange(Aidq, 0, 'A', 'Z', 1);
		addstr  (Aidq, 1, "-", 1);
		addstr  (Aidq, 1, "?", 2);
		addrange(Aidq, 1, '0', '9', 1);
		addrange(Aidq, 1, 'a', 'z', 1);
		addrange(Aidq, 1, 'A', 'Z', 1);
		Aidq.final(1);
		Aidq.final(2);

		//________________________________________

		// #6 оператор Aoper

		addstr(Aoper, 0, "+-*/=", 1);
		addstr(Aoper, 0, "<>", 2);
		addstr(Aoper, 2, "=", 1);
		Aoper.final(1);
		Aoper.final(2);

		//________________________________________

		// #7 булевские константы Abool

		addstr(Abool, 0, "#", 1);
		addstr(Abool, 1, "tf", 2);
		Abool.final(2);

		//________________________________________

		// #8 строка
		// ФАКУЛЬТАТИВНОЕ ЗАДАНИЕ*
		// дополнить esc-последовательностями \" и \\
		// тест L8.ss
		addstr  (Astr, 0, "\"", 1);
		addstr  (Astr, 1, "\"", 2);
		addrange(Astr, 1, ' ','"' - 1, 1);
		addrange(Astr, 1, '"' + 1, '\\' - 1, 1);
		addrange(Astr, 1, '\\' + 1, '~', 1);
		addstr	(Astr, 1, "\\", 3);
		addstr	(Astr, 3, "\\\"", 1);
		addrange(Astr, 1, -128, -1, 1);
		Astr.final(2);
		
		//________________________________________
	}
};

#endif
