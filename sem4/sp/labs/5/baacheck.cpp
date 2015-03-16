// baacheck_faq.cpp
#include "fsmcheck.h"
#include <queue>
#include <vector>

using namespace std;

tFSM::tStateSet tFSMcheck::unreached() //unreached, deadlocks
{
	report =
	"*** Developed by baa ***\n"
	"unreached: ";
	tFSM::tStateSet invalid; //создает пустое множество состояний
	
	// заносим все состояния в недостижимые кроме 0-го
	for (tFSM::tState s = 1; s < fsm.size(); ++s)
	{
		invalid.insert(s);
	}
	
	// просматривам все переходы из текущей вершины
	for (tFSM::tState s = 0; s < fsm.size(); ++s)
	{
		for (tFSM::tTransMap::const_iterator it = fsm.table[s].begin(); it != fsm.table[s].end(); ++it)
		{
			invalid.erase(it->second); // удаляем состояние из недостижимых
		}
	}

	return invalid;
}

int main()
{
	tFSM Afloat;
	tFSMcheck check(Afloat);
	////////////////////////////////////
	// построить тестовый автомат
	////////////////////////////////////

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
	// INFECTION
	addstr	(Afloat, 8, ".", 9);
	
	check.report_states(check.unreached());
	cout << check.get_report();

	//cin.get();
	return 0;
}
