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
	std::queue<tFSM::tState> q; // очередь для обхода в ширину
	std::vector<bool> used(fsm.size(), false); // вектор достигнутых вершин
	
	// заносим все состояния в недостижимые кроме 0-го
	for (tFSM::tState s = 1; s < fsm.size(); ++s)
	{
		invalid.insert(s);
	}
	
	// обход графа в ширину
	used[0] = true; 
	
	q.push(0); // начало с 0-го состояния

	while (!q.empty())
	{
		tFSM::tState state = q.front(); // текущее состояние

		q.pop(); // убираем текущее состояние из очереди
		
		invalid.erase(state); // удаляем состояние из недостижимых

		// добавляем в очередь все состояния, которые достижимы из текущего
		for (tFSM::tTransMap::const_iterator it2 = fsm.table[state].begin(); it2 != fsm.table[state].end(); ++it2)
		{
			// если состояние не было еще в очереди, то добавлем его
			if (!used[it2->second])
			{
				used[it2->second] = true;
				q.push(it2->second);
			}
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
