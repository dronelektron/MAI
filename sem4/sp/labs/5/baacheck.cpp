// baacheck.cpp
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

	invalid = fsm.finals; // заносим все терминирующие состояния в недостижимые
	
	// заносим все оставшиеся состояния в недостижимые кроме 0-го
	for (tFSM::tState s = 1; s < fsm.size(); ++s)
	{
		invalid.insert(s);
	}
	
	// обход в ширину графа
	used[0] = true; // начало с 0-го состояния

	q.push(0);

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

	// цикл, перечисляющий все команды автомата
	//   size_t n=fsm.size();//размер автомата
	//   for(size_t from=0; from<n; ++from){
	//    tFSM::tTransMap::const_iterator iter;
	//    for(iter=fsm.table[from].begin();
	//        iter!=fsm.table[from].end();
	//        ++iter){
	//         tFSM::tSymbol a=iter->first;
	//         tFSM::tState to=iter->second;
	//                 команда  (from,a) -> to
	//       }//for iter
	//    }// for from

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
	Afloat.final(8);
	Afloat.final(9);
	
	check.report_states(check.unreached());
	cout << check.get_report();

	//cin.get();
	return 0;
}
