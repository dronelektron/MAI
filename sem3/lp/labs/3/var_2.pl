% Вариант 2

% Три миссионера и три каннибала хотят переправиться с левого берега реки на правый.
% Как это сделать за минимальное число шагов, если в их распоряжении имеется трехместная лодка
% и ни при каких обстоятельствах (в лодке или на берегу) миссионеры не должны оставаться в меньшинстве.

% Решение
% [K, K, K, M, M, M], []
% 1) KM ->
% [K, K, M, M], [K, M]
% 2) M <-
% [K, K, M, M, M], [K]
% 3) KMM ->
% [K, M], [K, K, M, M]
% 4) KM <-
% [K, K, M, M], [K, M]
% 5) KMM ->
% [K], [K, K, M, M, M]
% 6) M <-
% [K, M], [K, K, M, M]
% 7) KM ->
% [], [K, K, K, M, M, M]

count_k(L, N):-
	delete(L, 'm', L1),
	length(L1, M), N is M.

count_m(L, N):-
	delete(L, 'k', L1),
	length(L1, M), N is M.

balance(L):-
	count_m(L, X),
	count_k(L, Y),
	(X >= Y; X == 0).

sort_items(A, B):-
	append(Begin, ['m', 'k' | Tail], A),
	append(Begin, ['k', 'm' | Tail], C),
	sort_items(C, B);
	append(A, [], B).

% KM ->
step(A, B):-
	append(Left, ['>' | Right], A),
	append(Begin, ['k', 'm' | Tail], Left),
	append(Right, ['k', 'm'], Right2),
	balance(Right2),
	sort_items(Right2, RightSorted),
	append(Begin, Tail, L),
	balance(L),
	append(L, ['<'], L1),
	append(L1, RightSorted, B).

% KMM ->
step(A, B):-
	append(Left, ['>' | Right], A),
	append(Begin, ['k', 'm', 'm' | Tail], Left),
	append(Right, ['k', 'm', 'm'], Right2),
	balance(Right2),
	sort_items(Right2, RightSorted),
	append(Begin, Tail, L),
	balance(L),
	append(L, ['<'], L1),
	append(L1, RightSorted, B).

% M <-
step(A, B):-
	append(Left, ['<' | Right], A),
	append(Begin, ['m' | Tail], Right),
	append(['m'], Left, Left2),
	balance(Left2),
	sort_items(Left2, LeftSorted),
	append(Begin, Tail, R),
	balance(R),
	append(['>'], R, R1),
	append(LeftSorted, R1, B).

% KM <-
step(A, B):-
	append(Left, ['<' | Right], A),
	append(Begin, ['k', 'm' | Tail], Right),
	append(['k', 'm'], Left, Left2),
	balance(Left2),
	sort_items(Left2, LeftSorted),
	append(Begin, Tail, R),
	balance(R),
	append(['>'], R, R1),
	append(LeftSorted, R1, B).

% Печать решения

print_way([_]).
print_way([_, B | Tail]):-
	print_way([B | Tail]),
	nl,
	write(B).
	
print_answer([A | _]):-
	nl,
	write(A),
	nl.

prolong([Temp | Tail], [New, Temp | Tail]):-
	step(Temp, New),
	not(member(New, [Temp | Tail])).

% Поиск в глубину

depth_search(Start, Finish):-
	depth([Start], Finish, Way),
	print_way(Way),
	print_answer(Way).

depth([Finish | Tail], Finish, [Finish | Tail]).
depth(TempWay, Finish, Way):-
	prolong(TempWay, NewWay),
	depth(NewWay, Finish, Way).

% Поиск в ширину

breadth_search(Start, Finish):-
	breadth([[Start]], Finish, Way),
	print_way(Way),
	print_answer(Way).

breadth([[Finish | Tail] | _], Finish, [Finish | Tail]).
breadth([TempWay | OtherWays], Finish, Way):-
	findall(W, prolong(TempWay, W), Ways),
	append(OtherWays, Ways, NewWays),
	breadth(NewWays, Finish, Way).

% Поиск с итерационным углублением

int(1).
int(N):- int(M), N is M + 1.

iter_search(Start, Finish):-
	int(Level),
	(
		Level > 100, !;
		id([Start], Finish, Way, Level), print_way(Way), print_answer(Way)
	).

id([Finish | Tail], Finish, [Finish | Tail], 0).
id(TempWay, Finish, Way, N):-
	N > 0,
	prolong(TempWay, NewWay),
	N1 is N - 1,
	id(NewWay, Finish, Way, N1).

:- use_module(library(statistics)).
