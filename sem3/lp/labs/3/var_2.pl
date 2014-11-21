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

delete_item([X | T], X, T).
delete_item([Y | T], X, [Y | Z]):-
	delete_item(T, X, Z).

count_k(L, N):-
	delete(L, m, L1),
	length(L1, M), N is M.

count_m(L, N):-
	delete(L, k, L1),
	length(L1, M), N is M.

balance(L):-
	count_m(L, X),
	count_k(L, Y),
	(X >= Y; X == 0).

go_right(L, R, Last):-
(
	% KMM ->
	Last \= [k, m, m],
	count_k(L, K), K > 0,
	count_m(L, M), M > 1,
	delete_item(L, k, L1),
	delete_item(L1, m, L2),
	delete_item(L2, m, LN),
	balance(LN),
	append(R, [k, m, m], RN),
	write('KMM ->'), nl,
	write(LN), write(' '), write(RN), nl,
	go_left(LN, RN, [k, m, m])
);
(
	% KM ->
	Last \= [k, m],
	count_k(L, K), K > 0,
	count_m(L, M), M > 0,
	delete_item(L, k, L1),
	delete_item(L1, m, LN),
	balance(LN),
	append(R, [k, m], RN),
	write('KM ->'), nl,
	write(LN), write(' '), write(RN), nl,
	go_left(LN, RN, [k, m])
).

go_left([], _, _).

go_left(L, R, Last):-
(
	% M <-
	Last \= [m],
	count_m(R, M), M > 0,
	delete_item(R, m, RN),
	balance(RN),
	append(L, [m], LN),
	write('M <-'), nl,
	write(LN), write(' '), write(RN), nl,
	go_right(LN, RN, [m])
);
(
	% KM <-
	Last \= [k, m],
	count_k(R, K), K > 0,
	count_m(R, M), M > 0,
	delete_item(R, k, R1),
	delete_item(R1, m, RN),
	balance(RN),
	append(L, [k, m], LN),
	write('KM <-'), nl,
	write(LN), write(' '), write(RN), nl,
	go_right(LN, RN, [k, m])
).
