% Реализация стандартных предикатов обработки списков

% Принадлежность элемента списку
% (элемент, список)
my_member(X, [X | _]).
my_member(X, [_ | T]):-my_member(X, T).

% Длина списка
% (список, длина)
my_length([], 0).
my_length([_ | L], N):-my_length(L, M), N is M + 1.

% Конкатинация списков
% (список1, список2, список1+2)
my_append([], L, L).
my_append([X | L1], L2, [X | L3]):-my_append(L1, L2, L3).

% Удаление элемента из списка
% (элемент, список, список без элемента)
my_delete(X, [X | T], T).
my_delete(X, [Y | T], [Y | Z]):-my_delete(X, T, Z).

% Перестановки элементов в списке
% (список, перестановка)
my_permutation([], []).
my_permutation(L, [X | T]):-my_delete(X, L, Y), my_permutation(Y, T).

% Подсписки списка
% (подсписок, список)
my_sublist(S, L):-my_append(_, L1, L), my_append(S, _, L1).

% Удаление последнего элемента в списке
% (список, список без последнего элемента)
delete_last(L1, L2):-append(L2, [_], L1).

% Удаление последнего элемента в списке (без стандартных предикатов)
% (список, список без последнего элемента)
delete_last_ns([_], []).
delete_last_ns([X | Y], [X | Z]):-delete_last(Y, Z).

% Удаление последнего элемента в списке (для порядкового представления)
% (список, список без последнего элемента)

delete_last_ph(N, [e(N, _) | T], T).
delete_last_ph(N, [e(M, X) | T], [e(M, X) | Z]):-delete_last_ph(N, T, Z).
delete_last_p(L1, L2):-length(L1, N), delete_last_ph(N, L1, L2).

% Перемножение элементов в списке
% (список, результат)

list_mul([], 0).
list_mul([H], H).
list_mul(L, N):-append([_], L1, L), append([E], L1, L), list_mul(L1, P), N is E * P.

% Перемножение элементов в списке (без стандартных предикатов)
% (список, результат)
list_mul_ns([], 0).
list_mul_ns([H], H).
list_mul_ns([H | T], N):-list_mul(T, P), N is H * P.

% Перемножение элементов в списке (для порядкового представления)
% (список, результат)

list_mul_p([], 0).
list_mul_p([e(_, X)], X).
list_mul_p([e(_, X) | T], N):-list_mul_p(T, P), N is X * P.

% Пример совместного использования предикатов

is_palindrome([]).
is_palindrome([_]).
is_palindrome([H | T]):-delete_last(T, L1), my_append(L1, [E], T), E == H, is_palindrome(L1).
