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

% Реализация предиката удаления последнего элемента в списке

% (список, список без последнего элемента)
delete_last([_], []).
delete_last([X | Y], [X | Z]):-delete_last(Y, Z).

% Реализация предиката перемножения всех элементов в списке

list_mul([], 0).
list_mul([H], H).
list_mul([H | T], N):-list_mul(T, P1), N is H * P1.
