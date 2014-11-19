% Вариант 2

% В одном городе живут семь любителей птиц. И фамилии у них птичьи.
% Каждый из них – "тезка" птицы, которой владеет один из его товарищей.
% У троих из них живут птицы, которые темнее, чем пернатые "тезки" их хозяев.
% "Тезка" птицы, которая живет у Воронова, женат.
% Голубев и Канарейкин – единственные холостяки из всей компании.
% Хозяин грача женат на сестре жены Чайкина.
% Невеста хозяина ворона очень не любит птицу, с которой возится ее жених.
% "Тезка" птицы, которая живет у Грачева, - хозяин канарейки.
% Птица, которая является "тезкой" владельца попугая, принадлежит "тезке" той птицы, которой владеет Воронов.
% У голубя и попугая оперение светлое.
% Кому принадлежит скворец?

tezka(voron, voronov).
tezka(golub, golubev).
tezka(kanareika, kanareikin).
tezka(grach, grachev).
tezka(chaika, chaikin).
tezka(popugai, popugaev).
tezka(skvorec, skvorcov).

solve(Ans, VladelecSkvorca):-
	Ans = [voronov/A, golubev/B, kanareikin/C, grachev/D, chaikin/E, popugaev/F, skvorcov/G],
	permutation([voron, golub, kanareika, grach, chaika, popugai, skvorec], [A, B, C, D, E, F, G]),
	A \= voron, B \= golub, C \= kanareika, D \= grach, E \= chaika, F \= popugai, G \= skvorec,
	Temnie = [voron, grach, skvorec], % Темные птицы
	member(VladelecVorona/voron, Ans), tezka(TezkaVV, VladelecVorona), not(member(TezkaVV, Temnie)),
	member(VladelecGracha/grach, Ans), tezka(TezkaVG, VladelecGracha), not(member(TezkaVG, Temnie)),
	member(VladelecSkvorca/skvorec, Ans), tezka(TezkaVS, VladelecSkvorca), not(member(TezkaVS, Temnie)),
	not(tezka(A, golubev)), not(tezka(A, kanareikin)), % Тёзка A женат, а Голубев и Канарейкин - холостяки
	B \= grach, C \= grach, % Хозяин грача - женат
	VladelecGracha \= chaikin, % Не может быть женат на сестре своей жены
	(B = voron; C = voron), % Раз у хозяина ворона есть невеста, то это кто-то из холостяков
	tezka(D, VladelecKanareiki), member(VladelecKanareiki/kanareika, Ans), % Тёзка птицы Грачёва - хозяин канарейки
	member(VladelecPopugaya/popugai, Ans), tezka(TezkaVP, VladelecPopugaya),
	member(VladelecTezkiVP/TezkaVP, Ans), tezka(TezkaVTVP, VladelecTezkiVP),
	member(voronov/TezkaVTVP, Ans).
