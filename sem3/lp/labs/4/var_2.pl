% Вариант 2

% Реализовать разбор предложений английского языка.
% В предложениях у объекта (подлежащего) могут быть заданы цвет, размер, положение.
% В результате разбора должны получиться структуры представленные в примере.

% ?- sentence(["The", "big", "book", "is", "under", "the", "table"], X).
% ?- sentence(["The", "red", "book", "is", "on", "the", "table"], X).
% ?- sentence(["The", "little", "pen", "is", "red"], X).

% X = s(location(object(book, size(big)), under(table))).
% X = s(location(object(book, color(red)), on(table))).
% X = s(object(pen, size(little)), color(red)).

art(a).
art(an).
art(the).

item(book).
item(disk).
item(lamp).
item(pen).
item(bottle).
item(table).

color(red).
color(green).
color(blue).
color(yellow).
color(orange).

size(little).
size(medium).
size(big).

location(in, X, in(X)).
location(on, X, on(X)).
location(under, X, under(X)).
location(behind, X, behind(X)).
location(before, X, before(X)).
location(after, X, after(X)).

analize([H], s(H)).
analize([A, B], s(A, B)).

analize([Art, Size, Item | T], Res):-
	art(Art),
	size(Size),
	item(Item),
	analize([object(Item, size(Size)) | T], Res).

analize([Art, Color, Item | T], Res):-
	art(Art),
	color(Color),
	item(Item),
	analize([object(Item, color(Color)) | T], Res).

analize([object(Item, size(Size)), is, X, Y, Z | T], Res):-
	art(Y),
	item(Z),
	location(X, Z, Loc),
	analize([location(object(Item, size(Size)), Loc) | T], Res).

analize([object(Item, color(Color)), is, X, Y, Z | T], Res):-
	art(Y),
	item(Z),
	location(X, Z, Loc),
	analize([location(object(Item, color(Color)), Loc) | T], Res).

analize([object(Item, size(Size)), is, X | T], Res):-
	color(X),
	analize([object(Item, size(Size)), color(X) | T], Res).

analize([object(Item, color(Color)), is, X | T], Res):-
	size(X),
	analize([object(Item, color(Color)), size(X) | T], Res).
