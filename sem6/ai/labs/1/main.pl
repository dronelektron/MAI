:- dynamic(answer/2).
:- dynamic(rule/1).
:- dynamic(type/1).
:- dynamic(visisted/1).
:- dynamic(is/1).

fact(X):-
	answer(X, Z),
	!,
	Z = yes.

fact(X):-
	writeln(X),
	read(A),
	asserta(answer(X, A)),
	A = yes.

rule(1):-
	not(visisted(1)),
	fact("много ног?"),
	fact("зеленый?"),
	fact("есть усики?"),
	asserta(type(гусеница)),
	asserta(visisted(1)).

rule(2):-
	not(visisted(2)),
	fact("есть крылья?"),
	fact("есть усики?"),
	fact("оранжевый?"),
	asserta(type(пчела)),
	asserta(visisted(2)).

rule(3):-
	not(visisted(3)),
	fact("розовый?"),
	asserta(is(лунтик)),
	asserta(visisted(3)).

rule(4):-
	not(visisted(4)),
	type(гусеница),
	fact("вишня?"),
	asserta(is(вупсень)),
	asserta(visisted(4)).

rule(5):-
	not(visisted(5)),
	type(гусеница),
	fact("слива?"),
	asserta(is(пупсень)),
	asserta(visisted(5)).

rule(6):-
	not(visisted(6)),
	fact("зеленый?"),
	fact("длинные ноги?"),
	asserta(is(кузя)),
	asserta(visisted(6)).

rule(7):-
	not(visisted(7)),
	fact("есть крылья?"),
	fact("красный?"),
	fact("женский род?"),
	asserta(is(мила)),
	asserta(visisted(7)).

rule(8):-
	not(visisted(8)),
	type(пчела),
	fact("женский род?"),
	asserta(is(капа)),
	asserta(visisted(8)).

rule(9):-
	not(visisted(9)),
	type(пчела),
	fact("есть фуражка?"),
	asserta(is(шер)),
	asserta(visisted(9)).

solve(X):-
	repeat,
	rule(_),
	is(X),
	!.
