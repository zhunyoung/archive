#program preference(aso_test).

% This rule is not useful for LPOD translation, but is useful for ASO
% in the case when all atoms in the head are false but body is true
one(P) :- preference(P,aso_test), not holds(F) : preference(P,_,R,for(F),_), R>1.

% the first atom is true
one(P) :- preference(P,aso_test), holds(F), preference(P,_,1,for(F),_).

% the body is false
one(P) :- preference(P,aso_test), not holds(F), preference(P,_,0,for(F),_).


one'(P) :- preference(P,aso_test), not holds'(F) : preference(P,_,R,for(F),_), R>1.
one'(P) :- preference(P,aso_test), holds'(F), preference(P,_,1,for(F),_).
one'(P) :- preference(P,aso_test), not holds'(F), preference(P,_,0,for(F),_).

better(P) :- preference(P,aso_test), one(P), not one'(P).
better(P) :- preference(P,aso_test), 
	preference(P,_,R,for(F),_), holds(F), R > 1,
	not one'(P), not holds'(G) : preference(P,_,R',for(G),_), 1 < R', R'<=R.

bettereq(P) :- preference(P, aso_test), one(P).
bettereq(P) :- preference(P, aso_test),
	preference(P,_,R,for(F),_), holds(F), R > 1,
	not one'(P), not holds'(G) : preference(P,_,R',for(G),_), 1 < R', R'<R.

#program preference(pareto_test).

better(P) :- preference(P,pareto_test),
             better(R),    preference(P,_,_,name(R),_),
             bettereq(Q) : preference(P,_,_,name(Q),_).
