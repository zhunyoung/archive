# Introduction

These three files are component programs in `asprin` and are used to find the Pareto-preferred answer sets of the following LPOD program named N_abc.
```
dom(1..n).

1{a(X): dom(X)}2.
1{c(X): dom(X)}2.

% b is true iff a is false
b(X) :- dom(X), not a(X).
:- a(X), b(X).

a(X) >> b(X) :- c(X).
```

We manually translate the N_abc program into an [`Answer Set Optimization`](http://www.cs.uky.edu/ai/papers.dir/aso-ijcai03.pdf) program, which is then translated into the language of `asprin`, including the [`P`](https://github.com/zhunyoung/archive/blob/master/LPOD_ASO_asprin/N_abc_P.txt), [`F_s`](https://github.com/zhunyoung/archive/blob/master/LPOD_ASO_asprin/N_abc_Fs.txt), and [`E_{t_s}`](https://github.com/zhunyoung/archive/blob/master/LPOD_ASO_asprin/E_ts.lp) components.

# Prerequisite
The utilization of these three files requires the installation of [`asprin`](https://github.com/potassco/asprin). You can install `asprin` via
```
pip install asprin
```

You can test if `asprin` is installed successfully via 
```
asprin --test
```

# How to use

You can find the Pareto-preferred answer sets of the above program (where n is, for example, assigned to 3) by executing
```
asprin N_abc_P.txt N_abc_Fs.txt E_ts.lp -c n=3
```
