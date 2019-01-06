from nltk import PCFG
from nltk.grammar import Nonterminal
from nltk import probability

g = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
    Det -> 'the' [0.8] | 'my' [0.2]
    N -> 'man' [0.5] | 'telescope' [0.5]
    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
    V -> 'ate' [0.35] | 'saw' [0.65]
    PP -> P NP [1.0]
    P -> 'with' [0.61] | 'under' [0.39]
    """)





print("-------------------------------------------------------")


def getProbDist(productions):
    dict = {}
    for p in productions:
        dict[p.rhs()]=p.prob()
        print(p)
    return probability.DictionaryProbDist(dict)


def get_random_production_helper(nt,g):
    if not isinstance(nt, Nonterminal):
        return ""
    else:
        productions = g.productions(nt)
        p = getProbDist(productions)
        t = p.generate()
        res = ""
        for x in t:
            res =  res + " ( " + str(x) + " " + get_random_production_helper(x, g) + " )"
        return res


def get_random_production(nt,g):
    res = get_random_production_helper(nt,g)
    return "( S " + res + " )"


s= get_random_production(g.start(),g)
print(s)
