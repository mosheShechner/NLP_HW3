from nltk import PCFG, ProbabilisticProduction, MLEProbDist, FreqDist
from nltk.grammar import Nonterminal, Production
from nltk import probability
import re
import numpy as np



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
            if isinstance(x,Nonterminal):
                res =  res + " (" + str(x) + " " + get_random_production_helper(x, g) + ")"
            else:
                res = res + str(x)
        return res


def pcfg_generate(g):
    res = get_random_production_helper(g.start(),g)
    return "(S" + res + ")"




print("-------------------------------------2.1.1----------------------------------")

from nltk.grammar import toy_pcfg2


def generate_corpus():
    f = open("toy_pcfg2.gen", "w+")
    for i in range(1000):
        s = pcfg_generate(toy_pcfg2)
        f.write(s + "\r\n")
    f.close()


#generate_corpus()
print("-------------------------------------2.1.2----------------------------------")

def create_tree(data):
    items = re.findall(r"\(|\)|\w+", data)
    def req(index):
        result = []
        item = items[index]
        while item != ")":
            if item == "(":
                subtree, index = req(index + 1)
                result.append(subtree)
            else:
                result.append(item)
            index += 1
            item = items[index]
        return result, index

    return req(1)[0]




def leaf(t):
    if len(t) != 2:
        return False
    for x in t:
        if isinstance(x, list):
            return False
    return True


def empirical_pcdg_helper(t,lcount,pcount):
    if leaf(t):
        nt = Nonterminal(t[0])
        lcount[nt] = lcount.get(nt,0)+1
        prod = Production(nt, [t[1]])
        pcount[prod] = pcount.get(prod,0) + 1
    else:
        left = Nonterminal(t[0])
        if len(t) == 2:
            right = Nonterminal(t[1][0])
            prod = Production(left,[right])
            lcount[left] = lcount.get(left, 0) + 1
            pcount[prod] = pcount.get(prod, 0) + 1
            empirical_pcdg_helper(t[1],lcount,pcount)
        else:
            right1 = Nonterminal(t[1][0])
            right2 = Nonterminal(t[2][0])
            lcount[left] = lcount.get(left, 0) + 1
            prod = Production(left,[right1,right2])
            pcount[prod] = pcount.get(prod, 0) + 1
            empirical_pcdg_helper(t[1],lcount,pcount)
            empirical_pcdg_helper(t[2], lcount, pcount)


def get_empirical_pcfg():
    global lcount, pcfg
    lcount = {}
    pcount = {}
    file = open("toy_pcfg2.gen", "r")
    lines = file.readlines()
    for s in lines:
        if s != "\n":
            t = create_tree(s)
            empirical_pcdg_helper(t, lcount, pcount)
    prods = [
        ProbabilisticProduction(p.lhs(), p.rhs(), prob=pcount[p] / lcount[p.lhs()])
        for p in pcount]
    pcfg = PCFG(toy_pcfg2.start(), prods)
    return pcfg



def getFreqDist(productions):
    dict = {}
    for p in productions:
        dict[p.rhs()]=p.prob()
    return FreqDist(dict)


pcfg = get_empirical_pcfg()
pcfg_nt = []
for p in pcfg.productions():
    if p.lhs() not in pcfg_nt:
        pcfg_nt.append(p.lhs())
dist_dict = {}
for nt in pcfg_nt:
    nt_prods = pcfg.productions(nt)
    fd = getFreqDist(nt_prods)
    dist_dict[nt] = fd




print("-------------------------------------2.1.3----------------------------------")

non_tr = []
for p in toy_pcfg2.productions():
    if not p.lhs() in non_tr:
        non_tr.append(p.lhs())



def get_kl(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.sum(a * np.log(a / b))


print(non_tr)
for nt in non_tr:
    mle = MLEProbDist(dist_dict[nt])
    prod_nt = toy_pcfg2.productions(nt)
    t_dist = []
    mle_dist = []
    for pr in prod_nt:
        t_dist.append(pr.prob())
        mle_dist.append(mle.prob(pr.rhs()))
    kl = get_kl(t_dist,mle_dist)
    print(str(nt) + ': ' + str(kl))

