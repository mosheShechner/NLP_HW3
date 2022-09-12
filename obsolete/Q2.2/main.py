import nltk
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk.grammar import Production, ProbabilisticProduction, PCFG
from nltk import Tree, Nonterminal
import matplotlib.pyplot as plt


def simplify_functional_tag(tag):
    if tag == "-NONE-":
        return tag
    if '-' in tag:
        tag = tag.split('-')[0]
    return tag

treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')


def get_tag(tree):
    if isinstance(tree, Tree):
        return Nonterminal(simplify_functional_tag(tree.label()))
    else:
        return tree

def tree_to_production(tree):
    return Production(get_tag(tree), [get_tag(child) for child in tree])

def tree_to_productions(tree):
    yield tree_to_production(tree)
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions(child):
                yield prod


def plot_freq(pcount):
    dict_vals = {}
    vals = pcount.values()
    for x in vals:
        dict_vals[x] = dict_vals.get(x,0) + 1

    s = sorted(dict_vals, key=dict_vals.get)
    print(dict_vals)
    x = s[-10:]
    y = []
    for x0 in x:
        y.append(dict_vals[x0])

    plt.bar(x,y)
    plt.show()


def pcfg_learn(treebank, n):
    trees = treebank.parsed_sents()[:n]
    pcount = {}
    lcount = {}
    for s in trees:
        curr = tree_to_productions(s)
        for prod in curr:
            if not ("-NONE-" in str(prod.lhs()) or "-NONE-" in str(prod.rhs())):
                lcount[prod.lhs()] = lcount.get(prod.lhs(), 0) + 1
                pcount[prod] = pcount.get(prod, 0) + 1

    prods = [
        ProbabilisticProduction(p.lhs(), p.rhs(), prob=pcount[p] / lcount[p.lhs()])
        for p in pcount
    ]

    incount = 0
    for p in prods:
        if len(p.rhs()) and not(isinstance(p.rhs()[0],Nonterminal)):
            incount += 1

    print(prods)
    print("number of internal nodes: " + str(incount))
    print("number of productions: " + str(len(pcount)))
    plot_freq(pcount)
    return PCFG(Nonterminal("S"), prods)

print("------  200 trees:")
res = pcfg_learn(treebank,200)
print("------  400 trees:")
res = pcfg_learn(treebank,400)




#--------------------------------------2.3-------------------------------------------------------


def pcfg_cnf_learn(treebank, n):
    trees = treebank.parsed_sents()[:n]
    pcount = {}
    lcount = {}
    for s in trees:
        nltk.treetransforms.chomsky_normal_form(s, factor='right', horzMarkov=1, vertMarkov=1, childChar='|',
                                                parentChar='^')
        curr = tree_to_productions(s)
        for prod in curr:
            if not ("-NONE-" in str(prod.lhs()) or "-NONE-" in str(prod.rhs())):
                lcount[prod.lhs()] = lcount.get(prod.lhs(), 0) + 1
                pcount[prod] = pcount.get(prod, 0) + 1
    prods = [
        ProbabilisticProduction(p.lhs(), p.rhs(), prob=pcount[p] / lcount[p.lhs()])
        for p in pcount
    ]

    print(prods)

    print("number of productions: " + str(len(pcount)))
    plot_freq(pcount)
    return PCFG(Nonterminal("S"), prods)

# print("------  200 trees:")
# #
# g = pcfg_cnf_learn(treebank,500)
# print("is cnf? " + str(g.is_chomsky_normal_form()))
# #---- 3.1
#
#
# vp=nltk.parse.viterbi.ViterbiParser(g)
# t = vp.parse(["Pierre", "Vinken", "will", "join", "the", "board"])
# for st in t:
#     print(st)
