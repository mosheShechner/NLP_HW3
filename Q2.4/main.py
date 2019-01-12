from nltk import Tree
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
import matplotlib.pyplot as plt
import numpy as np


def simplify_functional_tag(tag):
    if '-' in tag:
        tag = tag.split('-')[0]
    return tag

treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')

def plot_hist(dict):
    sorted_keys = sorted(dict, key=dict.get, reverse=True)
    x = []
    y = []
    for i in range(10):
        x.append(sorted_keys[i])
        y.append(dict[sorted_keys[i]])
    plt.bar(x, y)
    plt.show()



np_counter = [0]
np_dict = {}



def get_np_counters_sent(t,c,rhs_d):
    curr_label = simplify_functional_tag(t.label())
    if curr_label == "NP":
        c[0] = c[0] + 1
        for i in range(len(t)):
            if type(t[i])==Tree:
                rhs_d[simplify_functional_tag(t[i].label())] = rhs_d.get(simplify_functional_tag(t[i].label()),0) + 1
                get_np_counters_sent(t[i],c,rhs_d)
    else:
        for i in range(len(t)):
            if type(t[i]) == Tree:
                get_np_counters_sent(t[i],c,rhs_d)


def get_np_counters(sent_list,c,rhs_d):
    for s in sent_list:
        get_np_counters_sent(s,c,rhs_d)



get_np_counters(treebank.parsed_sents(),np_counter,np_dict)
print("Number of NP's = " + str(np_counter[0]))
print("hist:")
print(np_dict)
plot_hist(np_dict)


print("-----------------------------------------2.4.2----------------------------------")

np_below_s_counter = [0]
np_below_s_dict = {}

def get_np_below_s_counters_sent(t,c,rhs_d,parent):
    curr_label = simplify_functional_tag(t.label())
    if curr_label == "NP" and parent == "S":
        c[0] = c[0] + 1
        for i in range(len(t)):
            if type(t[i])==Tree:
                rhs_d[simplify_functional_tag(t[i].label())] = rhs_d.get(simplify_functional_tag(t[i].label()),0) + 1
                get_np_below_s_counters_sent(t[i],c,rhs_d,curr_label)
    else:
        for i in range(len(t)):
            if type(t[i]) == Tree:
                get_np_below_s_counters_sent(t[i],c,rhs_d,curr_label)


def get_np_below_s_counters(sent_list,c,rhs_d):
    for s in sent_list:
        get_np_below_s_counters_sent(s,c,rhs_d,s.label())


get_np_below_s_counters(treebank.parsed_sents(),np_below_s_counter,np_below_s_dict)
print("Number of NP's under S = " + str(np_below_s_counter[0]))
print("hist:")
print(np_below_s_dict)
plot_hist(np_below_s_dict)


print("-----------------------------------------2.4.3----------------------------------")

np_below_vp_counter = [0]
np_below_vp_dict = {}

def get_np_below_vp_counters_sent(t,c,rhs_d,parent):
    curr_label = simplify_functional_tag(t.label())
    if curr_label == "NP" and parent == "VP":
        c[0] = c[0] + 1
        for i in range(len(t)):
            if type(t[i])==Tree:
                rhs_d[simplify_functional_tag(t[i].label())] = rhs_d.get(simplify_functional_tag(t[i].label()),0) + 1
                get_np_below_vp_counters_sent(t[i],c,rhs_d,curr_label)
    else:
        for i in range(len(t)):
            if type(t[i]) == Tree:
                get_np_below_vp_counters_sent(t[i],c,rhs_d,curr_label)


def get_np_below_vp_counters(sent_list,c,rhs_d):
    for s in sent_list:
        get_np_below_vp_counters_sent(s,c,rhs_d,s.label())


get_np_below_vp_counters(treebank.parsed_sents(),np_below_vp_counter,np_below_vp_dict)
print("Number of NP's under VP = " + str(np_below_vp_counter[0]))
print("hist:")
print(np_below_vp_dict)
plot_hist(np_below_vp_dict)

print("-----------------------------------------2.4.4----------------------------------")


def dict_to_dist(d,s):
    for x in d:
        d[x] = d[x]/s
    return d

def get_joint_support(p,q):
    supp = []
    for x in p.keys():
        if x not in supp:
            supp.append(x)
    for x in q.keys():
        if x not in supp:
            supp.append(x)
    return supp

p = dict_to_dist(np_below_s_dict,np_below_s_counter[0])
q = dict_to_dist(np_below_vp_dict,np_below_vp_counter[0])
print(p)
print(q)

sp = p.keys()
cp = len(sp)

sq = q.keys()
cq = len(sq)

su = get_joint_support(np_below_vp_dict,np_below_s_dict)
cu = len(su)

eps = 0.00001

pc = eps*(cu-cp)/cp
qc = eps*(cu-cq)/cq

p_tag = {}
q_tag = {}

for x in su:
    if x in sp:
        p_tag[x] = p[x] - pc
    else:
        p_tag[x] = eps
    if x in sq:
        q_tag[x] = q[x] - qc
    else:
        q_tag[x] = eps

print(p_tag)
print(q_tag)

def get_kl(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.sum(a * np.log(a / b))

kl = get_kl([x for x in p_tag.values()],[x for x in q_tag.values()])
print(kl)