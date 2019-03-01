import math
import random
from sys import argv

def score(A, B): # For vertical images
    AA = set(A)
    BB = set(B)
    return min(len(AA.difference(BB)), len(BB.difference(AA)), len(AA.intersection(BB)))

def s_step(S):
    num = len(S)
    superlist = []
    for x in S:
        superlist.append([x])
    while num>1:
        #print(num)
        newSuper = []
        for i in range(0,len(superlist)-1,2):
            list1 = superlist[i]
            myfScore = 0
            mybScore = 0
            myfList2 = []
            mybList2 = []
            mybJ = 0
            myfJ = 0
            for j in range(i+1,len(superlist)):
                #if(j>len(superlist)-1):
                #    break
                #print(j)
                list2 = superlist[j]
                S1_0 = list1[0]
                S1_n = list1[-1]
                S2_0 = list2[0]
                S2_n = list2[-1]
                print(type(S1_n))
                print(type(S1_0))
                print(type(S2_0))
                print(type(S2_n))
                fScore = score(S1_n,S2_0)
                bScore = score(S1_0,S2_n)
                if fScore>myfScore:
                    myfScore = fScore
                    myfList2 = list2
                    myfJ = j
                if bScore > mybScore:
                    mybScore = bScore
                    mybList2 = list2
                    mybJ = j
            if bScore>fScore:
                newList = mybList2 + list1
                superlist[i+1],superlist[mybJ] = superlist[mybJ],superlist[i+1]
            else:
                newList = list1 + myfList2
                superlist[i+1],superlist[myfJ] = superlist[myfJ],superlist[i+1]
            newSuper.append(newList)
            #print(len(newSuper))
        superlist = copy.deepcopy(newSuper)
        num = len(superlist)
    return superlist

# Std dev cost
def get_cost(avg, indices, V, length):
    summation = 0.0
    for i in range(length - 1):
        summation += math.pow(avg - float(score(V[i], V[i + 1])), 2)
    return summation / length

def get_average(V):
    return float(sum([(len(i) - 1) for i in V])) / len(V)

def v_step(V, iterations):
    length = len(V)
    global_average = get_average(V)
    indices = random.sample(range(length), length)
    opt_indices = []
    min_cost = get_cost(global_average, indices, V, length)

    for i in range(iterations):
        value = get_cost(global_average, indices, V, length)
        if value < min_cost:
            min_cost = value
            opt_indices = indices.copy()

    new_v = [[V[i], V[i + 1]] for i in range(int(length / 2))]

    return new_v

# Solve the problem ... basically
def get_traversal(V, H):
    if len(V) != 0:
        new_v = v_step(V, 1000)
        print(new_v)
        H += new_v

    super_list = s_step(H)
    print(super_list)


f = open(argv[1], 'r')
N = int(f.readline())
V_set = []
H_set = []

count = 0

for i in range(N):
    image_details = f.readline().strip()
    if image_details[0] == 'V':
        V_set.append([str(count)] + image_details[3:].split())
        count += 1
    else:
        H_set.append([str(count)] + image_details[3:].split())
        count += 1

get_traversal(V_set, H_set)
