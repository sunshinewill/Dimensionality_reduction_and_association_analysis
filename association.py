import pandas as pd
import numpy as np
import itertools
from itertools import combinations

df = pd.read_csv(
        filepath_or_buffer="associationruletestdata.txt",
        header=None,
        sep='\t')



geneNum = df.shape[1]-1
patientNum = df.shape[0]
# AML: 101 ALL: 102  Breast Cancer：103  Colon Cancer:104
def lengthOneGenerator(min_support, dict):
    Length1 = []
    for i in range(df.shape[1] - 1):
        cnt = 0
        for j in range(patientNum):
            if df[i][j] == 'Up':
                cnt += 1
        upSupport = cnt * 1.0 / patientNum
        downSupport = 1 - upSupport

        if upSupport >= min_support:
            Length1.append(((i + 1),))
            dict[tuple([i+1])] = cnt

        if downSupport >= min_support:
            Length1.append(((-1 * (i + 1),)))
            dict[tuple([-1*(i+1)])] = patientNum-cnt

    amlCnt = 0
    allCnt = 0
    breastCnt = 0
    colonCnt = 0

    for j in range (patientNum):
        if(df.iloc[j,df.shape[1]-1]=="AML"):
            amlCnt+=1
        if (df.iloc[j, df.shape[1] - 1] == "ALL"):
            allCnt += 1
        if (df.iloc[j, df.shape[1] - 1] == "Breast Cancer"):
            breastCnt += 1
        if (df.iloc[j, df.shape[1] - 1] == "Colon Cancer"):
            colonCnt += 1
    if amlCnt*1.0/patientNum >= min_support:
        Length1.append(((101),))
        dict[tuple([101])] = amlCnt*1.0/patientNum
    if allCnt*1.0/patientNum >= min_support:
        Length1.append(((102),))
        dict[tuple([102])] = allCnt*1.0/patientNum
    if breastCnt*1.0/patientNum >= min_support:
        Length1.append(((103),))
        dict[tuple([103])] = breastCnt*1.0/patientNum
    if colonCnt*1.0/patientNum >= min_support:
        Length1.append(((104),))
        dict[tuple([104])] = colonCnt*1.0/patientNum

    return Length1

def checkSupport(newItemSet, min_support,dict):
    cnt = 0
    needed = min_support*patientNum
    for i in range(0,patientNum):
        allMatch = True
        j=0
        while(j<len(newItemSet)):
            val = newItemSet[j]
            index = abs(val)-1
            if val > 100:
                index = 100

            if df.iloc[i,index] == 'Up':
                if val<0:
                    allMatch = False
                    break
            elif df.iloc[i,index] == 'Down':
                if val>0:
                    allMatch = False
                    break
            elif df.iloc[i,index] == 'AML':
                if val != 101:
                    allMatch = False
                    break
            elif df.iloc[i,index] == 'ALL':
                if val != 102:
                    allMatch = False
                    break
            elif df.iloc[i,index] == 'Breast Cancer':
                if val != 103:
                    allMatch = False
                    break
            elif df.iloc[i,index] == 'Colon Cancer':
                if val != 104:
                    allMatch = False
                    break
            j+=1
        if allMatch:
            cnt+=1
    if cnt>=needed:
        dict[newItemSet] = cnt
        return True
    else:
        return False


def join_itemset(itemsets):
    res = []
    i = 0
    while i < len(itemsets):
        skip = 1

        *itemset_head, itemset_tail = itemsets[i]

        tail_items = [itemset_tail]

        for j in range(i + 1, len(itemsets)):
            *itemset_n_head, itemset_n_tail = itemsets[j]
            if itemset_head == itemset_n_head:

                tail_items.append(itemset_n_tail)
                skip += 1

            else:
                break
        itemset_head_tuple = tuple(itemset_head)

        for a, b in itertools.combinations(tail_items, 2):
            res.append(itemset_head_tuple + (a,) + (b,))

        i += skip
    return res

def generateCandidate(itemsets, possible_itemsets):
    candidates = []

    for possible_itemset in possible_itemsets:
        allfound = True
        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]
            if removed not in itemsets:
                allfound = False
                break

        if allfound:
            candidates.append(possible_itemset)
    return candidates

def apriori(support):
    min_support = support
    itemset_dict = {}

    Length1 = lengthOneGenerator(min_support,  itemset_dict)

    frequentItemSet = []

    frequentItemSet.append(Length1)

    k = 2
    while (k < df.shape[1]):
        LengthLast = frequentItemSet[k - 2]
        Lengthk = []

        possible_extensions = join_itemset(LengthLast)
        candidates = generateCandidate(LengthLast, possible_extensions)
        for candidate in candidates:
            if (checkSupport(candidate, min_support, itemset_dict)):
                Lengthk.append(candidate)
        frequentItemSet.append(Lengthk)
        k += 1

    print("Support is set to be " + str(min_support * 100) + "%")
    for l in range(geneNum):
        print("number of length-" + str(l + 1) + " frequent itemsets: " + str(len(frequentItemSet[l])))
        if (len(frequentItemSet[l]) == 0):
            break
    return frequentItemSet,itemset_dict





def rule_generation(frequentItemSet,itemset_dict, min_confidence):
    rule_list = []
    for i in range(1, len(frequentItemSet)):
        allitemSetThisLength = frequentItemSet[i]
        rule_dictThisLength = {}
        for itemSet in allitemSetThisLength:
            for bodyLength in range(1, len(itemSet)):
                for head in combinations(itemSet, len(itemSet) - bodyLength):
                    body = tuple([i for i in itemSet if i not in head])

                    confidence = itemset_dict[itemSet] * 1.0 / itemset_dict[head]
                    if confidence >= min_confidence:
                        if head in rule_dictThisLength:
                            rule_dictThisLength[head].append(body)
                        else:
                            rule_dictThisLength[head] = [body]


        rule_list.append(rule_dictThisLength)
    return rule_list

def prestep():
    frequentItemSet, itemset_dict = apriori(0.5)
    rule_list = rule_generation(frequentItemSet, itemset_dict, 0.7)
    return rule_list




def template1(target, num, list,rule_list):
    res = []
    transform = []
    for item in list:
        if item[0] == 'G':
            transformItem = int(item.split("_")[0][1:])
            if item[len(item)-1] == 'n':
                transformItem = -1*transformItem
            transform.append(transformItem)
        # AML: 101 ALL: 102  Breast Cancer：103  Colon Cancer:104
        elif item == "AML":
            transform.append(101)
        elif item == "ALL":
            transform.append(102)
        elif item == "Breast Cancer":
            transform.append(103)
        else:
            transform.append(104)
    transform = set(transform)

    if target == "RULE":
        for rule_perLength in rule_list:
            for head in rule_perLength:
                for body in rule_perLength[head]:
                    rule = set(head)|set(body)
                    overlap = len(rule & transform)
                    rule = head + ("->",) + body
                    if num == "ANY":
                        if overlap >=1:
                            res.append(rule)
                    elif num == "NONE":
                        if overlap ==0:
                            res.append(rule)
                    else:
                        if overlap == int(num):
                            res.append(rule)

    elif target == "HEAD":
        for rule_perLength in rule_list:
            for head in rule_perLength:
                overlap = len(set(head) & transform)
                if num == "ANY":
                    if overlap >= 1:
                        for body in rule_perLength[head]:
                            rule = head+("->",)+body
                            res.append(rule)
                elif num == "NONE":
                    if overlap == 0:
                        for body in rule_perLength[head]:
                            rule = head+("->",)+body
                            res.append(rule)
                else:
                    if overlap == int(num):
                        for body in rule_perLength[head]:
                            rule = head+("->",)+body
                            res.append(rule)
    else:
        for rule_perLength in rule_list:
            for head in rule_perLength:
                for body in rule_perLength[head]:
                    overlap = len(set(body) & transform)
                    if num == "ANY":
                        if overlap >=1:
                            rule = head+("->",)+body
                            res.append(rule)
                    elif num == "NONE":
                        if overlap ==0:
                            rule = head+("->",)+body
                            res.append(rule)
                    else:
                        if overlap == int(num):
                            rule = head+("->",)+body
                            res.append(rule)
    return res

def template2(target, size,rule_list):
    res = []
    size = int(size)
    if target == "HEAD":
        for rule_perLength in rule_list:
            for head in rule_perLength:
                if len(head) >= size:
                    for body in rule_perLength[head]:
                        rule = head+("->",)+body
                        res.append(rule)
    elif target == "BODY":
        for rule_perLength in rule_list:
            for head in rule_perLength:
                for body in rule_perLength[head]:
                    if len(body)>=size:
                        rule = head+("->",)+body
                        res.append(rule)
    else:
        for rule_perLength in rule_list:
            for head in rule_perLength:
                for body in rule_perLength[head]:
                    rule = set(head) | set(body)
                    if len(rule)>=size:
                        rule = head + ("->",) + body
                        res.append(rule)
    return res


# def template3(firsttemp, secondtemp, logic, target1, num1, list1, target2, num2, list2):
#     cnt = 0
#     transformlist1 = []
#     transformlist2 = []
#     if firsttemp == 1:
#         for item1 in list1:
#             transformItem1 = item1.split("_")[0][1:]
#             transformlist1.append(transformItem1)
#     if secondtemp == 1:
#         for item2 in list2:
#             transformItem2 = item2.split("_")[0][1:]
#             transformlist2.append(transformItem2)
#
#     transformItem1 = set(transformlist1)
#     transformlist2 = set(transformlist2)
#
#     if logic == 'or':
#         if target1 = "RULE":
#             if target2 = "RULE":
#                 for rule_perLength in rule_list:
#                     for head in rule_perLength:
#                         for body in rule_perLength[head]:
#                             rule = set(head) | set(body)
#                             firstpass = False
#                             if firsttemp == 1:
#                                 overlap1 = len(rule & transformlist1)
#                                 if num1 == "ANY":
#                                     if overlap1 >=1:
#                                         cnt+=1
#                                         firstpass = True
#                                 elif num1 == "NONE":
#                                     if overlap1 == 0:
#                                         cnt +=1
#                                         firstpass = True
#                                 else:
#                                     if overlap1 == int(num1):
#                                         cnt +=1
#                                         firstpass = True
#                             elif len(rule)>=num1:
#                                 cnt+=1
#                                 firstpass = True
#                             if not firstpass:
#                                 if secondtemp == 1:
#                                     overlap2 = len(rule & transformlist2)
#                                     if num2 == "ANY":






def start():
    rule_list = prestep()

    while True:
        template = input('Enter the template you want to test:')
        if template == '1':
            target = input('Enter your target, RULE, HEAD or BODY:')
            num = input('Enter the num you want:')
            liststring = input('Enter the list of items:')
            list = liststring.split(",")
            print(len(template1(target, num, list, rule_list)))


        elif template == '2':
            target = input('Enter your target, RULE, HEAD or BODY:')
            size = input('Enter the minimum size you want:')
            print(len(template2(target, size, rule_list)))

        else:
            firsttemp = template[0]
            secondtemp = template[len(template) - 1]
            logic = template[1:len(template) - 1]

            param1 = []
            if firsttemp == '1':
                target = input('Enter your target, RULE, HEAD or BODY:')
                num = input('Enter the num you want:')
                liststring = input('Enter the list of items:')
                list = liststring.split(",")
                param1.append(target)
                param1.append(num)
                param1.append(list)
            else:
                target = input('Enter your target, RULE, HEAD or BODY:')
                size = input('Enter the minimum size you want:')
                param1.append(target)
                param1.append(size)

            param2 = []
            if secondtemp == '1':
                target = input('Enter your target, RULE, HEAD or BODY:')
                num = input('Enter the num you want:')
                liststring = input('Enter the list of items:')
                list = liststring.split(",")
                param2.append(target)
                param2.append(num)
                param2.append(list)
            else:
                target = input('Enter your target, RULE, HEAD or BODY:')
                size = input('Enter the minimum size you want:')
                param2.append(target)
                param2.append(size)

            res1 = ()
            res2 = ()
            if firsttemp == '1':
                res1 = template1(param1[0], param1[1], param1[2], rule_list)
            else:
                res1 = template2(param1[0], param1[1], rule_list)

            if secondtemp == '1':
                res2 = template1(param2[0], param2[1], param2[2], rule_list)
            else:
                res2 = template2(param2[0], param2[1], rule_list)

            overlap = 0
            for rule in res1:
                if rule in res2:
                    overlap += 1
            if logic == "and":
                print(overlap)
            else:
                print(len(res1) + len(res2) - overlap)


start()