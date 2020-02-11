import pickle

from XPLAIN_class import *
from XPLAIN_utils.LACE_utils.LACE_utils1 import createDir


def convertOTable2Pandas(orangeTable, ids=None, sel="all", cl=None, mapName=None):
    import pandas as pd

    if sel == "all":
        dataK = [orangeTable[k].list for k in range(0, len(orangeTable))]
    else:
        dataK = [orangeTable[k].list for k in sel]

    columnsA = [i.name for i in orangeTable.domain.variables]

    if orangeTable.domain.metas != ():
        for i in range(0, len(orangeTable.domain.metas)):
            columnsA.append(orangeTable.domain.metas[i].name)
    data = pd.DataFrame(data=dataK, columns=columnsA)

    if cl != None and sel != "all" and mapName != None:
        y_pred = [mapName[cl(orangeTable[k], False)[0]] for k in sel]
        data["pred"] = y_pred

    if ids != None:
        data["instance_id"] = ids
        data = data.set_index('instance_id')

    return data

# input: imporules complete, impo_rules, separatore
def getExtractedRulesMapping(instT, impo_rules, impo_rules_c, sep=", "):
    rulesPrint = {}
    unionRulePrint = {}
    for r in impo_rules:
        # impo_rule
        if type(r) == str:
            rule = "{"
            for k in r.split(sep):
                rule = rule + instT[int(k) - 1].variable.name + "=" + instT[int(k) - 1].value + ", "
            rule = rule[:-2] + "}"
            rulesPrint[r.replace(sep, ",")] = rule
    if impo_rules_c != []:
        union = max(impo_rules_c, key=len)
        if union.replace(",", ", ") not in impo_rules:
            rule = "{"
            for k in union.split(","):
                rule = rule + instT[int(k) - 1].variable.name + "=" + instT[int(k) - 1].value + ", "
            rule = rule[:-2] + "}"
            unionRulePrint[union] = rule

    return rulesPrint, unionRulePrint


# input: imporules complete, impo_rules, separatore
def getExtractedRulesMapping_old(instT, impo_rules, impo_rules_c, sep=", "):
    rulesPrint = {}
    unionRulePrint = {}
    for r in impo_rules:
        # impo_rule
        if type(r) == str:
            rule = "{"
            for k in r.split(sep):
                rule = rule + instT[int(k) - 1].variable.name + "=" + instT[int(k) - 1].value + ", "
            rule = rule[:-2] + "}"
            rulesPrint[r.replace(sep, ",")] = rule
    if len(impo_rules_c) > 1:
        union = max(impo_rules_c, key=len)
        if union.replace(",", ", ") not in impo_rules:
            rule = "{"
            for k in union.split(","):
                rule = rule + instT[int(k) - 1].variable.name + "=" + instT[int(k) - 1].value + ", "
            rule = rule[:-2] + "}"
            unionRulePrint[union] = rule

    return rulesPrint, unionRulePrint



def printMapping_v5(instT, impo_rules, impo_rules_c, y_label_mapping, sep=", "):
    rulesPrint, unionRulePrint = getExtractedRulesMapping(instT, impo_rules, impo_rules_c, sep)
    if rulesPrint != {}:
        print("\nLocal rules:")
        for r in sorted(rulesPrint, key=len):
            print(y_label_mapping[r], " -> ", rulesPrint[r])
    if unionRulePrint != {}:
        print("Union of rule bodies:")
        for r in unionRulePrint:
            print(y_label_mapping[r], " -> ", unionRulePrint[r])


def savePickle(model, dirO, name):
    import pickle
    createDir(dirO)
    with open(dirO + "/" + name + '.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def openPickle(dirO, name):
    from os import path
    if path.exists(dirO + "/" + name + '.pickle'):
        with open(dirO + "/" + name + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return False
