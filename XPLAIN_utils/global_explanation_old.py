#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

import numpy as np
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter('ignore')
from XPLAIN_class import *
from Explanation_w import *
import seaborn as sns
import copy 

sns.set_palette('muted')
sns.set_context("notebook", #font_scale=1.5,
                rc={"lines.linewidth": 2.5})


class Global_Explanation:
    def __init__(self, xplain_obj):
        self.XPLAIN_obj=xplain_obj
        self.attr_list=getAttrList(self.XPLAIN_obj.explain_dataset[0])
        print(self.attr_list)
        self.map_avM={}
        self.map_av={}
        self.map_rules={}
        self.map_rules_abs={}
        self.map_a = {}
        self.map_aM = {}

    def getGlobalExplanation(self):
        attr_list=self.attr_list
        map_avS={}
        map_avSM={}
        map_aS = {k:0 for k in attr_list}
        map_aSM = {k:0 for k in attr_list}
        cnt=0
        for instance in self.XPLAIN_obj.explain_dataset:
            c=self.XPLAIN_obj.classifier(instance, False)
            target_class=self.XPLAIN_obj.map_names_class[c[0]]
            e=self.XPLAIN_obj.explain_instance(instance, target_class)
            diff_s=copy.deepcopy(e.diff_single)
            map_aS = {attr_list[i]:map_aS[attr_list[i]]+diff_s[i] for i in range(0, len(attr_list))}
            map_aSM = {attr_list[i]:map_aSM[attr_list[i]]+abs(diff_s[i]) for i in range(0, len(attr_list))}
            attr_value_list=getAttrValueList(e.instance)
            self.map_rules, self.map_rules_abs=self.updateRuleMapping(self.map_rules, self.map_rules_abs, e.map_difference, e.diff_single, e.instance)
            for i in range(0, len(attr_value_list)):
                k=attr_value_list[i]
                map_avS=self.initD_Cnt(k, map_avS)
                map_avSM=self.initD_Cnt(k, map_avSM)
                map_avS[k]={"d":map_avS[k]["d"]+e.diff_single[i], "cnt":map_avS[k]["cnt"]+1}
                map_avSM[k]={"d":map_avSM[k]["d"]+abs(e.diff_single[i]), "cnt":map_avSM[k]["cnt"]+1}
            cnt=cnt+1
            #print(".",end="")
            print(cnt)
            if cnt==100:
                break

        self.map_a=sortByValue({k:map_aS[k]/len(self.XPLAIN_obj.explain_dataset) for k in map_aS.keys()})
        self.map_aM=sortByValue({k:map_aSM[k]/len(self.XPLAIN_obj.explain_dataset) for k in map_aSM.keys()})
        self.map_av=self.computeAvgD_Cnt(map_avS)
        self.map_avM=self.computeAvgD_Cnt(map_avSM)
        self.map_rules=self.computeAvgD_Cnt(self.map_rules)
        self.map_rules_abs=self.computeAvgD_Cnt(self.map_rules_abs)
    
        return self

    def computeAvgD_Cnt(self, mapS):    
        map_avg=sortByValue({k: mapS[k]["d"]/mapS[k]["cnt"] for k in mapS.keys()})
        return map_avg

        #input: imporules complete, impo_rules, separatore
    def getRM(self, r, instT, sep=", "):
        if type(r)==str:
            rule1=""
            for k in r.split(sep):
                rule1=rule1+instT[int(k)-1].variable.name+"="+instT[int(k)-1].value+", "
            rule1=rule1[:-2]
        return rule1


    def initD_Cnt(self, r, map_contr):
        if r not in map_contr:
            map_contr[r]={"d":0, "cnt":0}
        return map_contr

    def updateRuleMapping(self, map_rules, map_rules_abs, e_map_difference, single_diff, e_instance):
        for k in e_map_difference:
            r=self.getRM(k, e_instance, sep=",")
            map_rules=self.initD_Cnt(r, map_rules)
            map_rules_abs=self.initD_Cnt(r, map_rules_abs)

            map_rules[r]={"d":map_rules[r]["d"]+e_map_difference[k], "cnt":map_rules[r]["cnt"]+1}
            map_rules_abs[r]={"d":map_rules_abs[r]["d"]+abs(e_map_difference[k]), "cnt":map_rules_abs[r]["cnt"]+1}
        return map_rules, map_rules_abs



    def getGlobalAttributeContribution(self):
        return self.map_a

    def getGlobalAttributeContributionAbs(self):
        return self.map_aM

    def getGlobalAttributeValueContribution(self, k=False):
        if k:
            return getKMap(self.map_av, k)
        return self.map_av

    def getGlobalAttributeValueContributionAbs(self, k=False):
        if k:
            return getKMap(self.map_avM, k)
        return self.map_avM


    def getGlobalRuleContribution(self):
        return self.map_rules

    def getGlobalRuleContributionAbs(self):
        return self.map_rules_abs

    def getRuleMapping(self, map_i, k=False):
        map_i=getKMap(map_i, k) if k else map_i
        mapping_rule={"Rule_"+str(i+1): key for (key, i) in zip(map_i.keys(), range(len(map_i)))}
        return mapping_rule
    """
    def plotGlobalAttributeContribution(self, absV=True, sortedV=True, classname="predicted"):
        map_plot=self.map_aM if absV else self.map_a
        self.plotGlobalInfo(self.attr_list, map_plot, classname=classname, sortedV=sortedV)

    def plotGlobalAttributeValueContribution(self, absV=True, sortedV=True, classname="predicted", k=False):
        map_plot=self.map_avM if absV else self.map_av
        if k:
            map_plot=getKMap(map_plot, k)
        self.plotGlobalInfo(sorted(list(map_plot.keys())), map_plot, classname=classname, sortedV=sortedV)

    def plotGlobalRuleContribution(self, absV=True, sortedV=True, classname="predicted", k=False):
        map_plot=self.map_rules_abs if absV else self.map_rules
        if k:
            map_plot=getKMap(map_plot, k)
        map_plot, mapping_rule=self.getConvertRuleMapping(map_plot)
        self.plotGlobalInfo(sorted(list(map_plot.keys())), map_plot, classname=classname, sortedV=sortedV)
        self.printRuleMapping(mapping_rule)

    def plotGlobalInfo(self, attr_list, map_info, classname, sortedV=True):
        plotTheInfo_v5(attr_list, map_info, self.XPLAIN_obj.dataname, classname, self.XPLAIN_obj.classif, sortedV=sortedV)

    def plotGlobalExplanations(self, absV=True, sortedV=True, classname="predicted", k=False):
        self.plotGlobalAttributeContribution(absV=absV, sortedV=sortedV, classname=classname)
        self.plotGlobalAttributeValueContribution(absV=absV, sortedV=sortedV, classname=classname, k=k)
        self.plotGlobalRuleContribution(absV=absV, sortedV=sortedV, classname=classname, k=k)

    def printRuleMapping(self, mapping_rule):
        print("Mapping:")
        for k in mapping_rule:
            print(k, ":", mapping_rule[k])


    def getGlobalRuleMapping(self, k=False):
        return self.getRuleMapping(self.map_rules, k=k)

    def getGlobalRuleMappingAbs(self, k=False):
        return self.getRuleMapping(self.map_rules_abs, k=k)

    def printGlobalRuleMapping(self, k=False):
        return self.getRuleMapping(self.map_rules, k=k)

    def printGlobalRuleMappingAbs(self, k=False):
        self.printRuleMapping(self.getGlobalRuleMappingAbs(k=k))

    def printGlobalRuleMapping(self, k=False):
        self.printRuleMapping(self.getGlobalRuleMapping(k=k))


    def getConvertRuleMapping(self, map_i):
        map_converted={}
        mapping_rule={}
        cnt=1
        for k in map_i.keys():
            map_converted["Rule_"+str(cnt)]=map_i[k]
            mapping_rule["Rule_"+str(cnt)]=k
            cnt=cnt+1
        return map_converted, mapping_rule
    """
def getAttrList(e_instance):
    attributes_list=[]
    for i in e_instance.domain.attributes:
        attributes_list.append(str(i.name))
    return attributes_list


def getAttrValueList(e_instance):
    attributes_list=[]
    for i in e_instance.domain.attributes:
        attributes_list.append(str(i.name+"="+str(e_instance[i])))
    return attributes_list


def sortByValue(map_i):
    return {k: v for k, v in sorted(map_i.items(), key=lambda item: item[1], reverse=True)}


def getKMap(map_i, k):
    return dict(list(map_i.items())[:k])