#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from XPLAIN_utils.LACE_utils.LACE_utils1 import *
from XPLAIN_utils.LACE_utils.LACE_utils2 import *
from XPLAIN_utils.LACE_utils.LACE_utils3 import *
from XPLAIN_utils.LACE_utils.LACE_utils4 import *
from XPLAIN_utils.LACE_utils.LACE_utils5 import *
from XPLAIN_class import *
import ipywidgets as widgets
from IPython.display import display
from IPython.display import display, clear_output
import ipywidgets as widgets

def getIndexRule(rule, instance):
    rule_Index=[]
    for r in rule:
        indexA=getIndexAttribute(r, instance)
        if indexA==-1:
            return False
        rule_Index.append(indexA)
    return rule_Index

def getIndexAttribute(attribute, instanceI):
    index=1
    for i in instanceI.domain.attributes:
        if i.name == attribute:
            return index
        index=index+1
    return -1

class XPLAIN_explanation:

  def __init__(self, XPLAIN_explainer_o, targetClass, instance, diff_single, impo_rules, n_inst, KNN, error, map_difference, impo_rules_complete): 

    self.XPLAIN_explainer_o=XPLAIN_explainer_o
    self.n_inst=n_inst
    self.diff_single=diff_single
    self.impo_rules=impo_rules
    self.impo_rulesUser=deepcopy(impo_rules)
    self.map_difference=deepcopy(map_difference)
    self.map_differenceUser=deepcopy(map_difference)
    self.KNN=KNN
    self.error=error
    self.impo_rules_complete=deepcopy(impo_rules_complete)
    self.impo_rules_completeUser=deepcopy(impo_rules_complete)
    self.instance=instance
    self.errorUser=error
    c1=self.XPLAIN_explainer_o.classifier(self.instance, True)[0]   
    self.targetClass=targetClass
    self.indexI=self.XPLAIN_explainer_o.getIndexI(self.targetClass)
    self.pred=c1[self.indexI]
    self.pred_str=str(round(c1[self.indexI], 2))



  def insertInteractiveRule(self):
    from ipywidgets import  HBox
    feature_names=[]
    for f in self.instance.domain.attributes:
        feature_names.append(f.name)
    w=widgets.SelectMultiple(
    options=feature_names,
    description='Feature',
    disabled=False
    )

    display(w)

    def clearAndShow(btNewObj):
        clear_output()
        display(w)
        display(h)


    def getRuleInteractiveButton(btn_object):
        rule=list(w.value)
        #clear_output()
        rule_Index=self.getNewExplanation(rule)


    btnExpl = widgets.Button(description='Compute')
    btnExpl.on_click(getRuleInteractiveButton)
    btnNewSel = widgets.Button(description='Clear')
    btnNewSel.on_click(clearAndShow)
    h=HBox([btnExpl, btnNewSel])
    display(h)

  def getNewExplanation(self, rule, update=False):
    rule_Index=getIndexRule(rule, self.instance)
    rule_print="{"
    for r in rule:
        rule_print=rule_print+r+"="+self.instance[r].value+", "
    rule_print=rule_print[:-2]+"}"

    print("Rule inserted:", rule_print)

    if update:
      if rule_Index in self.impo_rules_completeUser:
          print("Rule already present")
          #Solo imporules+user
          plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,  self.targetClass,  self.XPLAIN_explainer_o.classif, self.map_differenceUser, self.pred_str, self.impo_rules)

      else:
          user_rules=self.getUserRules(rule_Index, update)
          
          map_difference_new={}
          map_difference_new=computePredictionDifferenceSubsetRandomOnlyExisting(self.XPLAIN_explainer_o.training_dataset, self.instance, user_rules, self.targetClass, self.XPLAIN_explainer_o.classifier, self.indexI, map_difference_new)

          self.updateUserRules(user_rules, map_difference_new, rule_Index)

          error, PI_rel2=computeApproxErrorRule(self.XPLAIN_explainer_o.mappa_class,self.pred,self.diff_single, user_rules, self.targetClass, self.map_differenceUser)
          #print("error", error)
          plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,  self.targetClass,  self.XPLAIN_explainer_o.classif, self.map_differenceUser, self.pred_str, self.impo_rules,  user_rules)

          return rule_Index

    else:     
      if rule_Index in self.impo_rules_complete:
          print("Rule already present")

          plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,  self.targetClass,  self.XPLAIN_explainer_o.classif, self.map_difference, self.pred_str, self.impo_rules)

      else:
          user_rules=self.getUserRules(rule_Index, update)
          
          map_difference_new={}
          map_difference_new=computePredictionDifferenceSubsetRandomOnlyExisting(self.XPLAIN_explainer_o.training_dataset, self.instance, user_rules, self.targetClass, self.XPLAIN_explainer_o.classifier, self.indexI, map_difference_new)

          tmp=deepcopy(self.map_difference)
          map_difference_new.update(tmp)

          
          error, PI_rel2=computeApproxErrorRule(self.XPLAIN_explainer_o.mappa_class,self.pred,self.diff_single, user_rules, self.targetClass, map_difference_new)
          #print("error",error)
          #print(map_difference_new)
          plotTheInfo_v4(self.instance, self.diff_single, self.n_inst, self.XPLAIN_explainer_o.dataname, self.KNN,  self.targetClass,  self.XPLAIN_explainer_o.classif, map_difference_new, self.pred_str, self.impo_rules, user_rules)

          return rule_Index


  def getUserRules(self, rule_Index, update=False):
    if update:
      max_rule=max(self.impo_rules_completeUser, key=len)
    else:
      max_rule=max(self.impo_rules_complete, key=len)
    rule_union=max_rule.copy()
    for i in rule_Index:
      rule_union.append(i)
    rule_union.sort()
    rule_union=list(set(rule_union))   

    user_rules=[]
    user_rules.append(rule_Index)
    user_rules.append(rule_union)
    return user_rules

  def updateUserRules(self, user_rules, map_difference_new, rule_Index):
    for r in user_rules:
        self.impo_rules_completeUser.append(r)
    self.map_differenceUser.update(map_difference_new)
    r=', '.join(map(str, rule_Index))    
    self.impo_rulesUser.append(r)



  def interactiveAttributeLevelInspection(self):
    from copy import deepcopy
    from ipywidgets import  HBox

    feature_names=[]
    for f in self.instance.domain.attributes:
        feature_names.append(f.name)
    w=widgets.SelectMultiple(
    options=feature_names,
    description='Feature',
    disabled=False
    )

    display(w)
    sa={}

    def clearAndShow(btNewObj):
        clear_output()
        display(w)
        display(btn)

    def getSlidersValues(btn_obj_expl):
        inst1=deepcopy(self.instance)
        n=""
        for s in sa.values():
            inst1[s.description]=s.value
            n=n+"_"+s.description+"_"+s.value
        self.XPLAIN_explainer_o.getExplanationPerturbed_i(str(inst1.id)+n, inst1)


    def compareOriginal(btnCompare):
        inst1=deepcopy(self.instance)
        attr=[i.name for i in inst1.domain.attributes]
        n=""
        attr_name=[]
        for s in sa.values():
            if inst1[s.description]!=s.value:
              attr_name.append(s.description)
            inst1[s.description]=s.value
            n=n+"_"+s.description+"_"+s.value
        self.XPLAIN_explainer_o.comparePerturbed( str(self.n_inst)  ,str(inst1.id)+n, inst1, [attr.index(i) for i in attr_name])


    def getSlidersAttributes(btn_object):
        sa.clear()
        for a in w.value:
            for f in self.instance.domain.attributes:
                if f.name==a:
                    wa=widgets.SelectionSlider(
                        options=f.values,
                        description=f.name,
                        disabled=False,
                        value=self.instance[f.name],
                    continuous_update=False
                    )
                    sa[f.name]=wa
        for s in sa.values():
            display(s)
        btnExpl = widgets.Button(description='Get explanation')
        btnExpl.on_click(getSlidersValues)
        btnNewSel = widgets.Button(description='New selection')
        btnNewSel.on_click(clearAndShow)
        btnCompare = widgets.Button(description='Compare original')
        btnCompare.on_click(compareOriginal)
        h=HBox([btnExpl, btnCompare, btnNewSel])
        display(h)

    btn = widgets.Button(description='Select attributes')
    btn.on_click(getSlidersAttributes)
    display(btn)