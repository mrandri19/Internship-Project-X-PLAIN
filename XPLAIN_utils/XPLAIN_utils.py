import pickle

from os import path

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


def savePickle(model, dirO, name):
    createDir(dirO)
    with open(dirO + "/" + name + '.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def openPickle(dirO, name):
    if path.exists(dirO + "/" + name + '.pickle'):
        with open(dirO + "/" + name + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        return False
