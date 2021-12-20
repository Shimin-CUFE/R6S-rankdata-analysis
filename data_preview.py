
import numpy as np
import pandas as pd

rankdata_000 = pd.read_csv(
    r'D:\Study\Python\Workspace\rainbowSixSiege_analysis\datadump_s5_ranked_data\datadump_s5-000.csv')
col = rankdata_000.columns

data = rankdata_000.loc[:, ['matchid', 'roundnumber', 'gamemode', 'mapname', 'roundduration', 'skillrank', 'role', 'haswon']]


opdata = pd.get_dummies(rankdata_000['operator'], drop_first=False, prefix='OP')
newdata = pd.merge(data, opdata, left_index=True, right_index=True)
new_matchid = newdata['matchid'].map(str) + "_" + newdata['roundnumber'].map(str) + "_" + newdata['role'].map(str)
newdata = newdata.drop(labels=['matchid', 'roundnumber', 'role'], axis=1)
newdata.insert(0, 'newmatchid', new_matchid, allow_duplicates=False)

newdata['roundduration'].describe()


newdata['gamemode'] = newdata['gamemode'].replace({'HOSTAGE': 1, 'BOMB': 2, 'SECURE_AREA': 3})
newdata['mapname'] = newdata['mapname'].replace(
    {'CLUB_HOUSE': 1, 'PLANE': 2, 'KANAL': 3, 'HEREFORD_BASE': 4, 'CONSULATE': 5,
     'YACHT': 6, 'OREGON': 7, 'BORDER': 8, 'SKYSCRAPER': 9, 'BANK': 10, 'COASTLINE': 11,
       'BARTLETT_U.': 12, 'HOUSE': 13, 'KAFE_DOSTOYEVSKY': 14, 'FAVELAS': 15, 'CHALET': 16})
newdata['skillrank'] = newdata['skillrank'].replace(
    {'Gold': 4, 'Unranked': 0, 'Platinum': 5, 'Silver': 3, 'Bronze': 1, 'Copper': 2})
res = newdata.groupby(newdata['newmatchid']).agg(max).reset_index()
print(res)

SCALE = 25000
label = np.array(res.loc[:SCALE, :]['haswon'])
print(label.shape)
train_data = np.array(res.drop(labels=['newmatchid', 'haswon', 'gamemode', 'mapname', 'roundduration', 'skillrank'], axis=1).loc[:SCALE, :])
print(train_data.shape)
