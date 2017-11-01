#!/usr/bin/env python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xgboost as xgb
from sklearn.linear_model import RandomizedLasso
from xgboost.sklearn import XGBClassifier

train = pd.read_csv("sbtrain.csv")
test = pd.read_csv("sbtest.csv")

x = []
for col in train.columns:
    if train[col].std() == 0:
        x.append(col)

train.drop(x, axis=1, inplace=True)
test.drop(x, axis=1, inplace=True)

x = []
cols = train.columns
for i in range(len(cols)-1):
    vals = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(vals, train[cols[j]].values):
            x.append(cols[j])

train.drop(x, axis=1, inplace=True)
test.drop(x, axis=1, inplace=True)

bool_train_names = [col for col in train if train[[col]].isin([0, 1]).all().values]

cols = train.columns
for i in range(len(cols)-1):
    found = False
    for bool_col in bool_train_names:
        if (~ (cols[i] in bool_col)):
            found = True
            continue
    if ~found:
        if (min(train[cols[i]]) < 1):
            train[cols[i]] = np.log(train[cols[i]] + np.abs(np.min(train[cols[i]])) + 1)
            test[cols[i]] = np.log(test[cols[i]]  + np.abs(np.min(train[cols[i]])) + 1)
        else:
            train[cols[i]] = np.log(train[cols[i]] - np.abs(np.min(train[cols[i]])) + 1)
            test[cols[i]] = np.log(test[cols[i]] - np.abs(np.min(train[cols[i]])) + 1)
        train[cols[i]] = (train[cols[i]] - np.mean(train[cols[i]])) / (np.max(train[cols[i]]) - np.min(train[cols[i]]))


y_train = train['TARGET'].values
X_train = train.drop(['ID','TARGET'], axis=1).values

y_test = test['ID']
X_test = test.drop(['ID'], axis=1).values

rf = RandomForestRegressor()
X_train = rf.fit_transform(X_train, y_train)
print "Features sorted by their score:"
print sorted(map(lambda x: round(x, 4), rf.feature_importances_))



#rlasso = RandomizedLasso(alpha=0.025)
#X_train = rlasso.fit_transform(X_train, y_train)

#print "Features sorted by their score:"
#print sorted(map(lambda x: round(x, 4), rlasso.scores_), reverse=True)

xgb1 = XGBClassifier(
 learning_rate =0.10,
 n_estimators=600,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.6808,
 colsample_bytree=0.697,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgtrain = xgb.DMatrix(X_train, label=y_train)
cvresult = xgb.cv(xgb1.get_xgb_params(), xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=10,\
metrics=['auc'], early_stopping_rounds=100)
xgb1.set_params(n_estimators=cvresult.shape[0])

print cvresult

xgb1.fit(X_train, y_train, eval_metric='auc')
output = xgb1.predict_proba(X_test)[:,1]

# Using Pearson correlation to evaluate threashholds
# MIC verification
output[(test['num_var30'] > 9 ).values] = 0

net_saldo = test['num_var33']+test['saldo_medio_var33_ult3']+test['saldo_medio_var44_hace2']+test['saldo_medio_var44_hace3']+\
test['saldo_medio_var33_ult1']+test['saldo_medio_var44_ult1']

output[(net_saldo > 0).values] = 0
output[(test['num_var37_0'] > 45 ).values] = 0

#output[(test['num_var10_0' ] < 20 ).values] = 0
output[(test['num_var13_0' ] > 6 ).values] = 0
output[(test['num_var33_0' ] > 0 ).values] = 0

output[(test['imp_op_var40_comer_ult1'] > 3637.54 ).values] = 0
output[(test['var21'] > 7488 ).values] = 0
output[(test['var38'] > 3988755 ).values] = 0
output[(test['imp_op_var39_comer_ult3'] > 13184 ).values] = 0
output[(test['saldo_var8'] > 60082 ).values] = 0

output[(test['saldo_var26'] > 10380.11 ).values] = 0
output[(test['var15'] < 23 ).values] = 0
output[(test['saldo_var33'] > 0 ).values] = 0

#output[(test['saldo_var04_1' ] > 66789).values] = 0
#output[(test['saldo_medio_var12_hace1' ] > 8900).values] = 0

output[(test['num_var13_largo_0'] > 3 ).values] = 0

output[(test['saldo_var14'] > 19039.11 ).values] = 0
output[(test['saldo_medio_var5_hace2'] > 160000 ).values] = 0
output[(test['imp_ent_var16_ult1' ] > 5989 ).values] = 0
output[(test['saldo_var5'] > 137602 ).values] = 0

output[((test['num_var45_ult3'] + test['var36'] \
  + test['num_var45_hace3'] + test['var15']) <= 24).values] = 0
output[(test['saldo_medio_var5_ult3'] > 108233 ).values] = 0
output[(test['saldo_var17'] > 288188.97 ).values] = 0



submission = pd.DataFrame({"ID":y_test, "TARGET":output})
submission.to_csv("submission.csv", index=False)
