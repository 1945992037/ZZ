import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
from sklearn.tree import DecisionTreeClassifier as DTC
data = pd.read_csv("./train.csv",low_memory=False)
test = pd.read_csv("./test.csv",low_memory=False)
MeaninglessColumns = [
    "OPEN_ORG_NUM",
    "IDF_TYP_CD",
    "GENDER",
    "CUST_EUP_ACCT_FLAG",
    "CUST_AU_ACCT_FLAG",
    "CUST_DOLLER_FLAG",
    "CUST_INTERNATIONAL_GOLD_FLAG",
    "CUST_INTERNATIONAL_COMMON_FLAG",
    "CUST_INTERNATIONAL_SIL_FLAG",
    "CUST_INTERNATIONAL_DIAMOND_FLAG",
    "CUST_GOLD_COMMON_FLAG",
    "CUST_STAD_PLATINUM_FLAG",
    "CUST_LUXURY_PLATINUM_FLAG",
    "CUST_PLATINUM_FINANCIAL_FLAG",
    "CUST_DIAMOND_FLAG",
    "CUST_INFINIT_FLAG",
    "CUST_BUSINESS_FLAG",
]
def del_features(data):
    columns = []
    for name in data.columns:
        unique = data[name].unique().shape[0]
        full = data[name].shape[0]
        if (unique == 1) | (unique == full):
            columns.append(name)
    return columns
for i in del_features(data):
    MeaninglessColumns.append(i)

data.drop(MeaninglessColumns, axis=1, inplace=True)
MeaninglessColumns.remove("CUST_ID")
test.drop(MeaninglessColumns, axis=1, inplace=True)
pd.set_option("display.max_rows", None)
data = data.drop_duplicates(keep="first")
data_num = data.select_dtypes(include=[np.number])
data_non_num = data.select_dtypes(exclude=[np.number])
data_non_num_num = pd.get_dummies(data_non_num)
data = pd.concat([data_num, data_non_num_num], axis=1)
test_num = test.select_dtypes(include=[np.number])
test_non_num = test.select_dtypes(exclude=[np.number])
test_non_num_num = pd.get_dummies(test_non_num)
test = pd.concat([test_num, test_non_num_num], axis=1)
data_target = data["bad_good"]
data.drop(["bad_good"], axis=1, inplace=True)
x_train_local, x_test_local, y_train_local, y_test_local = train_test_split(
    data, data_target, test_size=0.3
)

times = []
scores = []
models = [DTC()]
names = ["决策树"]
for i in range(1):
    time1 = time.time()
    # 10次交叉验证
    score = cross_val_score(models[i], x_train_local, y_train_local, cv=10).mean()
    time2 = time.time()
    scores.append(float(score))
    endTime = time2 - time1
    times.append(endTime)

plt.rcParams["font.sans-serif"] = ["SimHei"]
x = np.arange(4)
width = 0.25
ax = plt.subplot(1, 1, 1)
ax.bar(x, times, width, color="r", label="耗时")
ax.bar(x + width, scores, width, color="b", label="分数")
ax.set_xticks(x + width)
ax.set_xticklabels(names)
ax.legend()
plt.show()
# 选择模型并输出结果
x_test = test.drop(["CUST_ID"], axis=1)
print("选择的模型是:" + names[np.argmax(scores)])
pred = models[np.argmax(scores)].fit(x_train_local, y_train_local).predict(x_test)
pred = pd.DataFrame(pred)
pred["bad_good"] = pred
pred.drop(0, axis=1, inplace=True)
sub = pd.concat([test["CUST_ID"], pred], axis=1)
sub.to_csv("./submission.csv", index=0)
