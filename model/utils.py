import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr
from rdkit import Chem
from sklearn import linear_model
from sklearn.model_selection import KFold

def canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def delect_standard_deviation(x):
    features_delect = []
    for i in x:
        std = np.std(x[i])
        if (std == 0):
            features_delect.append(i)
    return features_delect


def delect_similarity(x, y, threshold=0.95):
    features_delect_R = []

    for i in range(x.shape[1]):
        if x.columns[i] not in features_delect_R:
            a = x.iloc[:, i].values
            for j in range(i):
                if x.columns[j] not in features_delect_R:
                    b = x.iloc[:, j].values
                    R = abs(pearsonr(a, b)[0])
                    # |R|>thresholdなら、yと相関が低い方を消す
                    if ((R > threshold) and (i != j)):
                        cor_a = abs(pearsonr(a, y)[0])
                        cor_b = abs(pearsonr(b, y)[0])
                        if cor_a <= cor_b:
                            features_delect_R.append(x.columns[i])
                        else:
                            features_delect_R.append(x.columns[j])

    features_delect_R = list(set(features_delect_R))
    return features_delect_R


def entire_preprocessing(x, y, threshold=0.95):
    features_delect = delect_standard_deviation(x)
    x = x.drop(features_delect, axis = 1)
    print('Standard deviation: {}'.format(len(features_delect)))

    features_delect_R = delect_similarity(x, y, threshold)
    x = x.drop(features_delect_R, axis = 1)
    print('Similarity: {}'.format(len(features_delect_R)))

    # Z-score
    feature_map_pep = x.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    print('Feature map shape: {}'.format(feature_map_pep.shape))
    return features_delect, features_delect_R, feature_map_pep




def Bolasso(M, alpha_, x, y, kf, cv):
  # lasso
    lasso = linear_model.Lasso(alpha=alpha_, max_iter=5000)
    feature = []

    # 分割
    train_, valid_ = {}, {}
    i = 0
    for train_index, valid_index in kf.split(x):
        train_[i] = train_index
        valid_[i] = valid_index
        i+=1

    for i in range(cv):
        # seed固定
        random.seed(i)
        # cv
        y_cv = y.iloc[train_[i]]
        x_cv = x.iloc[train_[i]]

        for j in range(M):
            N = random.randrange(int(len(y_cv)*0.1), int(len(y_cv)*0.9))
            boost = random.choices([k for k in range(len(y_cv))], k=N)
            X = x_cv.iloc[boost]
            Y = y_cv.iloc[boost]
            lasso.fit(X, Y)
            for k in range(len(lasso.coef_)):
                if lasso.coef_[k] != 0:
                    feature.append(x.columns[k])

    rank = pd.Series(feature).value_counts()
    return rank


def select_descriptors_by_Bolasso(x, y, M=20, list_alpha=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
                                  K=20, top_=10, cv=10):
    # M回ループする
    # K回以上出現したら記録
    # 上位top_次元の詳細を保存する
    # M回のループをcv回交差検証で回す

    kf = KFold(n_splits=cv, shuffle=True, random_state=2333)

    tmp_dim = []
    df_tmp = pd.DataFrame([], index=range(top_))

    for alpha_ in list_alpha:
        select_ = pd.DataFrame(Bolasso(M, alpha_, x, y, kf, cv))
        select_ = select_[select_['count'] >= K]
        tmp_dim.append(len(select_))
        select_name = select_[:top_].index.to_list()
        select_count = select_['count'][:top_].to_list()

        for j in range(top_-len(select_name)):
            select_name.append(0)
            select_count.append(0)

        df_tmp[str(alpha_)+'_name'] = select_name
        df_tmp[str(alpha_)+'_count'] = select_count

    select_dim = tmp_dim
    selece_detial = df_tmp

    return select_dim, selece_detial