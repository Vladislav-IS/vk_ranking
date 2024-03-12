import pandas as pd
import numpy as np
import catboost
from sklearn.metrics import ndcg_score
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rm_corrs(df, threshold=0.95):
    corr_m = df.corr().abs()
    upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)


def train_model(file_name):
    train_df = pd.read_csv(file_name).drop_duplicates().reset_index(drop=True)
    train_df = train_df.loc[:, train_df[list(train_df)].std() != 0]
    train_target = train_df['target'].copy(deep=True)
    train_df = rm_corrs(train_df.loc[:, :'feature_78'])
    train_df['target'] = train_target
    params = {'n_estimators': [50, 100, 200, 300], 'depth': [4, 6, 10], 'learning_rate': [0.05, 0.1, 0.5]}
    boost = catboost.CatBoostClassifier(eval_metric='F1', auto_class_weights='Balanced')
    catboost_result = boost.grid_search(params, train_df.loc[:, :'feature_77'], train_df['target'])
    boost.save_model('last')
    return 'Обучение завершено, лучший результат получен для параметров: ', catboost_result['params']


def get_ndcg(file_name):
    boost = catboost.CatBoostClassifier()
    boost.load_model('last')
    test = pd.read_csv(file_name)
    test = test.loc[:, test[list(test)].std() != 0]
    test_target = test['target'].copy(deep=True)
    test = rm_corrs(test.loc[:, :'feature_78'])
    test['target'] = test_target
    ndcg = 0
    count = 0
    for query in test['search_id'].unique():  # группируем NDCG по запросам
        subdf = test[test['search_id'] == query].sort_values(by=['target'], ascending=False)
        preds = boost.predict_proba(subdf.loc[:, :'feature_77'])[:, 1]
        preds = (preds >= 0.5)
        if subdf.head(1)['target'].values:
            if subdf.shape[0] > 1:
                ndcg += ndcg_score([subdf['target']], [preds.astype(int)])
            else:
                ndcg += 1 * preds
            count += 1
    return 'NDCG score = ' + str(ndcg[0]/count)
