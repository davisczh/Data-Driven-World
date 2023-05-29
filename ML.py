import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def comb_dfs(dfs=None):  # input 10 dataframes with country-year and 3year avg predictors as columns in a *list*.
    # returns one csv with contry-year, target, and 10 cols of predictors

    df_main = dfs.pop()

    for index in range(len(dfs)):
        # df_main = pd.concat([df_main, dfs[index]], axis=1, ignore_index=False)
        df_main = df_main.merge(dfs[index])
    return df_main


def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target


def normalize_z(df):
    dfout = (df - df.mean(axis=0)) / df.std(axis=0)
    return dfout


def prepare_feature(df_feature):
    cols = len(df_feature.columns)
    feature = df_feature.to_numpy().reshape(-1, cols)
    X = np.concatenate((np.ones((feature.shape[0], 1)), feature), axis=1)
    return X


def prepare_target(df_target):
    return df_target.to_numpy()


def predict(df_feature, beta):
    X = prepare_feature(normalize_z(df_feature))
    return calc_linear(X, beta)


def calc_linear(X, beta):
    return np.matmul(X, beta)

def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    np.random.seed(random_state)
    size = df_feature.shape[0] * test_size
    row_idxs = np.array(range(df_feature.shape[0]))

    test_set = np.random.choice(row_idxs, size=int(size), replace=False)
    mask = np.ones(len(row_idxs), dtype=bool)
    for i in range(len(row_idxs)):
        if row_idxs[i] in test_set:
            mask[i] = False
    train_set = row_idxs[mask]

    df_feature_train = df_feature.iloc[train_set, :]
    df_feature_test = df_feature.iloc[test_set, :]
    df_target_train = df_target.iloc[train_set, :]
    df_target_test = df_target.iloc[test_set, :]
    return df_feature_train, df_feature_test, df_target_train, df_target_test


def compute_cost(X, y, beta):  # can use other cost functions
    m = X.shape[0]
    yhat = calc_linear(X, beta)
    cost = yhat - y
    cost_2 = np.matmul(cost.T, cost)
    return 1 / (2 * m) * cost_2


def gradient_descent(X, y, beta, alpha, num_iters):
    m = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    for i in range(num_iters):
        beta = (beta
                - alpha
                * 1 / m
                * np.matmul(X.T, (calc_linear(X, beta) - y)))
        J_storage[i] = compute_cost(X, y, beta)
    return beta, J_storage


def mean_squared_error(target, pred):
    ###
    error = target - pred
    error_sum_sq = np.matmul(error.T, error)
    mse = (1 / target.shape[0]) * error_sum_sq
    return mse[0][0]


def adj_r2_score(y, ypred, n, k):  # n is number of observations (im guessing is number of test set??? ), k is number of predictors
    y = y.to_numpy()
    tot = np.sum((y - np.mean(y))**2)
    res = np.sum((y - ypred)**2)
    r2 = 1 - res/tot
    adjr2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))  # adjusted r2 formula from online
    return adjr2


def learning_main(df=[], feature_names=[], target_names=[], iter=1500, alpha=0.01):  # CHANGE CSVS TO DFS!!!!!

    df = comb_dfs(df) # to test if it works with the 10 data frames

    # df = pd.read_csv(fr'C:\Users\dczqd\OneDrive - Singapore University of Technology and Design\Documents\SUTD\term 3\ddw\{csvs[0]}')
    # i used the above line to test the machine learning codes below
    df_features, df_targets = get_features_targets(df, feature_names, target_names)

    # do any transformation here (linear regression thing might need to x^2 the featues etc.)

    df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_targets, 100, 0.3)
    df_features_train_z = normalize_z(df_features_train)

    X = prepare_feature(df_features_train_z)
    target = prepare_target(df_target_train)

    beta = np.zeros((len(feature_names) + 1, 1))
    beta, J = gradient_descent(X, target, beta, alpha, iter)
    target = df_target_test
    pred = predict(df_features_test, beta)

    plt.scatter(df_features_test[feature_names[0]], pred)
    plt.scatter(df_features_test[feature_names[0]], target)
    plt.show()
    print(beta)

    adjr2 = adj_r2_score(df_target_test, pred, len(df_target_test),
                         len(feature_names))  # double check if input is correct
    return adjr2  # or adjust r2

#
# learning_main(['housing_processed.csv'], ['RM', 'DIS', 'INDUS'], ['MEDV'], 1500,
#               0.01)  # FIRST INPUT SHOULD BE 10 DATAFRAMES IN A LIST
