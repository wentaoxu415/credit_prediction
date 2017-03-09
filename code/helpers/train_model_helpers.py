import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    make_scorer,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
f2_score = make_scorer(fbeta_score, beta=2, pos_label=True, average='binary')


def make_input_df(df):
    X = pd.DataFrame()
    X['CASHMTA'] = df['CASHMTA_win']
    X['EXRET_AVG'] = df['EXRET_AVG_win']
    X['MB'] = df['MB_win']
    X['NIMTA_AVG'] = df['NIMTA_AVG_win']
    X['PRICE'] = df['PRICE_win']
    X['RSIZE'] = df['RSIZE_win']
    X['SIGMA'] = df['SIGMA_win']
    X['TLMTA'] = df['TLMTA_win']

    return X


def make_output_df(df):
    y = pd.DataFrame()
    y['is_bankrupt'] = df['is_bankrupt_within_12']

    return y


def make_train_and_test_df(train_df, test_df):
    X_train = make_input_df(train_df)
    X_test = make_input_df(test_df)
    y_train = make_output_df(train_df)
    y_test = make_output_df(test_df)

    return X_train, X_test, y_train, y_test


def make_base_rate_model(X):
    y = np.zeros(X.shape[0], dtype=bool)

    return y


def make_time_series_iterable(df, start_year):
    end_year = df['year'].max() + 1

    iterator = []
    for year in range(start_year, end_year):
        train_index = df[df['year'] < year].index.values.astype(int)
        test_index = df[df['year'] == year].index.values.astype(int)
        iterator.append((train_index, test_index))

    return iterator


def plot_roc_curve(y_test, y_pred):
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


def plot_confusion_matrix(test_output, predicted_output, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(test_output, predicted_output)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def display_all_results(x_test, y_test, estimator, probability=False):
    y_pred_bin = estimator.predict(x_test)
    print("F1 Score")
    print(f1_score(y_test, y_pred_bin))

    print("F2 Score")
    print(fbeta_score(y_test, y_pred_bin, beta=2, pos_label=True, average='binary'))

    print("Recall Score")
    print(recall_score(y_test, y_pred_bin))

    print("Precision Score")
    print(precision_score(y_test, y_pred_bin))

    print("Detailed classification report:")

    print(classification_report(y_test, y_pred_bin))
    print()

    print("Confusion Matrix:")
    plot_confusion_matrix(y_test, y_pred_bin, ['Active', 'Bankrupt'])
    print()

    if probability:
        print("Roc Curve:")
        y_pred_prob = estimator.predict_proba(x_test)[:, 1]
        plot_roc_curve(y_test, y_pred_prob)
        print()


def train_model_by_random_search_cv(x_train, x_test, y_train, y_test, estimator, tuned_parameters, custom_iterable=None, num_iter=10, scores=None, probability=False):

    if scores is None:
        scores = ['roc_auc', 'precision']

    for score in scores:
        print("Tuning hyper-parameters for %s" % score)
        print()

        clf = RandomizedSearchCV(estimator, tuned_parameters, cv=custom_iterable, scoring=score, verbose=1, n_iter=num_iter, n_jobs=-1)
        clf.fit(x_train, y_train.values.ravel())

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()

        display_all_results(x_test, y_test, clf, probability)

    return clf


def train_model_by_grid_search_cv(x_train, x_test, y_train, y_test, estimator, tuned_parameters, custom_iterable, scores=None, probability=False):

    if scores is None:
        scores = ['roc_auc', 'precision']

    for score in scores:
        print("Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(estimator, tuned_parameters, cv=custom_iterable, scoring=score, verbose=1, n_jobs=-1)
        clf.fit(x_train, y_train.values.ravel())

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()

        display_all_results(x_test, y_test, clf, probability)

    return clf
