import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
from copy import deepcopy
from .ipython_helpers import print_full
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def run_trainers(trainers, names):
    # display results for entire test set
    for i, trainer in enumerate(trainers):
        print("******************************entire test set - {0} model******************************".format(names[i]))
        trainer.make_dev_and_test_df(threshold_year=2005)
        trainer.train_model()
        trainer.display_results(trainer.y_test, trainer.y_predicted)

    # display results for delta test set
    for i, trainer in enumerate(trainers):
        print("******************************delta test set - {0} model******************************".format(names[i]))
        trainer.display_results(trainer.y_test_delta, trainer.y_predicted_delta)


class RatingModelTrainer:
    def __init__(self, df, model, output_type, dev_type):
        self.df = df.copy(deep=True)
        self.model = deepcopy(model)
        self.output_type = output_type
        self.dev_type = dev_type

    def make_dev_and_test_df(self, threshold_year):
        self._make_state_columns()
        self.dev_df = self.df[self.df['year'] <= threshold_year]
        self.test_df = self.df[self.df['year'] > threshold_year]

        if self.dev_type == 'whole':
            specified_dev_df = self.dev_df
        elif self.dev_type == 'delta':
            self.dev_delta_df = self._make_dev_delta_df()
            specified_dev_df = self.dev_delta_df
        elif self.dev_type == 'mixed':
            self.dev_mixed_df = self._make_dev_mixed_df()
            specified_dev_df = self.dev_mixed_df

        self.X_train = self._make_input_df(specified_dev_df)
        self.y_train = self._make_output_df(specified_dev_df)
        self.X_test = self._make_input_df(self.test_df)
        self.y_test = self._make_output_df(self.test_df)

    def _make_dev_delta_df(self):
        return self.dev_df[self.dev_df['current_state'] != self.dev_df['next_state']]

    def _make_dev_mixed_df(self):
        dev_delta_df = self._make_dev_delta_df()
        dev_static_df = self.dev_df[~self.dev_df.index.isin(dev_delta_df.index)]
        np.random.seed(0)
        static_index = np.random.choice(dev_static_df.index, size=dev_delta_df.shape[0], replace=False)
        mixed_index = np.append(static_index, dev_delta_df.index)

        return self.dev_df[self.dev_df.index.isin(mixed_index)]

    def _make_state_columns(self):
        if self.output_type == 'ranking':
            self.df.loc[:, 'current_state'] = self.df['ranking']
            self.df.loc[:, 'next_state'] = self.df['next_ranking']
        elif self.output_type == 'windsorized_ranking':
            self.df.loc[:, 'current_state'] = self.df['windsorized_ranking']
            self.df.loc[:, 'next_state'] = self.df['next_windsorized_ranking']
        elif self.output_type == 'broad_ranking':
            self.df.loc[:, 'current_state'] = self.df['broad_ranking']
            self.df.loc[:, 'next_state'] = self.df['next_broad_ranking']
        elif self.output_type == 'is_investment_grade':
            self.df.loc[:, 'current_state'] = self.df['is_investment_grade']
            self.df.loc[:, 'next_state'] = self.df['next_is_investment_grade']

    def _make_input_df(self, df):
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

    def _make_output_df(self, df):
        return df['next_state'].ravel()

    def train_model(self, predict_proba=False):
        self.model.fit(self.X_train, self.y_train)
        self.y_predicted = self.model.predict(self.X_test)
        self._make_delta_df()

    def _mark_rating_change_direction(self, row, change_type):
        current_state = row['current_state']

        if change_type == 'actual':
            next_state = row['next_state']
        elif change_type == 'predicted':
            next_state = row['predicted_state']

        if current_state == next_state:
            return "keep"
        if current_state > next_state:
            return "upgrade"
        elif current_state < next_state:
            return "downgrade"

    def _make_delta_df(self):
        self.test_df.loc[:, 'predicted_state'] = self.y_predicted
        self.test_df.loc[:, 'next_direction'] = self.test_df.apply(lambda row: self._mark_rating_change_direction(row, 'actual'), axis=1)
        self.test_df.loc[:, 'predicted_direction'] = self.test_df.apply(lambda row: self._mark_rating_change_direction(row, 'predicted'), axis=1)

        self.test_delta_df = self.test_df[
            (self.test_df['current_state'] != self.test_df['next_state']) |
            (self.test_df['current_state'] != self.test_df['predicted_state'])
        ]
        self.y_test_delta = self.test_delta_df['next_state']
        self.y_predicted_delta = self.test_delta_df['predicted_state']

    def _make_results_df(self, true_classes, predicted_classes):
        results_dict = OrderedDict()
        score_types = ['Precision', 'Recall', 'F1 Score']
        average_types = ['micro', 'macro', 'weighted']

        for score_type in score_types:
            results_dict[score_type] = []
            for average_type in average_types:
                if score_type == 'Precision':
                    score = precision_score(true_classes, predicted_classes, average=average_type)
                elif score_type == 'Recall':
                    score = recall_score(true_classes, predicted_classes, average=average_type)
                elif score_type == 'F1 Score':
                    score = f1_score(true_classes, predicted_classes, average=average_type)
                score = round(score, 4)
                results_dict[score_type].append(score)

        return pd.DataFrame(results_dict, index=average_types)

    def display_results(self, true_classes, predicted_classes, predict_proba=False):
        results_df = self._make_results_df(true_classes, predicted_classes)
        print_full(results_df)

        print("Detailed classification report:")
        print(classification_report(true_classes, predicted_classes))

        print("Confusion Matrix:")
        self._display_confusion_matrix(true_classes, predicted_classes)

        if predict_proba:
            print("Roc Curve:")
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            self._display_roc_curve()

    def _display_confusion_matrix(self, true_classes, predicted_classes):
        change_directions = ["downgrade", "keep", "upgrade"]
        if self.output_type == 'is_investment_grade':
            classes = [True, False]
        elif any(direction in true_classes for direction in change_directions):
            classes = change_directions
        else:
            min_ranking = int(self.y_test.min())
            max_ranking = int(self.y_test.max())
            classes = [x for x in range(min_ranking, max_ranking + 1)]

        cm = confusion_matrix(true_classes, predicted_classes, labels=classes)
        cmap = plt.cm.Blues

        #plt.figure(figsize=(8, 8))
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        print('Confusion matrix, without normalization')
        print_full(pd.DataFrame(cm, columns=classes, index=classes))

        thresh = cm.max() * 3/ 4.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                fontsize='small',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        plt.show()

    def _display_roc_curve(self):
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.y_pred_proba)
        plt.figure()
        plt.plot(self.fpr, self.tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()

    def make_time_series_iterable(self, start_year=1996):
        end_year = self.dev_df['year'].max() + 1
        iterator = []
        for year in range(start_year, end_year):
            train_index = self.dev_df[self.dev_df['year'] < year].index.values.astype(int)
            test_index = self.dev_df[self.dev_df['year'] == year].index.values.astype(int)
            iterator.append((train_index, test_index))

        self.iterator = iterator
