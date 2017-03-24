import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataExplorer:
    def __init__(self, df):
        self.df = df

    def make_streak_df(self, period):
        ranking_type_list = ['ranking', 'windsorized_ranking', 'broad_ranking', 'is_investment_grade']
        for ranking_type in ranking_type_list:
            ranking_list = []
            streaks_list = []

            last_gvkey = self.df.iloc[0]['GVKEY']
            last_ranking = self.df.iloc[0][ranking_type]
            last_month_diff = self.df.iloc[0]['month_diff']
            streaks = 1

            for index, row in self.df.iloc[1:].iterrows():
                gvkey = row['GVKEY']
                ranking = row[ranking_type]
                month_diff = row['month_diff']

                if (last_gvkey == gvkey) and (last_ranking == ranking) and (last_month_diff + period == month_diff):
                    streaks += 1
                else:
                    ranking_list.append(last_ranking)
                    streaks_list.append(streaks)
                    streaks = 1

                last_gvkey = gvkey
                last_ranking = ranking
                last_month_diff = month_diff

            ranking_list.append(last_ranking)
            streaks_list.append(streaks)

            streak_df = pd.DataFrame({'ranking': ranking_list, 'streaks': streaks_list})
            if ranking_type == 'ranking':
                self.streak_df = streak_df
            elif ranking_type == 'windsorized_ranking':
                self.streak_windsorized_ranking_df = streak_df
            elif ranking_type == 'broad_ranking':
                self.streak_broad_ranking_df = streak_df
            elif ranking_type == 'is_investment_grade':
                self.streak_is_investment_grade_df = streak_df

    # Plot functions plot charts
    def plot_count_ranking(self, ranking_type):
        if ranking_type == 'ranking':
            ax = sns.countplot(data=self.df, x=ranking_type, color='c')
            ax.set(xlabel="Ranking", ylabel='Count')
        elif ranking_type == 'windsorized_ranking':
            ax = sns.countplot(data=self.df, x=ranking_type, color='c')
            ax.set(xlabel="Windsorized Ranking", ylabel='Count')
        elif ranking_type == 'broad_ranking':
            ax = sns.countplot(data=self.df, x=ranking_type, color='c')
            ax.set(xlabel="Broad Ranking", ylabel='Count')
        elif ranking_type == 'is_investment_grade':
            ax = sns.countplot(data=self.df, x=ranking_type, order=(True, False), color='c')
            ax.set(xlabel="Is Investment Grade", ylabel='Count')

    def plot_heat_map_current_to_next_state(self, ranking_type):
        if ranking_type == 'ranking':
            df = self.df[['ranking', 'next_ranking']]
            df = df.pivot_table(index='ranking', columns='next_ranking', aggfunc=len, fill_value=0)
        elif ranking_type == 'windsorized_ranking':
            df = self.df[['windsorized_ranking', 'next_windsorized_ranking']]
            df = df.pivot_table(index='windsorized_ranking', columns='next_windsorized_ranking', aggfunc=len, fill_value=0)
        elif ranking_type == 'broad_ranking':
            df = self.df[['broad_ranking', 'next_broad_ranking']]
            df = df.pivot_table(index='broad_ranking', columns='next_broad_ranking', aggfunc=len, fill_value=0)
        elif ranking_type == 'is_investment_grade':
            df = self.df[['is_investment_grade', 'next_is_investment_grade']]
            df = df.pivot_table(index='is_investment_grade', columns='next_is_investment_grade', aggfunc=len, fill_value=0)

        sns.heatmap(df, fmt="d", linewidths=0.5)

        return df

    def plot_hist_ranking_transition(self, ranking_type):
        max_value = self.df[ranking_type].max()

        if ranking_type == 'ranking':
            next_ranking_type = 'next_ranking'
            font_size = 8
        elif ranking_type == 'windsorized_ranking':
            next_ranking_type = 'next_windsorized_ranking'
            font_size = 10
        elif ranking_type == 'broad_ranking':
            next_ranking_type = 'next_broad_ranking'
            font_size = 12
        elif ranking_type == 'is_investment_grade':
            next_ranking_type = 'next_is_investment_grade'

        g = sns.FacetGrid(self.df, col=ranking_type, col_wrap=4, sharex=False)
        g.map(plt.hist, next_ranking_type, bins=range(1, max_value + 2, 1), color='c', align='left')
        g.set(xticks=range(1, max_value + 1, 1))

        if ranking_type != 'is_investment_grade':
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_fontsize(font_size)
                    label.set_rotation(90)

    def plot_hist_streaks(self, ranking_type):
        if ranking_type == 'ranking':
            streak_df = self.streak_df
        elif ranking_type == 'windsorized_ranking':
            streak_df = self.streak_windsorized_ranking_df
        elif ranking_type == 'broad_ranking':
            streak_df = self.streak_broad_ranking_df
        elif ranking_type == 'is_investment_grade':
            streak_df = self.streak_is_investment_grade_df
        sns.distplot(streak_df['streaks'], kde=False, color='c')

    def plot_box_streaks(self, ranking_type):
        if ranking_type == 'ranking':
            streak_df = self.streak_df
        elif ranking_type == 'windsorized_ranking':
            streak_df = self.streak_windsorized_ranking_df
        elif ranking_type == 'broad_ranking':
            streak_df = self.streak_broad_ranking_df
        elif ranking_type == 'is_investment_grade':
            streak_df = self.streak_is_investment_grade_df
        sns.boxplot(streak_df['streaks'], color='c')

    def plot_box_streaks_by_ranking(self, ranking_type):
        if ranking_type == 'ranking':
            streak_df = self.streak_df
        elif ranking_type == 'windsorized_ranking':
            streak_df = self.streak_windsorized_ranking_df
        elif ranking_type == 'broad_ranking':
            streak_df = self.streak_broad_ranking_df
        elif ranking_type == 'is_investment_grade':
            streak_df = self.streak_is_investment_grade_df
        sns.boxplot(data=streak_df, x='streaks', y='ranking', color='c', orient='h')

    # Display functions show statistics in table
    def display_stats_streaks(self, ranking_type):
        if ranking_type == 'ranking':
            streak_df = self.streak_df
        elif ranking_type == 'windsorized_ranking':
            streak_df = self.streak_windsorized_ranking_df
        elif ranking_type == 'broad_ranking':
            streak_df = self.streak_broad_ranking_df
        elif ranking_type == 'is_investment_grade':
            streak_df = self.streak_is_investment_grade_df
        return pd.DataFrame(streak_df['streaks'].describe().transpose())

    def display_stats_streaks_by_ranking(self, ranking_type):
        if ranking_type == 'ranking':
            streak_df = self.streak_df
        elif ranking_type == 'windsorized_ranking':
            streak_df = self.streak_windsorized_ranking_df
        elif ranking_type == 'broad_ranking':
            streak_df = self.streak_broad_ranking_df
        elif ranking_type == 'is_investment_grade':
            streak_df = self.streak_is_investment_grade_df
        df = streak_df.groupby('ranking').describe().reset_index()
        df = df.pivot(index='ranking', columns='level_1', values='streaks')
        df = df[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

        return df
