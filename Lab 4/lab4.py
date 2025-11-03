#!/usr/bin/env python3

import sys
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from matplotlib import pyplot as plt
from common import describe_data, test_env
import GeneralInformation as gI
import AllData as aD
import Hobbies as hI
import PersonalInformation as pI
import YouVsOther as yO
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO
#   STUDENT SHALL ADD NEEDED IMPORTS


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)  # , nrows=4000)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'match'
    df.drop(columns=['iid', 'id'], inplace=True)
    for cols in df.columns:
        if df[cols].isna().sum() / len(df) * 100 > 10:
            df.drop(columns=cols, inplace=True)
    # print(
    #     'Left over columns after removing columns with lack of data >10%',
    #     df.columns)

    # Features can be excluded by adding column name to list
    drop_columns = [
        'attr1_1',
        'sinc1_1',
        'intel1_1',
        'fun1_1',
        'amb1_1',
        'shar1_1',
        'attr2_1',
        'sinc2_1',
        'intel2_1',
        'fun2_1',
        'amb2_1',
        'shar2_1',
        'attr3_1',
        'sinc3_1',
        'fun3_1',
        'intel3_1',
        'amb3_1',
        'pf_o_att',
        'pf_o_sin',
        'pf_o_int',
        'pf_o_fun',
        'pf_o_amb',
        'pf_o_sha'

    ]

    categorical_columns = [
        'from', 'career', 'field'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values

    y = y.astype(float)

    # Drop also dependent variable column to leave only features
    # drop_columns.append(y_column)
    for drop_column in drop_columns:
        df = df.drop(drop_column, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # TODO
    #   STUDENT SHALL HANDLE MISSING VALUES

    # replace missing categorical values with common label
    for column in categorical_columns:
        df[column] = df[column].fillna(value='Missing')

    # TODO
    #   STUDENT SHALL ENCODE CATEGORICAL FEATURES
    for column in categorical_columns:
        df = pd.get_dummies(
            df,
            prefix=[column],
            columns=[column],
            drop_first=True)

    # Replace rest of missing values (missing exam points) with 0.
    # noqa
    for col in df.columns:  # noqa
        if not categorical_columns.__contains__(col):  # noqa
            df[col] = df[col].fillna(value=0)  # noqa
    # noqa
    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


def plot_clusters(X, y, figure, file=''):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:olive']
    markers = ['o', 'X', 's', 'D']
    color_idx = 0
    marker_idx = 0

    plt.figure(figure)

    for cluster in range(0, len(set(y))):
        plt.scatter(X[y == cluster, 0], X[y == cluster, 1],
                    s=5, c=colors[color_idx], marker=markers[marker_idx])
        color_idx = 0 if color_idx == (len(colors) - 1) else color_idx + 1
        marker_idx = 0 if marker_idx == (len(markers) - 1) else marker_idx + 1

    plt.title(figure)
    # Remove axes numbers because those are not relevant for visualisation
    plt.xticks([])
    plt.yticks([])

    if file:
        plt.savefig(file, papertype='a4')

    plt.show()


def k_means_cluster(X, n_clusters=8):
    k_means_clust = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = k_means_clust.fit_predict(X)
    return y_kmeans


def hierarchical_cluster(X):
    hc = AgglomerativeClustering()
    y_hc = hc.fit_predict(X)
    return y_hc


def DB_scan_cluster(X):
    db = DBSCAN(eps=0.1, min_samples=8)
    y_DB = db.fit_predict(X)
    return y_DB


# https://stackabuse.com/k-means-clustering-with-scikit-learn/
def plot_wcss(X, max_clusters, name='all Data'):
    wcss = []
    for number_of_clusters in range(1, max_clusters):
        k_means_plot = KMeans(
            n_clusters=number_of_clusters,
            init='k-means++',
            random_state=42)
        k_means_plot.fit_predict(X)
        wcss.append(k_means_plot.inertia_)

    ks = list(range(1, max_clusters))
    plt.plot(ks, wcss)
    plt.title('The ' + name + ' Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('results/' + name + '_plot.png', format='png')
    plt.show()


def vizualize(df):
    # men_df = df[df['gender'] == 1]
    # women_df = df[df['gender'] == 0]
    # age occurrences
    age = df[np.isfinite(df['age'])]['age']

    plt.hist(age.values)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # age difference

    age = df[np.isfinite(df['d_age'])]['d_age']

    plt.hist(age.values)
    plt.xlabel('Age_Difference')
    plt.ylabel('Frequency')
    plt.show()

    # different important characteristics
    atributes_gender = df[['gender',
                           'attr1_1',
                           'sinc1_1',
                           'intel1_1',
                           'fun1_1',
                           'amb1_1',
                           'shar1_1']].groupby('gender').mean()
    cols = [
        'Attractive',
        'Sincere',
        'Intelligente',
        'Fun',
        'Ambituous',
        'Shared Interests']
    atributes_gender.columns = cols

    trace1 = go.Bar(
        y=list(atributes_gender.iloc[1]),
        x=atributes_gender.columns.values,
        name='Men',
        marker=dict(
            color='darkblue'
        )
    )
    trace2 = go.Bar(
        y=list(atributes_gender.iloc[0]),
        x=atributes_gender.columns.values,
        name='Women',
        marker=dict(
            color='pink'
        )
    )

    data1 = [trace1, trace2]
    layout = go.Layout(
        title='What People Are Looking For in the Opposite Sex',
        font=dict(
            size=16
        ),
        legend=dict(
            font=dict(
                size=16
            )
        )
    )
    fig = go.Figure(data=data1, layout=layout)
    # py.plot(fig)
    fig.show()

    # different hobbbies

    activities_interested = [
        'sports',
        'tvsports',
        'exercise',
        'dining',
        'museums',
        'art',
        'hiking',
        'gaming',
        'clubbing',
        'reading',
        'tv',
        'theater',
        'movies',
        'concerts',
        'music',
        'shopping',
        'yoga']
    activities = df.groupby(['gender']).mean()[activities_interested].values

    trace1 = go.Bar(
        x=activities_interested,
        y=activities[0, :],
        name='Women',
        # orientation = 'h',
        marker=dict(
            color='pink'
        )
    )
    trace2 = go.Bar(
        x=activities_interested,
        y=activities[1, :],
        name='Men',
        # orientation = 'h',
        marker=dict(
            color='darkblue'
        )
    )

    data3 = [trace1, trace2]
    layout = go.Layout(
        title='Interest by activities Men vs Women',
        font=dict(
            size=16
        ),
        barmode='stack',
        legend=dict(
            font=dict(
                size=16
            )
        )
    )
    fig = go.Figure(data=data3, layout=layout)
    # py.plot(fig)
    fig.show()


def t_SNE_visualization(X, titel='general'):
    X_tsne = TSNE(n_components=2,
                  random_state=0).fit_transform(X)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title('The t-SNE Method of ' + titel)
    plt.savefig('results/t_SNE_plot.png', format='png')
    plt.show()


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    # https://www.kaggle.com/datasets/whenamancodes/speed-dating?datasetId=2513916&select=Speed+Dating+Data.csv
    # for full accuracy remove the sample method but it takes longer!!!
    dates = read_data('data/Speed Dating Data.xlsx').sample(1000)

    print("Visualize data")
    vizualize(dates)

    # data overview
    print("Print data overview")
    describe_data.print_overview(
        dates, file='results/dating_overview.txt')
    describe_data.print_categorical(
        dates, file='results/dating_categorical_data.txt')
    no_match = dates[(
        dates['match'] == 0)]
    describe_data.print_overview(
        no_match, file='results/no_match_overview.txt')
    describe_data.print_categorical(
        no_match, file='results/no_match_categorical_data.txt')

    print("All Data")
    aD.main_sub()
    print("General Information")
    gI.main_sub()
    print("Personal Information")
    pI.main_sub()
    print("Hobbies Information")
    hI.main_sub()
    print("YvsO Information")
    yO.main_sub()

    print('Done')
