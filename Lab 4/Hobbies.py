from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import lab4 as ma
from sklearn.manifold import TSNE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main_sub():
    # for full accuracy remove the sample method but it takes longer!!!
    dates = ma.read_data('data/Speed Dating Data.xlsx').sample(1000)

    # preprocess data
    match_X, match_y = ma.preprocess_data(dates, True)

    # Personal Information we want to have a closer look on
    possible_criteria_hobbies = [
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
        'match',
        'yoga'
    ]

    match_X = match_X[possible_criteria_hobbies]
    # possible clusters 2,3,6
    number_clusters = 10
    # elbow method plot
    ma.plot_wcss(match_X, max_clusters=number_clusters, name='Hobbies Data')

    # get values of df
    X_val = match_X.values

    # clustering with kmeans++
    y_clust = ma.k_means_cluster(X_val, n_clusters=6)

    # t-SNE visualization
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_val)

    # There are no clusters. Create fake array with one cluster
    ma.plot_clusters(X_tsne, np.full(X_tsne.shape[0], 0),
                     'All Data t-SNE visualisation without clusters')

    ma.plot_clusters(X_tsne, y_clust, 'All Data clusters with TSNE')

    # Visualise with PCA - but PCA primary goal is not visualisation
    X_pca = PCA(n_components=2, random_state=0).fit_transform(X_val)
    ma.plot_clusters(X_pca, np.full(X_pca.shape[0], 0),
                     'All Data PCA visualisation without clusters')
    ma.plot_clusters(X_pca, y_clust,
                     'All Data PCA visualisation with clusters')
    # add cluster information
    match_X['cluster'] = y_clust
    print('Hobbies Information Kmeans pairplot')

    # sample data to not take too long
    # if you want more accurate charts than increase n_samples >=1000
    n_samples = 200
    match_X_sampled1 = match_X.sample(n_samples)

    # pairplot data for correlations
    sns_plot1 = sns.pairplot(match_X_sampled1)  # , hue='cluster')
    sns_plot1.savefig("results/pairplotHobbies.png")
    plt.clf()  # Clean parirplot figure from sns

    print('Hobbies Information clustering is done!')


if __name__ == '__main__':
    main_sub()
