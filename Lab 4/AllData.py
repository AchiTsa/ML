from sklearn.manifold import TSNE

import lab4 as ma
import numpy as np
from sklearn.decomposition import PCA


def main_sub():
    #for full accuracy remove the sample method but it takes longer!!!
    dates = ma.read_data('data/Speed Dating Data.xlsx').sample(1000)

    # preprocess data
    match_X, match_y = ma.preprocess_data(dates, True)

    # elbow method plot
    ma.plot_wcss(match_X, max_clusters=100)

    # get values of df
    X_val = match_X.values

    # clustering with kmeans++
    y_clust = ma.k_means_cluster(X_val)

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


if __name__ == '__main__':
    main_sub()
