import matplotlib.pyplot as plt
import pylab as pl
from sklearn.decomposition import PCA

def plot_clusters(data, k, labels, feature):
    pca = PCA(n_components=2).fit(data)
    pca_2d = pca.transform(data)
    pl.figure('K-means with '+str(k)+' clusters using '+str(feature))
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels)
    pl.savefig('../Plots/'+str(feature)+'_'+str(k)+'.png')
