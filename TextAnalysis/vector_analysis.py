from gensim.models.doc2vec import Doc2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from PaperProcessing import process_abstract
from sklearn.cluster import KMeans
import pandas as pd
from nltk.tokenize import word_tokenize


def get_doc2vec_model(tagged_documents,
                      vector_size,
                      window=5,
                      min_count=1,
                      workers=4,
                      epochs=100):
    return Doc2Vec(documents=tagged_documents,
                   vector_size=vector_size,
                   window=window,
                   min_count=min_count,
                   workers=workers,
                   epochs=epochs)


def get_n_similar(model : Doc2Vec, pos_doc, neg_doc, n):
    pos_doc_vec = model.infer_vector(word_tokenize(pos_doc))

    neg_doc_vec = None
    if neg_doc:
        neg_doc_vec = model.infer_vector(word_tokenize(neg_doc))
    most_sim = model.docvecs.most_similar(positive=pos_doc_vec, negative=neg_doc_vec, topn=n)
    return most_sim


class VectorAnalysis:

    def __init__(self, n_sim, doc_vec_model: Doc2Vec, all_raw_abstracts):
        self.n_sim = n_sim

        self.doc_indices = [pair[0] for pair in n_sim]
        raw_abstracts = [all_raw_abstracts[idx] for idx in self.doc_indices]
        self.raw_abstracts = raw_abstracts
        self.all_abstracts = all_raw_abstracts
        self.vectors = np.array([doc_vec_model.dv[idx] for idx in self.doc_indices])
        self.dv_model = doc_vec_model
        pca = PCA(n_components='mle', svd_solver='full')
        pca.fit(self.vectors)
        self.pca_vectors = pca.fit_transform(self.vectors)
        self.pca = pca

        tsne = TSNE(n_components=2)
        self.tsne_vectors = tsne.fit_transform(self.pca_vectors)
        self.tsne = tsne

        self.k_means = self.get_best_kmeans()

    def get_2d_representations(self):
        return self.tsne_vectors

    def get_best_kmeans(self):
        best_inertia = np.infty
        best_model = None
        for n_cluster in range(3, 8):
            k_means = KMeans(n_clusters=n_cluster)
            k_means.fit(self.pca_vectors)
            cur_inertia = k_means.inertia_
            if best_inertia > cur_inertia:
                best_model = k_means
        return best_model

    # return the representative documents for each cluster
    def get_cluster_documents(self):
        cluster_centers_low_dim = self.k_means.cluster_centers_

        # what would these cluster centers look like in the original doc2vec space?
        cluster_centers_high_dim = self.pca.inverse_transform(cluster_centers_low_dim)
        closest_docs = []
        for idx in range(cluster_centers_high_dim.shape[0]):
            top_match = self.dv_model.docvecs.most_similar(cluster_centers_high_dim[idx, :], topn=1)[0]
            closest_docs.append(top_match)
        return closest_docs

    def get_results(self):
        # dataframe for 2d representation
        df_2d = pd.DataFrame()
        df_2d['X'] = self.tsne_vectors[:, 0]
        df_2d['Y'] = self.tsne_vectors[:, 1]
        df_2d['Doc Indices'] = self.doc_indices
        sim_scores = np.array([pair[1] for pair in self.n_sim]).reshape((-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 5))
        df_2d['Similarity Score'] =scaler.fit_transform(sim_scores).reshape(-1, 1)
        df_2d['Titles'] = [pair[0] for pair in self.raw_abstracts]
        df_2d['Abstracts'] = [pair[1] for pair in self.raw_abstracts]
        doc_cluster_indices = self.k_means.predict(self.pca_vectors)
        df_2d['Cluster Idx'] = doc_cluster_indices
        doc_cluster_dists = self.k_means.transform(self.pca_vectors)
        df_2d['Distance to Centroid'] = [np.linalg.norm(self.k_means.cluster_centers_[doc_cluster_indices[idx]]
                                                             - self.pca_vectors[idx, :])
                                              for idx in range(len(self.pca_vectors))]
        print(len(df_2d['Distance to Centroid'] ))



        # dataframe for clustering summary data
        closest_docs = self.get_cluster_documents()
        df_clustering_summary = pd.DataFrame()
        df_clustering_summary['Closest Doc Idx'] = [doc[0] for doc in closest_docs]
        df_clustering_summary['Group'] = [f"Cluster {idx}; Papers like \"{self.all_abstracts[pair[0]][0]}\""
                                                  for idx, pair in enumerate(closest_docs)]
        df_clustering_summary['Dist from Center Vector'] = [doc[1] for doc in closest_docs]
        df_clustering_summary['Number of Docs'] = np.bincount(doc_cluster_indices)

        return df_2d, df_clustering_summary





