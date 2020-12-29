import networkx as nx
import urllib
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from community import community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.base import ClusterMixin
from IPython.core.display import display

logger = logging.getLogger()
logging.basicConfig(level=logging.WARN)
logging.getLogger('matplotlib').disabled = True
logging.getLogger('gensim').disabled = True


CUSTOM_STOP_WORDS = ['hotel', 'room']
nlp_light = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

def create_directed_graph(edge_list, link_threshold=0):
    G = nx.DiGraph()

    duplicated_edge = 0
    self_loop = 0
    for entry in edge_list:
        source = urllib.parse.unquote(entry['name'])
        if len(entry['links']) >= link_threshold:
            G.add_node(source)
            for l in entry['links']:
                target = urllib.parse.unquote(l)
                if G.has_edge(*(source, target)):
                    duplicated_edge += 1
                else:
                    if target == source:
                        self_loop += 1
                    else:
                        G.add_edge(*(source, target))
    logger.info(f'number of duplicated edge: {duplicated_edge}')
    logger.info(f'number of self loop: {self_loop}')

    return G

def plot_degree_dist(graph):
    degrees = pd.Series([graph.degree(n) for n in graph.nodes()])
    display((degrees.value_counts().sort_values())/graph.number_of_nodes())

    degrees_df = degrees.value_counts().reset_index()
    degrees_df.columns = ['x', 'y']
    degrees_df['log_x'] = np.log(degrees_df['x'])
    degrees_df['log_y'] = np.log(degrees_df['y'])
    fig, axs = plt.subplots(nrows=2, figsize=(15, 20))

    axs[0].hist(degrees)
    sns.regplot(data=degrees_df, x='log_x', y='log_y', fit_reg=True, order=1, ax=axs[1])


def generate_community(network):
    communities = community_louvain.best_partition(network)
    community_members = {}
    for k, v in communities.items():
        if v in community_members:
            community_members[v].append(k)
        else:
            community_members[v] = [k]

    return community_members


def get_top_n_community(community, n=10000):
    if n > len(community):
        raise ValueError(f'There are less than {n} communities for subset')
    top_n_index = np.argsort([len(v) for v in community.values()])[::-1][:n]
    top_n_community = {i: v for i, (k, v) in enumerate(community.items())
                       if k in top_n_index}
    return top_n_community

def generate_wordcloud(text):
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud)
    plt.axis('off')


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


class ReviewClusterer(ClusterMixin):
    def __init__(self, n_features, n_clusters):
        self.n_features = n_features,
        self.n_clusters = n_clusters


        vectoriser = TfidfVectorizer(
            max_features=500,#self.n_features,
            stop_words='english',
            tokenizer=self._spacy_tokenizer
        )

        dense_transformer = DenseTransformer()
        scaler = StandardScaler()
        km_cluster = KMeans(n_clusters=self.n_clusters)

        self.pipeline = Pipeline(steps=[
            ('vectoriser', vectoriser),
            ('dense_transformer', dense_transformer),
            ('scaler', scaler),
            ('km_cluster', km_cluster)
        ])

    def _spacy_tokenizer(sel, doc):
        return [x.orth_ for x in nlp_light(doc)
                if not x.is_stop and x.is_alpha and len(x.text) > 2]

    def fit(self, X):
        self.pipeline.fit(X)

        # create cluster result dataframe
        feature_names = self.pipeline.named_steps['vectoriser'].get_feature_names()
        centers = self.pipeline.named_steps['km_cluster'].cluster_centers_
        self.cluster_center_df_ = pd.DataFrame(centers, columns=feature_names).T

        return self

    def transform(self, X):
        self.pipeline.transform(X)

    def visualise_cluster(self):
        plt.figure(figsize=(30, 20))
        sns.heatmap(self.cluster_center_df_)

    def inspect_cluster_keywords(self, k=10):
        for i in range(self.n_clusters):
            print(f'cluster {i}: ' + ' + '.join(
                self.cluster_center_df_[i]
                .sort_values()
                .tail(k)
                .index.tolist())
            )
