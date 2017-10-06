from gensim.models.keyedvectors import KeyedVectors
from sklearn.manifold import TSNE
from pprint import pprint
import logging
import matplotlib.pyplot as plt

def test_embeddings(corpus,valid_examples,gensim_model):
    logging.info('Load the matrix using gensim word2vec model')
    tf_model_embeddings = KeyedVectors.load_word2vec_format(
        'embeddings/final_embeddings.txt',
        binary=False
    )
    tf_model_weights = KeyedVectors.load_word2vec_format(
        'embeddings/final_weights.txt',
        binary=False
    )
    for item in valid_examples:
        print item
        pprint(zip(tf_model_embeddings.most_similar(item),gensim_model.most_similar(item)))

    def plot(embeddings, labels):
        plt.figure(figsize=(10, 10))
        for i, label in enumerate(labels):
            x, y = embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                         ha='right', va='bottom')
        plt.show()

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    words = [i for i in valid_examples]
    two_d_embeddings = tsne.fit_transform(tf_model_embeddings[words])
    gensim_two_d_embeddings = tsne.fit_transform(gensim_model[words])
    plot(two_d_embeddings, words)
    plot(gensim_two_d_embeddings, words)
