import time
from utils import get_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import gensim
from random import randint
import numpy as np
import logging
from scipy.sparse import csr_matrix
from utils import get_class_labels

logger = logging.getLogger()
logger.setLevel("INFO")

def subgraph2vec_tokenizer (s):
    '''
    Tokenize the string from subgraph2vec sentence (i.e. <target> <context1> <context2> ...). Just target is to be used
    and context strings to be ignored.
    :param s: context of subgraph2vec file.
    :return: List of targets from subgraph2vec file.
    '''
    return [line.split(' ')[0] for line in s.split('\n')]

def get_subgraph_kernel (gensim_model, subgraph_vocab):
    '''
    compute the subgraph kernel (diagonal matrix) from the subgraph vectors
    :param gensim_model: gensim model containing the subgraph vectors
    :param subgraph_vocab: vocabulary of rooted subgraphs
    :return: subgraph kernel (diag matrix)
    '''
    t0 = time.time()
    subgraph_kernel = csr_matrix ((len(subgraph_vocab), len(subgraph_vocab)))
    for i,subgraph in enumerate(subgraph_vocab):
        subgraph_vector = gensim_model[subgraph]
        subgraph_self_sim = np.dot(subgraph_vector, subgraph_vector.T)
        subgraph_kernel[i,i] = subgraph_self_sim
    logging.info('Computed subgraph kernel matrix (i.e., M) in {} sec'.format(round(time.time()-t0)))
    return subgraph_kernel

def linear_kernel_svm_classify (X_train, X_test, Y_train, Y_test):
    '''
    Classifier with WL kernel
    :param X_train: training feature vectors
    :param X_test: testing feature vectors
    :param Y_train: training set labels
    :param Y_test: test set labels
    :return: None
    '''
    classifier = SVC (kernel='precomputed')

    train_kernel = np.dot(X_train, X_train.T)
    classifier.fit(train_kernel.toarray(), Y_train)

    test_kernel = np.dot(X_test, X_train.T)
    Y_pred = classifier.predict(test_kernel.toarray())

    acc = accuracy_score(Y_test, Y_pred)
    print 'SVM with Weisfiler-Lehman kernel, accuracy: {}'.format(acc)

    print classification_report(Y_test, Y_pred)

def deep_kernel_svm_classify (X_train, X_test, Y_train, Y_test, subgraph_kernel):
    '''
    Classifier with deep WL kernel
    :param X_train: training feature vectors
    :param X_test: testing feature vectors
    :param Y_train: training set labels
    :param Y_test: test set labels
    :return: None
    '''
    classifier = SVC (kernel='precomputed')

    train_kernel = np.dot(np.dot(X_train, subgraph_kernel), X_train.T)
    classifier.fit(train_kernel.toarray(), Y_train)

    test_kernel = np.dot(np.dot(X_test, subgraph_kernel), X_train.T)
    Y_pred = classifier.predict(test_kernel.toarray())

    acc = accuracy_score(Y_test, Y_pred)
    print 'SVM with deep Weisfiler-Lehman kernel, accuracy: {}'.format(acc)

    print classification_report(Y_test, Y_pred)


def perform_classification (corpus_dir, extn, embedding_fname, class_labels_fname):
    '''
    Perform classification from
    :param corpus_dir: folder containing subgraph2vec sentence files
    :param extn: extension of subgraph2vec sentence files
    :param embedding_fname: file containing subgraph vectors in word2vec format (refer Mikolov et al (2013) code)
    :param class_labels_fname: files containing labels of each graph
    :return: None
    '''
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(fname=embedding_fname)
    logging.info('Loaded gensim model of subgraph vectors')

    subgraph_vocab = sorted(gensim_model.vocab.keys())
    logging.info('Vocab consists of {} subgraph features'.format(len(subgraph_vocab)))

    wlk_files = get_files(corpus_dir, extn)
    logging.info('Loaded {} graph WL kernel files for performing classification'.format(len(wlk_files)))
    c_vectorizer = CountVectorizer(input='filename',
                                   tokenizer=subgraph2vec_tokenizer,
                                   lowercase=False,
                                   vocabulary=subgraph_vocab)
    normalizer = Normalizer()

    X = c_vectorizer.fit_transform(wlk_files)
    X = normalizer.fit_transform(X)
    logging.info('X (sample) matrix shape: {}'.format(X.shape))


    Y = np.array(get_class_labels(wlk_files, class_labels_fname))
    logging.info('Y (label) matrix shape: {}'.format(Y.shape))

    seed = randint(0, 1000)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)
    logging.info('Train and Test matrix shapes: {}, {}, {}, {} '.format(X_train.shape, X_test.shape,
                                                                        Y_train.shape, Y_test.shape))

    linear_kernel_svm_classify(X_train, X_test, Y_train, Y_test)

    subgraph_kernel = get_subgraph_kernel (gensim_model, subgraph_vocab)
    deep_kernel_svm_classify (X_train, X_test, Y_train, Y_test, subgraph_kernel)



if __name__ == '__main__':
    pass
