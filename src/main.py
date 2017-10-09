import argparse,os,logging,random,time,psutil
import numpy as np
from data_utils import Corpus
from utils import save_embeddings
from train_utils import Skipgram
from test import test_embeddings
from classify import perform_classification

from make_subgraph2vec_corpus import dump_subgraph2vec_sentences
from utils import get_files, get_class_labels

from joblib import Parallel,delayed
import time

logger = logging.getLogger()
logger.setLevel("INFO")

def train_skipgram (corpus_dir, extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, output_dir,valid_size):

    op_fname = os.path.join(output_dir, 'final_embeddings.txt')
    if os.path.isfile(op_fname):
        logging.info('The embedding file: {} is already present, hence NOT training skipgram model '
                     'for subgraph vectors'.format(op_fname))
        return op_fname

    logging.info("Initializing SKIPGRAM...")
    corpus = Corpus(corpus_dir, extn = extn, max_files=0)  # just load 'max_files' files from this folder
    corpus.scan_and_load_corpus()
    valid_examples = np.concatenate((np.random.choice(corpus.high_freq_word_ids, valid_size, replace=False),
                                     np.random.choice(corpus.low_freq_word_ids, valid_size, replace=False)))

    model_skipgram = Skipgram(
        doc_size=corpus._vocabsize,  # for doc2vec skipgram model, the doc size should be same as word size
        vocabulary_size=corpus._vocabsize,  # size of i/p and o/p layers
        learning_rate=learning_rate,  # will decay over time?
        embedding_size=embedding_size,  # hidden layer neurons
        num_negsample=num_negsample,
        num_steps=epochs,  # no. of time the training set will be iterated through
        corpus=corpus,  # data set of (target,context) tuples
        valid_dataset=valid_examples,  # validation set (a small subset) of (target, context) tuples?
    )

    final_embeddings, final_weights = model_skipgram.train(
        corpus=corpus,
        batch_size=batch_size,
        valid_dataset=valid_examples,
    )


    logging.info('Write the matrix to a word2vec format file')
    save_embeddings(corpus, final_embeddings, embedding_size, op_fname)
    logging.info('Completed writing the final embeddings, pls check file: {} for the same'.format(op_fname))
    return op_fname

def main(args):
    corpus_dir = args.corpus
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    embedding_size = args.embedding_size
    num_negsample = args.num_negsample
    learning_rate = args.learning_rate
    valid_size = args.valid_size
    n_cpus = args.n_cpus
    wlk_h = args.wlk_h
    label_filed_name = args.label_filed_name
    class_labels_fname = args.class_labels_file_name

    wl_extn = 'WL'+str(wlk_h)

    assert os.path.exists(corpus_dir), "File {} does not exist".format(corpus_dir)
    assert os.path.exists(output_dir), "Dir {} does not exist".format(output_dir)

    graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)
    logging.info('Loaded {} graph file names form {}'.format(len(graph_files),corpus_dir))

    class_labels = get_class_labels(graph_files, class_labels_fname)

    t0 = time.time()
    Parallel(n_jobs=n_cpus)(delayed(dump_subgraph2vec_sentences)(f, wlk_h, label_filed_name) for f in graph_files)
    logging.info('Dumped subgraph2vec sentences for all {} graphs in {} in {} sec'.format(len(graph_files),
                                                                                          corpus_dir, round(time.time()-t0)))

    t0 = time.time()
    embedding_fname = train_skipgram(corpus_dir, wl_extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, output_dir,valid_size)
    logging.info('Trained the skipgram model in {} sec.'.format(round(time.time()-t0, 2)))

    perform_classification (corpus_dir, wl_extn, embedding_fname, class_labels)




def parse_args():
    args = argparse.ArgumentParser("subgraph2vec")
    # args.add_argument("--corpus", default = "wlfile/DrebinADGs_5k_malware/",
    args.add_argument("--corpus", default = "/mnt/anna_laptop/subgraph2vec/kdd_datasets/dir_graphs/ptc",
                      help="Path to directory containing graph files to be used for graph classification or clustering")

    args.add_argument('--class_labels_file_name', default='/mnt/anna_laptop/subgraph2vec/kdd_datasets/ptc.Labels',
                      help='File name containg the name of the sample and the class labels')

    args.add_argument("--max_files",type=int,
                      default=20, help="Number of files to be loaded from the corpus. 0 = load all files ")

    # args.add_argument("--output_dir", default = "embeddings/DrebinADGs_5k_malware/",
    args.add_argument("--output_dir", default = ".",
                      help="Path to directory for storing output embeddings")

    args.add_argument("--batch_size", default=128, type=int,
                      help="Number of samples per training batch")

    args.add_argument("--epochs", default=10, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument("--embedding_size", default=32, type=int,
                      help="Intended subgraph embedding size to be learnt")

    args.add_argument("--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument("--learning_rate", default=1.0, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--valid_size", default=10, type=int,
                      help="Number of samples to validate training process from time to time")

    # args.add_argument("--n_cpus", default=psutil.cpu_count(), type=int,
    args.add_argument("--n_cpus", default=20, type=int,
                      help="Maximum no. of cpu cores to be used for WL kernel feature extraction from graphs")

    args.add_argument("--wlk_h", default=2, type=int, help="Height of WL kernel")

    args.add_argument('--label_filed_name', default='Label', help='Label field to be used for coloring nodes in graphs '
                                                                  'using WL kenrel')
    return args.parse_args()



if __name__=="__main__":
    args = parse_args()
    main(args)
