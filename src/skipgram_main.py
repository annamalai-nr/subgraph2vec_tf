import argparse,os,logging,random,time
import numpy as np
from data_utils import Corpus
from utils import save_embeddings
from train_utils import Skipgram
from test import test_embeddings

logger = logging.getLogger()
logger.setLevel("INFO")
def main(args):
    corpus_dir = args.corpus
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    embedding_size = args.embedding_size
    num_negsample = args.num_negsample
    learning_rate = args.learning_rate
    valid_size = args.valid_size

    assert os.path.exists(corpus_dir), "File {} does not exist".format(corpus_dir)
    assert os.path.exists(output_dir), "Dir {} does not exist".format(output_dir)

    logging.info("Initializing...")

    corpus = Corpus(corpus_dir,max_files=0) # just load 'max_files' files from this folder
    corpus.scan_and_load_corpus()
    valid_examples = np.concatenate((np.random.choice(corpus.high_freq_word_ids, valid_size, replace=False),
                                     np.random.choice(corpus.low_freq_word_ids, valid_size, replace=False)))

    model_skipgram = Skipgram(
        doc_size = corpus._vocabsize, #for doc2vec skipgram model, the doc size should be same as word size
        vocabulary_size = corpus._vocabsize, #size of i/p and o/p layers
        learning_rate = learning_rate, #will decay over time?
        embedding_size = embedding_size, #hidden layer neurons
        num_negsample = num_negsample,
        num_steps=epochs, #no. of time the training set will be iterated through
        corpus= corpus, #data set of (target,context) tuples
        valid_dataset=valid_examples, #validation set (a small subset) of (target, context) tuples?
    )

    start_time = time.time()
    final_embeddings,final_weights = model_skipgram.train(
        corpus = corpus,
        batch_size = batch_size,
        valid_dataset=valid_examples,
    )
    train_time_tf = time.time()-start_time
    logging.info('Trained the skipgram model in {} sec.'.format(round(train_time_tf,2)))


    logging.info('Write the matrix to a word2vec format file')
    op_fname = os.path.join(output_dir,'final_embeddings.txt')
    save_embeddings(corpus, final_embeddings, embedding_size, op_fname)
    logging.info ('Completed writing the final embeddings, pls check file: {} for the same'.format(op_fname))



def parse_args():
    args = argparse.ArgumentParser("subgraph2vec")
    # args.add_argument("--corpus", default = "wlfile/DrebinADGs_5k_malware/",
    args.add_argument("--corpus", default = "/mnt/csl/OLMD/OLMD/MKLDroid/tmp/amd_dataset_graphs_wlfiles/adgs_wl2_sg2vec_root_neihood_sentences",
                      help="Path to directory containing documents files")

    args.add_argument("--max_files",
                      default=0, help="Number of files to be loaded from the corpus. 0 = load all files ")

    # args.add_argument("--output_dir", default = "embeddings/DrebinADGs_5k_malware/",
    args.add_argument("--output_dir", default = ".",
                      help="Path to directory for storing output embeddings")

    args.add_argument("--batch_size", default=128, type=int,
                      help="Number of samples per training batch")

    args.add_argument("--epochs", default=2, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument("--embedding_size", default=64, type=int,
                      help="Intended subgraph embedding size to be learnt")

    args.add_argument("--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument("--learning_rate", default=1.0, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--valid_size", default=20, type=int,
                      help="Number of samples to validate training process from time to time")

    return args.parse_args()



if __name__=="__main__":
    args = parse_args()
    main(args)
