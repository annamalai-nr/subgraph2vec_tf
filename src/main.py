import argparse,os,logging,random,time
import numpy as np
from data_utils import Corpus,save_embeddings
from train_utils import Skipgram
from gensim.models import Word2Vec
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

    corpus = Corpus(corpus_dir)
    corpus.all_sentences()
    valid_examples = np.concatenate((np.random.choice(corpus.high_freq_word_ids, valid_size, replace=False),
                                     np.random.choice(corpus.low_freq_word_ids, valid_size, replace=False)))

    model_skipgram = Skipgram(
        doc_size = corpus._vocabsize,
        vocabulary_size = corpus._vocabsize,
        learning_rate = learning_rate,
        embedding_size = embedding_size,
        num_negsample = num_negsample,
        num_steps=epochs,
        corpus= corpus,
        valid_dataset=valid_examples,
    )

    start_time = time.time()
    final_embeddings,final_weights = model_skipgram.train(
        corpus = corpus,
        batch_size = batch_size,
        valid_dataset=valid_examples,
    )
    time_tf = time.time()-start_time


    logging.info('Write the matrix to a word2vec format file')

    save_embeddings(corpus, final_embeddings, embedding_size, os.path.join(output_dir,'final_embeddings.txt'))



def parse_args():
    args = argparse.ArgumentParser("subgraph2vec")
    # args.add_argument("--corpus", default = "wlfile/DrebinADGs_5k_malware/",
    args.add_argument("--corpus", default = "/mnt/csl/OLMD/OLMD/MKLDroid/tmp/amd_dataset_graphs_wlfiles/adgs_wl2_sg2vec_root_neihood_sentences",
                      help="Absolute path to directory containing documents files")

    # args.add_argument("--output_dir", default = "embeddings/DrebinADGs_5k_malware/",
    args.add_argument("--output_dir", default = "embeddings/anna_DrebinADGs_5k_malware/",
                      help="Absolute path to directory for storing output data")

    args.add_argument("--batch_size", default=256, type=int,
                      help="Number of samples per training batch")

    args.add_argument("--epochs", default=2, type=int,
                      help="Number of rounds of documents")

    args.add_argument("--embedding_size", default=64, type=int,
                      help="The size of word vector representation")

    args.add_argument("--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument("--learning_rate", default=1.0, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--valid_size", default=20, type=int,
                      help="Number of val_data")

    return args.parse_args()



if __name__=="__main__":
    args = parse_args()
    main(args)
