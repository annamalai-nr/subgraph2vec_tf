import argparse,os,logging,psutil,time
from joblib import Parallel,delayed

from utils import get_files
from train_utils import train_skipgram
from classify import perform_classification
from make_subgraph2vec_corpus import dump_subgraph2vec_sentences


logger = logging.getLogger()
logger.setLevel("INFO")


def main(args):
    '''
    :param args: arguments for
    1. training the skigram model for learning subgraph representations
    2. construct the deep WL kernel using the learnt subgraph representations
    3. performing graph classification using  the WL and deep WL kernel
    :return: None
    '''
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

    t0 = time.time()
    Parallel(n_jobs=n_cpus)(delayed(dump_subgraph2vec_sentences)(f, wlk_h, label_filed_name) for f in graph_files)
    # for f in graph_files: dump_subgraph2vec_sentences (f, wlk_h, label_filed_name)
    logging.info('Dumped subgraph2vec sentences for all {} graphs in {} in {} sec'.format(len(graph_files),
                                                                                          corpus_dir, round(time.time()-t0)))

    t0 = time.time()
    embedding_fname = train_skipgram(corpus_dir, wl_extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, output_dir,valid_size)
    logging.info('Trained the skipgram model in {} sec.'.format(round(time.time()-t0, 2)))

    perform_classification (corpus_dir, wl_extn, embedding_fname, class_labels_fname)




def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser("subgraph2vec")
    # args.add_argument("--corpus", default = "wlfile/DrebinADGs_5k_malware/",
    args.add_argument("-c","--corpus", default = "../data/kdd_datasets/ptc",
                      help="Path to directory containing graph files to be used for graph classification or clustering")

    args.add_argument('-l','--class_labels_file_name', default='../data/kdd_datasets/ptc.Labels',
                      help='File name containg the name of the sample and the class labels')

    # args.add_argument("--output_dir", default = "embeddings/DrebinADGs_5k_malware/",
    args.add_argument('-o', "--output_dir", default = "../embeddings",
                      help="Path to directory for storing output embeddings")

    args.add_argument('-b',"--batch_size", default=128, type=int,
                      help="Number of samples per training batch")

    args.add_argument('-e',"--epochs", default=3, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument('-d',"--embedding_size", default=32, type=int,
                      help="Intended subgraph embedding size to be learnt")

    args.add_argument('-neg', "--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument('-lr', "--learning_rate", default=1.0, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--n_cpus", default=psutil.cpu_count(), type=int,
                      help="Maximum no. of cpu cores to be used for WL kernel feature extraction from graphs")

    args.add_argument("--wlk_h", default=2, type=int, help="Height of WL kernel (i.e., degree of rooted subgraph features to be considered for representation learning)")

    args.add_argument('-lf', '--label_filed_name', default='Label', help='Label field to be used for coloring nodes in graphs '
                                                                  'using WL kenrel')

    args.add_argument('-v',"--valid_size", default=10, type=int,
                      help="Number of samples to validate training process from time to time")

    return args.parse_args()



if __name__=="__main__":
    args = parse_args()
    main(args)
