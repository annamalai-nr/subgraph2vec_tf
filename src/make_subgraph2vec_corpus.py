import networkx as nx
import os,sys,json,logging
from time import time
from networkx.readwrite import json_graph
from joblib import Parallel,delayed
from utils import get_files

logger = logging.getLogger()
logger.setLevel("INFO")

def read_from_json_gexf(fname=None,label_field_name='APIs',conv_undir = False):
    '''
    Load the graph files (.gexf or .json only supported)
    :param fname: graph file name
    :param label_field_name: filed denoting the node label
    :param conv_undir: convert to undirected graph or not
    :return: graph in networkx format
    '''
    if not fname:
        logging.error('no valid path or file name')
        return None
    else:
        try:
            try:
                with open(fname, 'rb') as File:
                    org_dep_g = json_graph.node_link_graph(json.load(File))
            except:
                org_dep_g = nx.read_gexf (path=fname)

            g = nx.DiGraph()
            for n, d in org_dep_g.nodes_iter(data=True):
                g.add_node(n, attr_dict={'label': '-'.join(d[label_field_name].split('\n'))})
            g.add_edges_from(org_dep_g.edges_iter())
        except:
            logging.error("unable to load graph from file: {}".format(fname))
            # return 0
    logging.debug('loaded {} a graph with {} nodes and {} egdes'.format(fname, g.number_of_nodes(),g.number_of_edges()))
    if conv_undir:
        g = nx.Graph (g)
        logging.debug('converted {} as undirected graph'.format (g))
    return g


def get_graph_as_bow (g, h):
    '''
    Get subgraph2vec sentences from the goven graph
    :param g: networkx graph
    :param h: WL kernel height
    :return: sentence of the format <target> <context1> <context2> ...
    '''
    for n,d in g.nodes_iter(data=True):
        for i in xrange(0, h+1):
            center = d['relabel'][i]
            neis_labels_prev_deg = []
            neis_labels_next_deg = []

            if -1 != i-1:
                neis_labels_prev_deg = [g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g, n)] + [d['relabel'][i-1]]
            NeisLabelsSameDeg = [g.node[nei]['relabel'][i] for nei in nx.all_neighbors(g,n)]
            if not i+1 > h:
                neis_labels_next_deg = [g.node[nei]['relabel'][i+1] for nei in nx.all_neighbors(g,n)] + [d['relabel'][i+1]]


            nei_list = NeisLabelsSameDeg + neis_labels_prev_deg + neis_labels_next_deg
            try:
                nei_list.append(d['relabel'][i-1]) #prev degree subgraph from the current node
            except:
                pass
            try:
                nei_list.append(d['relabel'][i+1]) #next degree subgraph from the current node
            except:
                pass

            nei_list = ' '.join (nei_list)

            sentence = center + ' ' + nei_list
            yield sentence


def wlk_relabel(g,h):
    '''
    Perform node relabeling (coloring) according 1-d WL relabeling process (refer Shervashidze et al (2009) paper)
    :param g: networkx graph
    :param h: height of WL kernel
    :return: relabeled graph
    '''
    for n in g.nodes_iter():
        g.node[n]['relabel'] = {}

    for i in xrange(0,h+1): #xrange returns [min,max)
        for n in g.nodes_iter():
            # degree_prefix = 'D' + str(i)
            degree_prefix = ''
            if 0 == i:
                g.node[n]['relabel'][0] = degree_prefix + str(g.node[n]['label']).strip() + degree_prefix
            else:
                nei_labels = [g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g,n)]
                nei_labels.sort()
                sorted_nei_labels = (','*i).join(nei_labels)

                current_in_relabel = g.node[n]['relabel'][i-1] +'#'*i+ sorted_nei_labels
                g.node[n]['relabel'][i] = degree_prefix + current_in_relabel.strip() + degree_prefix

    return g #relabled graph


def dump_subgraph2vec_sentences (f, h, label_filed_name):
    '''
    Get WL features and make the subgraph2vec sentence of the format "<target> <context1> <context2> ..." and dump the
    same into a text file.
    :param f: gexf or json graph file name
    :param h: height of WL kernel
    :param label_filed_name: the node attribute that denotes node label
    :return: None
    '''
    if f.endswith('json'):
        opfname = f.replace('.json','.WL'+str(h))
    else:
        opfname = f.replace('.gexf', '.WL' + str(h))

    if os.path.isfile(opfname):
        logging.debug('file: {} exists, hence skipping WL feature extraction'.format(opfname))
        return

    T0 = time()
    logging.debug('processing ',f)
    g = read_from_json_gexf(f, label_filed_name)
    if not g:
        return
    g = wlk_relabel(g,h)

    if f.endswith('json'):
        opfname = f.replace('.json','.WL'+str(h))
    else:
        opfname = f.replace('.gexf', '.WL' + str(h))

    subgraph2vec_sentences = get_graph_as_bow(g, h)
    with open(opfname, 'w') as fh:
        for w in subgraph2vec_sentences:
            print >> fh, w

    logging.debug('dumped wlk file in {} sec'.format(round(time()-T0,2)))



if __name__ == '__main__':
    # if sys.argv[1] in ['-h','--help']:
    #     print 'command line args: <gexf/json graph_dir> <height of WL kernel> <num of cpu cores for multi-processing>'
    #     exit (0)

    graph_dir = "/home/annamalai/OLMD/OLMD/MKLDroid/tmp/amd_dataset_graphs_wlfiles/adgs"#folder containing the graph's gexf/json format files
    h = 2 #height of WL kernel (i.e., degree of neighbourhood to consdider)
    n_cpus = 36  # number of cpus to be used for multiprocessing
    extn = '.gexf'

    files_to_process = get_files(dirname = graph_dir, extn = extn)
    print files_to_process
    raw_input('have to procees a total of {} files with {} parallel processes... hit any key to proceed...'.
              format(len(files_to_process), n_cpus))

    Parallel(n_jobs=n_cpus)(delayed(dump_subgraph2vec_sentences)(f, h) for f in files_to_process)
