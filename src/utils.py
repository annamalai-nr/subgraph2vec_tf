import os


def get_files(dirname, extn, max_files=0):
    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                all_files.append(os.path.join(root, f))

    all_files = list(set(all_files))
    all_files.sort()
    if max_files:
        return all_files[:max_files]
    else:
        return all_files


def save_embeddings(corpus, final_embeddings, embedding_size, opfname):
    lines_to_write = []
    lines_to_write.append(str(corpus._vocabsize) + ' ' + str(embedding_size))
    lines_to_write.extend([corpus._id_to_word_map[i] + ' ' +
                           ' '.join(final_embeddings[i].astype('str').tolist()) for i in xrange(corpus._vocabsize)])
    with open(opfname, 'w') as fh:
        for l in lines_to_write:
            print >>fh, l


def get_class_labels(graph_files, class_labels_fname):
    graph_to_class_label_map = {l.split()[0].split('.')[0]: int(l.split()[1].strip()) for l in open (class_labels_fname)}
    # print graph_files
    # raw_input()
    labels = [graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files]
    return labels

if __name__ == '__main__':
    print 'nothing to do'
