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

if __name__ == '__main__':
    print 'nothing to do'
