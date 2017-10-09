from nltk.tokenize import RegexpTokenizer
import os
import numpy as np
import logging
import operator
from collections import defaultdict
from random import shuffle
from pprint import pprint
from utils import get_files


class Corpus(object):
    def __init__(self, corpus_folder=None, sample = 0.001, extn='WL2', max_files=0):
        assert corpus_folder != None, "please specify the corpus folder"
        self.sample = sample
        self.corpus_folder = corpus_folder
        self.subgraph_index = 0
        self.graph_index = 0
        self.epoch_flag = 0
        self.max_files = max_files
        self.doc_shuffle = []
        self.extn = extn




    def scan_corpus(self):
        word_to_freq_map = defaultdict(int) #word to freq map
        for fname in self.doc_list:
            target_words = (l.split()[0] for l in open(fname).xreadlines()) #just take the first word of every sentence
            for w in target_words:
                word_to_freq_map[w] += 1

        word_to_id_map = {w:i for i,w in enumerate(word_to_freq_map.iterkeys())}
        id_to_word_map = {v:k for k,v in word_to_id_map.iteritems()}

        word_to_id_map["UNK"] = len(word_to_id_map)
        id_to_word_map[len(word_to_id_map)-1] = "UNK"
        word_to_freq_map["UNK"] = 1

        self._word_to_freq_map = word_to_freq_map
        self._word_to_id_map = word_to_id_map
        self._id_to_word_map = id_to_word_map
        self._wordcount = sum(word_to_freq_map.values()) + 1 #for the word UNK
        self._vocabsize = len(self._word_to_id_map)

        sorted_word_to_freq_map = sorted(self._word_to_freq_map.items(), key=operator.itemgetter(1)) #least to most freq word
        sorted_word_ids = [self._word_to_id_map[word] for word,freq in sorted_word_to_freq_map]
        high_freq_word_ids = sorted_word_ids[-50:] #just getting top 50 most freq subgraphs
        low_freq_words_ids = sorted_word_ids[:50] #just getting top 50 least freq
        self.high_freq_word_ids = high_freq_word_ids
        self.low_freq_word_ids = low_freq_words_ids

        word_id_freq_map_as_list = [] #id of this list is the word id and value is the freq of word with corresponding word id
        for i in xrange(len(self._word_to_freq_map)):
            word_id_freq_map_as_list.append(self._word_to_freq_map[self._id_to_word_map[i]])
        self.word_id_freq_map_as_list = word_id_freq_map_as_list

        return self._word_to_id_map


    def scan_and_load_corpus(self):
        self.doc_list = get_files(self.corpus_folder, extn=self.extn, max_files=self.max_files)
        word_to_id_map = self.scan_corpus()
        # self.get_reject_prob()

        logging.info('vocabulary size: %d' % len(word_to_id_map))
        logging.info('number of documents: %d' % len(self.doc_list))
        logging.info('number of words to be trained: %d' % self._wordcount)

        self.doc_shuffle = range(len(self.doc_list))
        np.random.shuffle(self.doc_shuffle)

    # def get_reject_prob (self):
    #     threshold = self.sample * self._wordcount
    #     n_tokens = self._vocabsize
    #
    #     reject_prob = np.zeros((n_tokens,)) #numpy array: index is word id and value is its reject probability
    #     for i in xrange(n_tokens):
    #         w = self._id_to_word_map[i]
    #         freq = 1.0 * self._word_to_freq_map[w]
    #         reject_prob[i] = max(0, 1 - np.sqrt(threshold / freq)-threshold/freq)
    #         #rejectProb[i] = 0
    #   # self._reject_prob = reject_prob


    def generate_batch_from_file(self, batch_size):
        targetword_ids = []
        contextword_ids = []

        doc_name = self.doc_list[self.doc_shuffle[self.graph_index]]
        doc = open(doc_name).readlines()
        while self.subgraph_index >= len(doc):
            self.subgraph_index = 0
            self.graph_index += 1
            if self.graph_index == len(self.doc_list):
                self.graph_index = 0
                np.random.shuffle(self.doc_shuffle)
                self.epoch_flag = True
            doc_name = self.doc_list[self.doc_shuffle[self.graph_index]]
            doc = open(doc_name).readlines()

        while len(contextword_ids) < batch_size:
            line_id = self.subgraph_index

            target_and_context = doc[line_id].rstrip().split()
            context_words = target_and_context[1:]
            target_word = target_and_context[0]

            contextword_ids.extend([self._word_to_id_map[cword] for cword in context_words])
            targetword_ids.extend([self._word_to_id_map[target_word]] * len(context_words))

            # print doc_name
            # print contextword_ids
            # print targetword_ids
            # print doc[line_id].rstrip().split()[1:]
            # print doc[line_id].rstrip().split()[0]
            # raw_input()

            self.subgraph_index+=1
            while self.subgraph_index == len(doc):
                self.subgraph_index = 0
                self.graph_index += 1
                if self.graph_index == len(self.doc_list):
                    self.graph_index = 0
                    np.random.shuffle(self.doc_shuffle)
                    self.epoch_flag = True
                doc_name = self.doc_list[self.doc_shuffle[self.graph_index]]
                doc = open(doc_name).readlines()

        target_context_pairs = zip(targetword_ids, contextword_ids)
        shuffle(target_context_pairs)
        targetword_ids, contextword_ids = zip(*target_context_pairs)

        targetword_ids = np.array(targetword_ids, dtype=np.int32)
        contextword_ids = np.array(contextword_ids, dtype=np.int32)

        contextword_outputs = np.reshape(contextword_ids, [len(contextword_ids), 1])

        return targetword_ids,contextword_outputs


