import tensorflow as tf
import math,logging
from pprint import  pprint

class skipgram(object):
    '''
    skipgram model - refer Mikolov et al (2013)
    '''

    def trainer_initial(self):
        graph = tf.Graph()
        with graph.as_default():
                batch_inputs = tf.placeholder(tf.int32, shape=([None, ]))
                batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))
                val_dataset = tf.constant(self.valid_data, dtype=tf.int32)

                doc_embeddings = tf.Variable(
                        tf.random_uniform([self.doc_size, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size))

                batch_doc_embeddings = tf.nn.embedding_lookup(doc_embeddings, batch_inputs)

                weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
                biases = tf.Variable(tf.zeros(self.vocabulary_size))

                #negative sampling part
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=weights,
                                   biases=biases,
                                   labels=batch_labels,
                                   inputs=batch_doc_embeddings,
                                   num_sampled=self.num_negsample,
                                   num_classes=self.vocabulary_size,
                                   sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                                       true_classes=batch_labels,
                                       num_true=1,
                                       num_sampled=self.num_negsample,
                                       unique=True,
                                       range_max=self.vocabulary_size,
                                       distortion=0.75,
                                       unigrams=self.corpus.word_id_freq_map_as_list)#word_id_freq_map_as_list is the
                                   # frequency of each word in vocabulary
                                   ))

                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                           global_step, 100000, 0.96, staircase=True) #linear decay over time
                learning_rate = tf.maximum(learning_rate,0.001) #cannot go below 0.001 to ensure at least a minimal learning
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

                norm = tf.sqrt(tf.reduce_mean(tf.square(doc_embeddings), 1, keep_dims=True))
                normalized_embeddings = doc_embeddings/norm
                norm = tf.sqrt(tf.reduce_mean(tf.square(weights), 1, keep_dims=True))
                normalized_weights = weights/norm
                valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, val_dataset)

                similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        return graph,batch_inputs, batch_labels, normalized_embeddings, loss, optimizer, normalized_weights,similarity


    def __init__(self,doc_size,vocabulary_size,learning_rate,embedding_size,
                 num_negsample,num_steps,corpus,valid_dataset):
        self.doc_size = doc_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_negsample = num_negsample
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.valid_data = valid_dataset
        self.corpus = corpus
        self.graph, self.batch_inputs, self.batch_labels,self.normalized_embeddings,\
        self.loss,self.optimizer,self.normalized_weights,self.similarity = self.trainer_initial()



    def train(self,corpus,batch_size,valid_dataset):
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)

            loss = 0

            for i in xrange(self.num_steps):

                step = 0
                while corpus.epoch_flag == False:
                    batch_data, batch_labels = corpus.generate_batch_from_file(batch_size)# get (target,context) wordid tuples

                    feed_dict = {self.batch_inputs:batch_data,self.batch_labels:batch_labels}
                    _,loss_val = sess.run([self.optimizer,self.loss],feed_dict=feed_dict)

                    loss += loss_val

                    if step % 10000 == 0:
                        if step > 0:
                            average_loss = loss/step
                            logging.info( 'Epoch: %d : Average loss for step: %d : %f'%(i,step,average_loss))
                    step += 1

                corpus.epoch_flag = False
                logging.info('#########################   Epoch: %d :  %f   #####################' % (i, loss/step))
                loss = 0
                sim = self.similarity.eval()
                for j in xrange(len(valid_dataset)):
                    top_k = 10  # Number of nearest neighbours
                    nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                    log_str = '***** Nearest to %s ***** \n' % corpus._id_to_word_map[valid_dataset[j]]
                    close_words = [corpus._id_to_word_map[nearest[k]] for k in xrange(top_k)]
                    log_str += '\n'.join(close_words)
                    log_str += '\n*****'
                    logging.info(log_str)

            #done with training
            final_embeddings = self.normalized_embeddings.eval()
            final_weights = self.normalized_weights.eval()
        return final_embeddings,final_weights
