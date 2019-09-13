from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import range# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import range# pylint: disable=redefined-builtin
import tensorflow as tf
import PersonalizedEmbedding


class ProductSearchEmbedding_model(object):
	def __init__(self, data_set, window_size,
				 embed_size, max_gradient_norm, batch_size, learning_rate, L2_lambda, dynamic_weight,
				 query_weight, net_struct, similarity_func, forward_only=False, negative_sample = 5):
		"""Create the model.
	
		Args:
			vocab_size: the number of words in the corpus.
			dm_feature_len: the length of document model features (query based).
			review_size: the number of reviews in the corpus.
			user_size: the number of users in the corpus.
			product_size: the number of products in the corpus.
			embed_size: the size of each embedding
			window_size: the size of half context window
			vocab_distribute: the distribution for words, used for negative sampling
			review_distribute: the distribution for reviews, used for negative sampling
			product_distribute: the distribution for products, used for negative sampling
			max_gradient_norm: gradients will be clipped to maximally this norm.
			batch_size: the size of the batches used during training;
			the model construction is not independent of batch_size, so it cannot be
			changed after initialization.
			learning_rate: learning rate to start with.
			learning_rate_decay_factor: decay learning rate by this much when needed.
			forward_only: if set, we do not construct the backward pass in the model.
			negative_sample: the number of negative_samples for training
		"""
		self.data_set = data_set
		self.negative_sample = negative_sample
		self.embed_size = embed_size
		self.window_size = window_size
		self.max_gradient_norm = max_gradient_norm
		self.batch_size = batch_size * (self.negative_sample + 1)
		self.init_learning_rate = learning_rate
		self.L2_lambda = L2_lambda
		self.dynamic_weight = dynamic_weight
		self.net_struct = net_struct
		self.similarity_func = similarity_func
		self.global_step = tf.Variable(0, trainable=False)
		if query_weight >= 0:
			self.Wu = tf.Variable(query_weight, name="user_weight", dtype=tf.float32, trainable=False)
		else:
			self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))
		self.query_max_length = data_set.query_max_length
		self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")
		self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		init_width = 0.5 / self.embed_size

		def entity(name, vocab):
			print('%s size %s' % (name,str(len(vocab))))
			return {
				'name' : name,
				'vocab' : vocab,
				'size' : len(vocab),
				'embedding' :tf.Variable( tf.random_uniform(									
							[len(vocab) + 1, self.embed_size], -init_width, init_width),				
							name="%s_emb"%name)	
			}
		self.entity_dict = {
			'user' : entity('user', data_set.user_ids),
			'product' : entity('product', data_set.product_ids),
			'word' : entity('word', data_set.words),
			'related_product' : entity('related_product',data_set.related_product_ids),
			'brand' : entity('brand', data_set.brand_ids),
			'categories' : entity('categories', data_set.category_ids),
		}

		self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
		self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")
		def relation(name, distribute, tail_entity):
			print('%s size %s' % (name, str(len(distribute))))
			return {
				'name' : name,
				'tail_entity' : tail_entity,
				'distribute' : distribute,
				'idxs' : tf.placeholder(tf.int64, shape=[None], name="%s_idxs"%name),
				'weight' : 	tf.placeholder(tf.float32, shape=[None], name="%s_weight"%name),
				'embedding' : tf.Variable( tf.random_uniform(									
								[self.embed_size], -init_width, init_width),				
								name="%s_emb"%name),
				'bias' : tf.Variable(tf.zeros([len(distribute)+1]), name="%s_b"%name)	
			}
		self.relation_dict = {
			'word' : relation('write', data_set.vocab_distribute, 'word'),
			'also_bought' : relation('also_bought', data_set.knowledge['also_bought']['distribute'], 'related_product'),
			'also_viewed' : relation('also_viewed', data_set.knowledge['also_viewed']['distribute'], 'related_product'),
			'bought_together' : relation('bought_together', data_set.knowledge['bought_together']['distribute'], 'related_product'),
			'brand' : relation('is_brand', data_set.knowledge['brand']['distribute'], 'brand'),
			'categories' : relation('is_category', data_set.knowledge['categories']['distribute'], 'categories')
		}

		# Select which relation to use
		self.use_relation_dict = {
			'also_bought' : False,
			'also_viewed' : False,
			'bought_together' : False,
			'brand' : False,
			'categories' : False,
		}
		if 'none' in self.net_struct:
			print('Use no relation')
		else:
			need_relation_list = []
			for key in self.use_relation_dict:
				if key in self.net_struct:
					self.use_relation_dict[key] = True
					need_relation_list.append(key)
			if len(need_relation_list) > 0:
				print('Use relation ' + ' '.join(need_relation_list))
			else:
				print('Use all relation')
				for key in self.use_relation_dict:
					self.use_relation_dict[key] = True

		print('L2 lambda ' + str(self.L2_lambda))

		# Training losses.
		self.loss = PersonalizedEmbedding.build_graph_and_loss(self)

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.gradients = tf.gradients(self.loss, params)
			
			self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																	 self.max_gradient_norm)
			self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
											 global_step=self.global_step)
			
			#self.updates = opt.apply_gradients(zip(self.gradients, params),
			#								 global_step=self.global_step)
		else:
			# Compute all information based on user+query
			# user + query -> product
			self.product_scores, uq_vec = PersonalizedEmbedding.get_product_scores(self, self.user_idxs, self.query_word_idxs)
			# user + write -> word
			user_vec = tf.nn.embedding_lookup(self.entity_dict['user']['embedding'], self.user_idxs)
			self.uw_scores, uw_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, user_vec, 'word', 'word')
			# user + query + write -> word
			self.uqw_scores, uqw_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, 'word', 'word')
			# Compute all information based on user + query
			self.uq_entity_list = [
				('search', 'product', self.product_scores), ('write', 'word', self.uw_scores),
			]
			for key in self.use_relation_dict:
				if self.use_relation_dict[key]:
					tail_entity = self.relation_dict[key]['tail_entity']
					scores, vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, key, tail_entity)
					self.uq_entity_list.append((key, tail_entity, scores))
			'''
			# user + query + also_bought -> product
			self.uqab_scores, uqab_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, 'also_bought', 'related_product')
			# user + query + also_viewed -> product
			self.uqav_scores, uqav_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, 'also_viewed', 'related_product')
			# user + query + bought_together -> product
			self.uqbt_scores, uqbt_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, 'bought_together', 'related_product')
			# user + query + is_brand -> brand
			self.uqib_scores, uqib_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, 'brand', 'brand')
			# user + query + is_category -> category
			self.uqic_scores, uqic_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, uq_vec, 'categories', 'categories')
			'''
			
			# Compute all information based on product
			# product + write -> word
			p_vec = tf.nn.embedding_lookup(self.entity_dict['product']['embedding'], self.product_idxs)
			self.pw_scores, pw_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, 'word', 'word')
			self.p_entity_list = [
				('write', 'word', self.pw_scores),
			]
			for key in self.use_relation_dict:
				if self.use_relation_dict[key]:
					tail_entity = self.relation_dict[key]['tail_entity']
					scores, vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, key, tail_entity)
					self.p_entity_list.append((key, tail_entity, scores))
			'''
			# product + also_bought -> product
			self.pab_scores, pab_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, 'also_bought', 'related_product')
			# product + also_viewed -> product
			self.pav_scores, pav_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, 'also_viewed', 'related_product')
			# product + bought_together -> product
			self.pbt_scores, pbt_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, 'bought_together', 'related_product')
			# product + is_brand -> brand
			self.pib_scores, pib_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, 'brand', 'brand')
			# product + is_category -> category
			self.pic_scores, pic_vec = PersonalizedEmbedding.get_relation_scores(self, 0.5, p_vec, 'categories', 'categories')
			'''
			
	
		self.saver = tf.train.Saver(tf.global_variables())
	
	def step(self, session, input_feed, forward_only, test_mode = 'product_scores'):
		"""Run a step of the model feeding the given inputs.
	
		Args:
			session: tensorflow session to use.
			learning_rate: the learning rate of current step
			user_idxs: A numpy [1] float vector.
			product_idxs: A numpy [1] float vector.
			review_idxs: A numpy [1] float vector.
			word_idxs: A numpy [None] float vector.
			context_idxs: list of numpy [None] float vectors.
			forward_only: whether to do the update step or only forward.
	
		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.
	
		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
			target_weights disagrees with bucket size for the specified bucket_id.
		"""
	
		# Output feed: depends on whether we do a backward step or not.
		entity_list = None
		if not forward_only:
			output_feed = [self.updates,	# Update Op that does SGD.
						 #self.norm,	# Gradient norm.
						 self.loss]	# Loss for this batch.
		else:
			if test_mode == 'output_embedding':
				self.embed_output_keys = []
				output_feed = []
				for key in self.entity_dict:
					self.embed_output_keys.append(key)
					output_feed.append(self.entity_dict[key]['embedding'])
				for key in self.relation_dict:
					self.embed_output_keys.append(key + '_embed')
					output_feed.append(self.relation_dict[key]['embedding'])
				for key in self.relation_dict:
					self.embed_output_keys.append(key + '_bias')
					output_feed.append(self.relation_dict[key]['bias'])
				self.embed_output_keys.append('Wu')
				output_feed.append(self.Wu)
			elif 'explain' in test_mode:
				if test_mode == 'explain_user_query':
					entity_list = self.uq_entity_list
				elif test_mode == 'explain_product':
					entity_list = self.p_entity_list
				output_feed = [scores for _, _, scores in entity_list]
			else:
				output_feed = [self.product_scores] #negative instance output
	
		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], None	# loss, no outputs, Gradient norm.
		else:
			if test_mode == 'output_embedding':
				return outputs, self.embed_output_keys
			elif 'explain' in test_mode:
				return [(entity_list[i][0], entity_list[i][1], outputs[i]) for i in range(len(entity_list))], None
			else:
				return outputs[0], None	# product scores to input user

	def setup_data_set(self, data_set, words_to_train):
		self.data_set = data_set
		self.words_to_train = words_to_train
		self.finished_word_num = 0
		#if self.net_struct == 'hdc':
		#	self.need_context = True

	def intialize_epoch(self, training_seq):
		self.train_seq = training_seq
		self.review_size = len(self.train_seq)
		self.cur_review_i = 0
		self.cur_word_i = 0

	def get_train_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		knowledge_idxs_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		knowledge_weight_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		query_word_idxs = []
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		review_idx, user_idx, product_idx, query_idx = None, None, None, None
		text_list, text_length, product_knowledge = None, None, None

		# Start from a new review
		review_idx = self.train_seq[self.cur_review_i]
		user_idx = self.data_set.review_info[review_idx][0]
		product_idx = self.data_set.review_info[review_idx][1]
		query_idx = random.choice(self.data_set.product_query_idx[product_idx])
		text_list = self.data_set.review_text[review_idx]
		text_length = len(text_list)
		# add knowledge
		product_knowledge = {key : self.data_set.knowledge[key]['data'][product_idx] for key in knowledge_idxs_dict}
		
		while len(word_idxs) < self.batch_size:
			#print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
			#if sample this word
			if self.data_set.sub_sampling_rate == None or random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
				user_idxs.append(user_idx)
				product_idxs.append(product_idx)
				query_word_idxs.append(self.data_set.query_words[query_idx])
				review_idxs.append(review_idx)
				word_idxs.append(text_list[self.cur_word_i])
				# Add knowledge
				for key in product_knowledge:
					if len(product_knowledge[key]) < 1:
						knowledge_idxs_dict[key].append(self.entity_dict[self.relation_dict[key]['tail_entity']]['size'])
						knowledge_weight_dict[key].append(0.0)
					else:
						knowledge_idxs_dict[key].append(random.choice(product_knowledge[key]))
						knowledge_weight_dict[key].append(1.0)

			#move to the next
			self.cur_word_i += 1
			self.finished_word_num += 1
			if self.cur_word_i == text_length:
				self.cur_review_i += 1
				if self.cur_review_i == self.review_size:
					break
				self.cur_word_i = 0
				# Start from a new review
				review_idx = self.train_seq[self.cur_review_i]
				user_idx = self.data_set.review_info[review_idx][0]
				product_idx = self.data_set.review_info[review_idx][1]
				query_idx = random.choice(self.data_set.product_query_idx[product_idx])
				text_list = self.data_set.review_text[review_idx]
				text_length = len(text_list)
				# add knowledge
				product_knowledge = {key : self.data_set.knowledge[key]['data'][product_idx] for key in knowledge_idxs_dict}


		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.user_idxs.name] = user_idxs
		input_feed[self.product_idxs.name] = product_idxs
		input_feed[self.query_word_idxs.name] = query_word_idxs
		input_feed[self.relation_dict['word']['idxs'].name] = word_idxs
		input_feed[self.relation_dict['word']['weight'].name] = [1.0 for _ in range(len(word_idxs))]
		for key in knowledge_idxs_dict:
			input_feed[self.relation_dict[key]['idxs'].name] = knowledge_idxs_dict[key]
			input_feed[self.relation_dict[key]['weight'].name] = knowledge_weight_dict[key]

		has_next = False if self.cur_review_i == self.review_size else True
		return input_feed, has_next

	def prepare_test_epoch(self):
		self.test_user_query_set = set()
		self.test_seq = []
		for review_idx in range(len(self.data_set.review_info)):
			user_idx = self.data_set.review_info[review_idx][0]
			product_idx = self.data_set.review_info[review_idx][1]	
			for query_idx in self.data_set.product_query_idx[product_idx]:
				if (user_idx, query_idx) not in self.test_user_query_set:
					self.test_user_query_set.add((user_idx, query_idx))
					self.test_seq.append((user_idx, product_idx, query_idx, review_idx))
		self.cur_uqr_i = 0

	def get_test_batch(self):
		user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
		knowledge_idxs_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		knowledge_weight_dict = {
			'also_bought' : [],
			'also_viewed' : [],
			'bought_together' : [],
			'brand' : [],
			'categories' : []
		}
		query_word_idxs = []
		learning_rate = self.init_learning_rate * max(0.0001, 
									1.0 - self.finished_word_num / self.words_to_train)
		start_i = self.cur_uqr_i
		user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

		while len(user_idxs) < self.batch_size:
			text_list = self.data_set.review_text[review_idx]
			user_idxs.append(user_idx)
			product_idxs.append(product_idx)
			query_word_idxs.append(self.data_set.query_words[query_idx])
			review_idxs.append(review_idx)
			word_idxs.append(text_list[0])
			# Add knowledge
			for key in knowledge_idxs_dict:
				knowledge_idxs_dict[key].append(self.entity_dict[self.relation_dict[key]['tail_entity']]['size'])
				knowledge_weight_dict[key].append(0.0)
			
			#move to the next review
			self.cur_uqr_i += 1
			if self.cur_uqr_i == len(self.test_seq):
				break
			user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		input_feed[self.learning_rate.name] = learning_rate
		input_feed[self.user_idxs.name] = user_idxs
		input_feed[self.product_idxs.name] = product_idxs
		input_feed[self.query_word_idxs.name] = query_word_idxs
		input_feed[self.relation_dict['word']['idxs'].name] = word_idxs
		input_feed[self.relation_dict['word']['weight'].name] = [0.0 for _ in range(len(word_idxs))]
		for key in knowledge_idxs_dict:
			input_feed[self.relation_dict[key]['idxs'].name] = knowledge_idxs_dict[key]
			input_feed[self.relation_dict[key]['weight'].name] = knowledge_weight_dict[key]

		has_next = False if self.cur_uqr_i == len(self.test_seq) else True
		return input_feed, has_next, self.test_seq[start_i:self.cur_uqr_i]
