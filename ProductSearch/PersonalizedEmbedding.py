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

																				
def get_product_scores(model, user_idxs, query_word_idx, product_idxs = None, scope = None):
	with variable_scope.variable_scope(scope or "embedding_graph"):										
		# get user embedding [None, embed_size]										
		user_vec = tf.nn.embedding_lookup(model.entity_dict['user']['embedding'], user_idxs)
		# get query embedding [None, embed_size]
		if model.dynamic_weight >= 0.0:
			print('Query as a dynamic relationship')
			query_vec, query_embs = get_query_embedding(model, query_word_idx, model.entity_dict['word']['embedding'], True)
		else:
			print('Query as a static relationship')
			query_vec = model.query_static_vec

		# get candidate product embedding [None, embed_size]										
		product_vec = None
		product_bias = None
		if product_idxs != None:										
			product_vec = tf.nn.embedding_lookup(model.entity_dict['product']['embedding'], product_idxs)									
			product_bias = tf.nn.embedding_lookup(model.product_bias, product_idxs)
		else:										
			product_vec = model.entity_dict['product']['embedding']
			product_bias = model.product_bias								
												
		print('Similarity Function : ' + model.similarity_func)										
		#example_vec = (1.0 - model.Wu) * user_vec + model.Wu * query_vec
		example_vec = user_vec + query_vec
		
		if model.similarity_func == 'product':										
			return tf.matmul(example_vec, product_vec, transpose_b=True), example_vec
		elif model.similarity_func == 'bias_product':
			return tf.matmul(example_vec, product_vec, transpose_b=True) + product_bias, example_vec
		else:										
			norm_vec = example_vec / tf.sqrt(tf.reduce_sum(tf.square(example_vec), 1, keep_dims=True))
			product_vec = product_vec / tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))									
			return tf.matmul(norm_vec, product_vec, transpose_b=True), example_vec

def get_relation_scores(model, add_weight, head_vec, relation_name, tail_name, tail_idxs = None, scope = None):
	with variable_scope.variable_scope(scope or "embedding_graph"):										
		# get relation embedding [None, embed_size]
		relation_vec = model.relation_dict[relation_name]['embedding']
		relation_bias = model.relation_dict[relation_name]['bias']

		# get candidate product embedding [None, embed_size]										
		tail_vec = None
		tail_bias = None
		if tail_idxs != None:										
			tail_vec = tf.nn.embedding_lookup(model.entity_dict[tail_name]['embedding'], tail_idxs)									
			tail_bias = tf.nn.embedding_lookup(relation_bias, tail_idxs)
		else:										
			tail_vec = model.entity_dict[tail_name]['embedding']
			tail_bias = relation_bias								
								
		#example_vec = (1.0 - add_weight) * head_vec + add_weight * relation_vec
		example_vec = head_vec + relation_vec
		
		if model.similarity_func == 'product':										
			return tf.matmul(example_vec, tail_vec, transpose_b=True), example_vec
		elif model.similarity_func == 'bias_product':
			return tf.matmul(example_vec, tail_vec, transpose_b=True) + tail_bias, example_vec
		else:										
			norm_vec = example_vec / tf.sqrt(tf.reduce_sum(tf.square(example_vec), 1, keep_dims=True))
			tail_vec = tail_vec / tf.sqrt(tf.reduce_sum(tf.square(tail_vec), 1, keep_dims=True))									
			return tf.matmul(norm_vec, tail_vec, transpose_b=True), example_vec


def get_fs_from_words(model, word_vecs, reuse, scope=None):
	with variable_scope.variable_scope(scope or 'f_s_abstraction',
										 reuse=reuse):
		# get mean word vectors
		mean_word_vec = tf.reduce_mean(word_vecs, 1)
		# get f(s)
		f_W = variable_scope.get_variable("f_W", [model.embed_size, model.embed_size])
		f_b = variable_scope.get_variable("f_b", [model.embed_size])
		f_s = tf.tanh(tf.nn.bias_add(tf.matmul(mean_word_vec, f_W), f_b))
		return f_s, [f_W, word_vecs]

def get_addition_from_words(model, word_vecs, reuse, scope=None):
	with variable_scope.variable_scope(scope or 'addition_abstraction',
										 reuse=reuse):
		# get mean word vectors
		mean_word_vec = tf.reduce_mean(word_vecs, 1)
		return mean_word_vec, [word_vecs]

def get_RNN_from_words(model, word_vecs, reuse, scope=None):
	with variable_scope.variable_scope(scope or 'RNN_abstraction',
										 reuse=reuse):
		cell = tf.contrib.rnn.GRUCell(model.embed_size)
		encoder_outputs, encoder_state = tf.nn.static_rnn(cell, tf.unstack(word_vecs, axis=1), dtype=dtypes.float32)
		return encoder_state, [word_vecs]

def get_attention_from_words(model, word_vecs, reuse, scope=None):
	with variable_scope.variable_scope(scope or 'attention_abstraction',
										 reuse=reuse,
										 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)):
		# build mask
		word_idxs = model.relation_dict['word']['idxs']
		mask = tf.maximum(tf.cast(word_idxs, tf.float32) + 1.0, 1.0) # [batch,query_max_length]
		# softmax weight
		#print(word_idxs.get_shape())
		gate_W = variable_scope.get_variable("gate_W", [model.embed_size])
		#print(tf.reduce_sum(word_vecs * gate_W,2).get_shape())
		word_weight = tf.exp(tf.reduce_sum(word_vecs * gate_W,2)) * mask 
		word_weight = word_weight / tf.reduce_sum(word_weight,1,keep_dims=True) 
		# weigted sum
		att_word_vec = tf.reduce_sum(word_vecs * tf.expand_dims(word_weight,2),1)
		return att_word_vec, [word_vecs]

def get_query_embedding(model, word_idxs, word_emb, reuse, scope = None):
	word_vecs = tf.nn.embedding_lookup(word_emb, word_idxs)
	if 'mean' in model.net_struct: # mean vector
		print('Query model: mean')
		return get_addition_from_words(model, word_vecs, reuse, scope)
	elif 'fs' in model.net_struct: # LSE f(s)
		print('Query model: LSE f(s)')
		return get_fs_from_words(model, word_vecs, reuse, scope)
	elif 'RNN' in model.net_struct: # RNN
		print('Query model: RNN')
		return get_RNN_from_words(model, word_vecs, reuse, scope)
	else:
		print('Query model: Attention')
		return get_attention_from_words(model, word_vecs, reuse, scope)


def build_graph_and_loss(model, scope = None):											
	with variable_scope.variable_scope(scope or "embedding_graph"):	
		loss = None
		regularization_terms = []
		batch_size = array_ops.shape(model.user_idxs)[0]#get batch_size	
		# user + query -> product
		query_vec = None
		if model.dynamic_weight >= 0.0:
			print('Treat query as a dynamic relationship')
			query_vec, qw_embs = get_query_embedding(model, model.query_word_idxs, model.entity_dict['word']['embedding'], None) # get query vector
			regularization_terms.extend(qw_embs)
		else:
			print('Treat query as a static relationship')
			init_width = 0.5 / model.embed_size
			model.query_static_vec = tf.Variable(tf.random_uniform([model.embed_size], -init_width, init_width),				
								name="query_emb")
			query_vec = model.query_static_vec
			regularization_terms.extend([query_vec])
		model.product_bias = tf.Variable(tf.zeros([model.entity_dict['product']['size'] + 1]), name="product_b")
		uqr_loss, uqr_embs = pair_search_loss(model, model.Wu, query_vec, model.user_idxs, # product prediction loss
							model.entity_dict['user']['embedding'], model.product_idxs, 
							model.entity_dict['product']['embedding'], model.product_bias, 
							len(model.entity_dict['product']['vocab']), model.data_set.product_distribute) 			
		regularization_terms.extend(uqr_embs)
		#uqr_loss = tf.Print(uqr_loss, [uqr_loss], 'this is uqr', summarize=5)
		dynamic_loss = tf.reduce_sum(uqr_loss)

		# user + write -> word
		uw_loss, uw_embs = relation_nce_loss(model, 0.5, model.user_idxs, 'user', 'word', 'word')
		regularization_terms.extend(uw_embs)
		#uw_loss = tf.Print(uw_loss, [uw_loss], 'this is uw', summarize=5)

		static_loss = uw_loss

		# product + write -> word
		pw_loss, pw_embs = relation_nce_loss(model, 0.5, model.product_idxs, 'product', 'word', 'word')
		regularization_terms.extend(pw_embs)
		#pw_loss = tf.Print(pw_loss, [pw_loss], 'this is pw', summarize=5)
		static_loss += pw_loss

		# product + also_bought -> product
		if model.use_relation_dict['also_bought']:
			pab_loss, pab_embs = relation_nce_loss(model, 0.5, model.product_idxs, 'product', 'also_bought', 'related_product')
			regularization_terms.extend(pab_embs)
			#pab_loss = tf.Print(pab_loss, [pab_loss], 'this is pab', summarize=5)
			static_loss += pab_loss

		# product + also_viewed -> product
		if model.use_relation_dict['also_viewed']:
			pav_loss, pav_embs = relation_nce_loss(model, 0.5, model.product_idxs, 'product', 'also_viewed', 'related_product')
			regularization_terms.extend(pav_embs)
			#pav_loss = tf.Print(pav_loss, [pav_loss], 'this is pav', summarize=5)
			static_loss += pav_loss

		# product + bought_together -> product
		if model.use_relation_dict['bought_together']:
			pbt_loss, pbt_embs = relation_nce_loss(model, 0.5, model.product_idxs, 'product', 'bought_together', 'related_product')
			regularization_terms.extend(pbt_embs)
			#pbt_loss = tf.Print(pbt_loss, [pbt_loss], 'this is pbt', summarize=5)
			static_loss += pbt_loss

		# product + is_brand -> brand
		if model.use_relation_dict['brand']:
			pib_loss, pib_embs = relation_nce_loss(model, 0.5, model.product_idxs, 'product', 'brand', 'brand')
			regularization_terms.extend(pib_embs)
			#pib_loss = tf.Print(pib_loss, [pib_loss], 'this is pib', summarize=5)
			static_loss += pib_loss

		# product + is_category -> categories
		if model.use_relation_dict['categories']:
			pic_loss, pic_embs = relation_nce_loss(model, 0.5, model.product_idxs, 'product', 'categories', 'categories')
			regularization_terms.extend(pic_embs)
			#pic_loss = tf.Print(pic_loss, [pw_loss], 'this is pic', summarize=5)
			static_loss += pic_loss

		# merge dynamic loss and static loss
		loss = None
		if model.dynamic_weight >= 0.0:
			print('Dynamic relation weight %.2f' % model.dynamic_weight)
			loss = 2 * (model.dynamic_weight * dynamic_loss + (1-model.dynamic_weight) * static_loss)
		else:
			# consider query as a static relation
			loss = dynamic_loss + static_loss

		# L2 regularization
		if model.L2_lambda > 0:
			l2_loss = tf.nn.l2_loss(regularization_terms[0])
			for i in range(1,len(regularization_terms)):
				l2_loss += tf.nn.l2_loss(regularization_terms[i])
			loss += model.L2_lambda * l2_loss

		return loss / math_ops.cast(batch_size, dtypes.float32)										

def relation_nce_loss(model, add_weight, example_idxs, head_entity_name, relation_name, tail_entity_name):
	relation_vec = model.relation_dict[relation_name]['embedding']
	example_emb = model.entity_dict[head_entity_name]['embedding']
	label_idxs = model.relation_dict[relation_name]['idxs']
	label_emb = model.entity_dict[tail_entity_name]['embedding']
	label_bias = model.relation_dict[relation_name]['bias']
	label_size = model.entity_dict[tail_entity_name]['size']
	label_distribution = model.relation_dict[relation_name]['distribute']
	loss, embs = pair_search_loss(model, add_weight, relation_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution)
	#print(loss.get_shape())
	#print(model.relation_dict[relation_name]['weight'].get_shape())
	return tf.reduce_sum(model.relation_dict[relation_name]['weight'] * loss), embs									

def pair_search_loss(model, add_weight, relation_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution):
	batch_size = array_ops.shape(example_idxs)[0]#get batch_size										
	# Nodes to compute the nce loss w/ candidate sampling.										
	labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])										
											
	# Negative sampling.										
	sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(										
			true_classes=labels_matrix,								
			num_true=1,								
			num_sampled=model.negative_sample,								
			unique=False,								
			range_max=label_size,								
			distortion=0.75,								
			unigrams=label_distribution))								
											
	#get example embeddings [batch_size, embed_size]
	#example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-add_weight) + relation_vec * add_weight							
	example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) + relation_vec
											
	#get label embeddings and bias [batch_size, embed_size], [batch_size, 1]										
	true_w = tf.nn.embedding_lookup(label_emb, label_idxs)										
	true_b = tf.nn.embedding_lookup(label_bias, label_idxs)										
											
	#get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]										
	sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)										
	sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)										
											
	# True logits: [batch_size, 1]										
	true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b										
											
	# Sampled logits: [batch_size, num_sampled]										
	# We replicate sampled noise lables for all examples in the batch										
	# using the matmul.										
	sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])										
	sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec										
											
	return nce_loss(true_logits, sampled_logits), [example_vec, true_w, sampled_w]						
											
def nce_loss(true_logits, sampled_logits):											
	"Build the graph for the NCE loss."										
											
	# cross-entropy(logits, labels)										
	true_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=true_logits, labels=tf.ones_like(true_logits))								
	sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(										
			logits=sampled_logits, labels=tf.zeros_like(sampled_logits))								
	#print(true_xent.get_shape())
	#print(sampled_xent.get_shape())
	# NCE-loss is the sum of the true and noise (sampled words)										
	# contributions, averaged over the batch.										
	nce_loss_tensor = (true_xent + tf.reduce_sum(sampled_xent, 1)) 										
	return nce_loss_tensor										



