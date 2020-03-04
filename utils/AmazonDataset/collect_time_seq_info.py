import os,sys
import gzip
from array import array
import struct, ast
import operator

def collect_time_seq_info(data_path, review_file_path):

	# read valid review ids
	review_id_to_idx = {}
	with gzip.open(data_path + '/review_id.txt.gz', 'rt') as fin:
		index = 0
		for line in fin:
			review_id_to_idx[line.strip()] = index
			index += 1

	# read review time information
	user_review_seq = {} # recording the sequence of user reviews in time
	item_review_seq = {} # recording the sequence of item reviews in time
	with gzip.open(review_file_path, 'r') as g:
		index = 0
		for l in g:
			l = eval(l)
			user = l['reviewerID']
			product = l['asin']
			review_text = l['reviewText']
			summary = l['summary']
			rating = l['overall']
			time = l['unixReviewTime']
			review_id = 'line_' + str(index)

			if review_id in review_id_to_idx:
				if user not in user_review_seq:
					user_review_seq[user] = []
				user_review_seq[user].append((review_id_to_idx[review_id], time))
				if product not in item_review_seq:
					item_review_seq[product] = []
				item_review_seq[product].append((review_id_to_idx[review_id], time))

			index += 1

	# read user list
	user_list = []
	with gzip.open(data_path + '/users.txt.gz', 'rt') as fin:
		for line in fin:
			user_list.append(line.strip())

	# read product list
	product_list = []
	with gzip.open(data_path + '/product.txt.gz', 'rt') as fin:
		for line in fin:
			product_list.append(line.strip())

	# Sort each user's reviews according to time and output to files
	review_loc_time_list = [[] for _ in range(len(review_id_to_idx))]
	with gzip.open(data_path + '/u_r_seq.txt.gz', 'wt') as fout:
		for user in user_list:
			review_time_list = user_review_seq[user]
			user_review_seq[user] = sorted(review_time_list, key=operator.itemgetter(1))
			fout.write(' '.join([str(x[0]) for x in user_review_seq[user]]) + '\n')
			for i in range(len(user_review_seq[user])):
				review_id = user_review_seq[user][i][0]
				time = user_review_seq[user][i][1]
				review_loc_time_list[review_id] = [i]

	# Sort each item's reviews according to time and output to files
	with gzip.open(data_path + '/p_r_seq.txt.gz', 'wt') as fout:
		for product in product_list:
			review_time_list = item_review_seq[product]
			item_review_seq[product] = sorted(review_time_list, key=operator.itemgetter(1))
			fout.write(' '.join([str(x[0]) for x in item_review_seq[product]]) + '\n')
			for i in range(len(item_review_seq[product])):
				review_id = item_review_seq[product][i][0]
				time = item_review_seq[product][i][1]
				review_loc_time_list[review_id].append(i)
				review_loc_time_list[review_id].append(time)

	# Output the location (sorted by time) of each review in the corresponding user review list for quick indexing.
	with gzip.open(data_path + '/review_uloc_ploc_and_time.txt.gz', 'wt') as fout:
		for t_l in review_loc_time_list:
			fout.write(' '.join([str(x) for x in t_l]) + '\n')

def main():
	data_path = sys.argv[1]
	review_file_path = sys.argv[2]
	collect_time_seq_info(data_path, review_file_path)


if __name__ == "__main__":
	main()
