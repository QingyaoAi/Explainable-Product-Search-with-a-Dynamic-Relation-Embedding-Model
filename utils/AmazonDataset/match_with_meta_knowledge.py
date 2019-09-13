import os,sys
import gzip
from array import array
import struct, ast

def match_and_create_knowledge(data_path, meta_path):

	# read needed product ids
	product_ids = []
	with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
		for line in fin:
			product_ids.append(line.strip())
	product_indexes = dict([(product_ids[i], i) for i in range(len(product_ids))])

	# match with meta data
	brand_vocab, brand_list = {}, []
	category_vocab, category_list = {}, []
	related_product_vocab, related_product_list = {}, []
	know_dict = {
		'also_bought' : [[] for _ in range(len(product_ids))],
		'also_viewed' : [[] for _ in range(len(product_ids))],
		'bought_together' : [[] for _ in range(len(product_ids))],
		'brand' : [[] for _ in range(len(product_ids))],
		'categories' : [[] for _ in range(len(product_ids))]
	}
	count_dict = {
		'also_bought' : 0,
		'also_viewed' : 0,
		'bought_together' : 0,
		'brand' : 0,
		'categories' : 0
	}
	# read meta_data
	with gzip.open(meta_path, 'rt') as fin:
		for line in fin:
			meta = ast.literal_eval(line)
			if meta['asin'] in product_indexes:
				pidx = product_indexes[meta['asin']]
				if 'related' in meta:
					related = meta['related']
					if 'also_bought' in related:
						for asin in related['also_bought']:
							if asin not in related_product_vocab:
								related_product_vocab[asin] = len(related_product_list)
								related_product_list.append(asin)
						know_dict['also_bought'][pidx] = [related_product_vocab[asin] for asin in related['also_bought']]
						count_dict['also_bought'] += len(know_dict['also_bought'][pidx])
					# also view
					if 'also_viewed' in related:
						for asin in related['also_viewed']:
							if asin not in related_product_vocab:
								related_product_vocab[asin] = len(related_product_list)
								related_product_list.append(asin)
						know_dict['also_viewed'][pidx] = [related_product_vocab[asin] for asin in related['also_viewed'] if asin in product_indexes]
						count_dict['also_viewed'] += len(know_dict['also_viewed'][pidx])
					# bought together
					if 'bought_together' in related:
						for asin in related['bought_together']:
							if asin not in related_product_vocab:
								related_product_vocab[asin] = len(related_product_list)
								related_product_list.append(asin)
						know_dict['bought_together'][pidx] = [related_product_vocab[asin] for asin in related['bought_together'] if asin in product_indexes]
						count_dict['bought_together'] += len(know_dict['bought_together'][pidx])
				# brand
				if 'brand' in meta:
					if meta['brand'] not in brand_vocab:
						brand_vocab[meta['brand']] = len(brand_list)
						brand_list.append(meta['brand'])
					know_dict['brand'][pidx] = [brand_vocab[meta['brand']]]
					count_dict['brand'] += 1
				# categories
				if 'categories' in meta:
					categories_set = set()
					for category_tree in meta['categories']:
						for category in category_tree:
							if category not in category_vocab:
								category_vocab[category] = len(category_list)
								category_list.append(category)
							categories_set.add(category_vocab[category])
					know_dict['categories'][pidx] = list(categories_set)
					count_dict['categories'] += len(know_dict['categories'][pidx])

	fout_dict = {
		'also_bought' : gzip.open(data_path + 'also_bought_p_p.txt.gz', 'wt'),
		'also_viewed' : gzip.open(data_path + 'also_viewed_p_p.txt.gz', 'wt'),
		'bought_together' : gzip.open(data_path + 'bought_together_p_p.txt.gz', 'wt'),
		'brand' : gzip.open(data_path + 'brand_p_b.txt.gz', 'wt'),
		'categories' : gzip.open(data_path + 'category_p_c.txt.gz', 'wt')
	}
	for key in fout_dict:
		for i in range(len(product_ids)):
			# Write to files
			str_list = [str(x) for x in know_dict[key][i]]
			fout_dict[key].write(' '.join(str_list) + '\n')
		fout_dict[key].close()
	with gzip.open(data_path + 'related_product.txt.gz', 'wt') as fout:
		for asin in related_product_list:
			fout.write(asin + '\n')
	with gzip.open(data_path + 'brand.txt.gz', 'wt') as fout:
		for brand in brand_list:
			fout.write(brand + '\n')
	with gzip.open(data_path + 'category.txt.gz', 'wt') as fout:
		for category in category_list:
			fout.write(category + '\n')
	with open(data_path + 'knowledge_statistics.txt', 'wt') as fout:
		fout.write('Total Brand num %d\n' % len(brand_list))
		fout.write('Total Category num %d\n' % len(category_list))
		fout.write('Avg also_bought per product %.3f\n' % (float(count_dict['also_bought'])/len(product_ids)))
		fout.write('Avg also_view per product %.3f\n' % (float(count_dict['also_viewed'])/len(product_ids)))
		fout.write('Avg bought_together per product %.3f\n' % (float(count_dict['bought_together'])/len(product_ids)))
		fout.write('Avg brand per product %.3f\n' % (float(count_dict['brand'])/len(product_ids)))
		fout.write('Avg category per product %.3f\n' % (float(count_dict['categories'])/len(product_ids)))


def main():
	data_path = sys.argv[1]
	meta_path = sys.argv[2]
	match_and_create_knowledge(data_path, meta_path)


if __name__ == "__main__":
	main()
