cd ../../

# Download Amazon review dataset "Cell_Phones_and_Accessories" 5-core.
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

# Download the meta data from http://jmcauley.ucsd.edu/data/amazon/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz

# Stem and remove stop words from the Amazon review datasets if needed. Here, we stem the field of “reviewText” and “summary” without stop words removal.
java -Xmx4g -jar ./utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar false ./reviews_Cell_Phones_and_Accessories_5.json.gz ./reviews_Cell_Phones_and_Accessories_5.processed.gz

# Index datasets
python ./utils/AmazonDataset/index_and_filter_review_file.py reviews_Cell_Phones_and_Accessories_5.processed.gz ./tmp_data/ 5

# Match the meta data with the indexed data to extract queries:
java -Xmx16G -jar ./utils/AmazonDataset/jar/AmazonMetaData_matching.jar false ./meta_Cell_Phones_and_Accessories.json.gz ./tmp_data/min_count5/

# Gather knowledge from meta data:
python ./utils/AmazonDataset/match_with_meta_knowledge.py ./tmp_data/min_count5/ meta_Cell_Phones_and_Accessories.json.gz

# Randomly split train/test
## The 30% purchases of each user are used as test data
## Also, we randomly sample 20% queries and make them unique in the test set.
python utils/AmazonDataset/random_split_train_test_data.py tmp_data/min_count5/ 0.3 0.3

# Sequentially split train/test
## The last 20% purchases of each user are used as test data
## Also, we manually sample 20% queries and make them unique in the test set.
#python utils/AmazonDataset/sequentially_split_train_test_data.py tmp_data/min_count5/ 0.2 0.2

# Run model
python ProductSearch/main.py --data_dir=./tmp_data/min_count5/ --input_train_dir=./tmp_data/min_count5/random_query_split/ --subsampling_rate=0.0001 --window_size=3 --embed_size=400 --negative_sample=5 --learning_rate=0.5 --batch_size=64 --max_train_epoch=20 net_struct=fs --rank_cutoff100  --dynamic_weight=0.2 --L2_lambda=0.005 --steps_per_checkpoint=400
#python ProductSearch/main.py --data_dir=./tmp_data/min_count5/ --input_train_dir=./tmp_data/min_count5/seq_query_split/ --subsampling_rate=0.0001 --window_size=3 --embed_size=400 --negative_sample=5 --learning_rate=0.5 --batch_size=64 --max_train_epoch=20 net_struct=fs --rank_cutoff100  --dynamic_weight=0.2 --L2_lambda=0.005 --steps_per_checkpoint=400

# Test model
python ProductSearch/main.py --data_dir=./tmp_data/min_count5/ --input_train_dir=./tmp_data/min_count5/random_query_split/ --subsampling_rate=0.0001 --window_size=3 --embed_size=400 --negative_sample=5 --learning_rate=0.5 --batch_size=64 --max_train_epoch=20 net_struct=fs --rank_cutoff100  --dynamic_weight=0.2 --L2_lambda=0.005 --steps_per_checkpoint=400 --decode=True
#python ProductSearch/main.py --data_dir=./tmp_data/min_count5/ --input_train_dir=./tmp_data/min_count5/seq_query_split/ --subsampling_rate=0.0001 --window_size=3 --embed_size=400 --negative_sample=5 --learning_rate=0.5 --batch_size=64 --max_train_epoch=20 net_struct=fs --rank_cutoff100  --dynamic_weight=0.2 --L2_lambda=0.005 --steps_per_checkpoint=400 --decode=True

# Download and install galago
wget https://iweb.dl.sourceforge.net/project/lemur/lemur/galago-3.16/galago-3.16-bin.tar.gz
tar xvzf galago-3.16-bin.tar.gz


# Compute ranking metrics
./galago-3.16-bin/bin/galago eval --judgments= ./tmp_data/min_count5/random_query_split/test.qrels --runs+ ./tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10
#./galago-3.16-bin/bin/galago eval --judgments= ./tmp_data/min_count5/seq_query_split/test.qrels --runs+ ./tmp/test.bias_product.ranklist  --metrics+recip_rank --metrics+ndcg10 --metrics+P10
