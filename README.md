# Overview #
This is an implementation of the Dynamic Relation Embedding Model (DREM) for personalized product search. Please cite the following paper if you plan to use it for your project：
    
*	Qingyao Ai, Yongfeng Zhang, Keping Bi, W. Bruce Croft. Explainable Product Search with a Dynamic Relation Embedding Model. ACM Transactions on Information Systems (TOIS). 2019
    	
The DREM is a deep neural network model that jointly learn latent representations for queries, products, users, and knowledge entities. 
It is designed as a generative model and the embedding representations for queries, users and items in the DREM are learned through optimizing the log likelihood of observed entity relationships. 
The probability (which is also the rank score) of an item being purchased by a user with a query can be computed with their corresponding latent representations. 
Please refer to the paper for more details.

* For data used in the original paper, please check https://github.com/QingyaoAi/Amazon-Product-Search-Datasets

### Requirements: ###
    1. To run the DREM model in ./ProductSearch/ and the python scripts in ./utils/, python 3.0+ and Tensorflow v1.3+ are needed. (In the paper, we used python 3.6 and Tensorflow v1.4.0)
    2. To run the jar package in ./utils/AmazonDataset/jar/, JDK 1.7 is needed
    3. To compile the java code in ./utils/AmazonDataset/java/, galago from lemur project (https://sourceforge.net/p/lemur/wiki/Galago%20Installation/) is needed. 

### Install

**Create virtual environment (optional):**
```
pip install --user virtualenv
~/.local/bin/virtualenv -p python3 ./venv
source venv/bin/activate
```

**Install DREM from the source:**
```
git clone https://github.com/QingyaoAi/Explainable-Product-Search-with-a-Dynamic-Relation-Embedding-Model.git
cd knowledgeembedding
python setup.py install #use setup-gpu.py for GPU support
```

**Run example:**
```
cd example/AmazonDataset/
bash exp_pipeline.sh
```

### Data Preparation ###
    1. Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/ (e.g. In our paper, we used 5-core data)
    2. Stem and remove stop words from the Amazon review datasets if needed (e.g. In our paper, we stem the field of “reviewText” and “summary” without stop words removal)
        1. java -Xmx4g -jar ./utils/AmazonDataset/jar/AmazonReviewData_preprocess.jar <jsonConfigFile> <review_file> <output_review_file>
            1. <jsonConfigFile>: A json file that specify the file path of stop words list. An example can be found in the root directory. Enter “false” if don’t want to remove stop words. 
            2. <review_file>: the path for the original Amazon review data
            3. <output_review_file>: the output path for processed Amazon review data
    3. Index datasets
        1. python ./utils/AmazonDataset/index_and_filter_review_file.py <review_file> <indexed_data_dir> <min_count>
            1. <review_file>: the file path for the Amazon review data
            2. <indexed_data_dir>: output directory for indexed data
            3. <min_count>: the minimum count for terms. If a term appears less then <min_count> times in the data, it will be ignored.
    4. Extract queries and Split train/test
        1. Download the meta data from http://jmcauley.ucsd.edu/data/amazon/ 
        2. Match the meta data with the indexed data:
            1. java -Xmx16G -jar ./utils/AmazonDataset/jar/AmazonMetaData_matching.jar <jsonConfigFile> <meta_data_file> <indexed_data_dir>
                1. <jsonConfigFile> : A json file that specify the file path of stop words list. An example can be found in the root directory. Enter “false” if don’t want to remove stop words. 
                2. <meta_data_file>:  the path for the meta data
                3. <indexed_data_dir>: the directory for indexed data
        3. Split datasets for training and test
            1. python ./utils/AmazonDataset/random_split_train_test_data.py <indexed_data_dir> <review_sample_rate> <query_sample_rate>
            2. <indexed_data_dir>: the directory for indexed data
            3. <review_sample_rate>: the proportion of reviews used in test for each user (e.g. in our paper, we used 0.3).
            4. <query_sample_rate>: the proportion of queries used in test (e.g. in our paper, we used 0.3).

	

### Model Training/Testing ###
    1. python ./ProductSearch/main.py --<parameter_name> <parameter_value> --<parameter_name> <parameter_value> … 
        1. learning_rate:  The learning rate in training. Default 0.5.
        2. learning_rate_decay_factor: Learning rate decays by this much whenever the loss is higher than three previous loss. Default 0.90
        3. max_gradient_norm: Clip gradients to this norm. Default 5.0
        4. subsampling_rate: The rate to subsampling. Default 1e-4. 
        5. L2_lambda: The lambda for L2 regularization. Default 0.0
        6. query_weight: The weight for queries in the joint model of queries and users. Default 0.5
        7. batch_size: Batch size used in training. Default 64
        8. data_dir: Data directory, which should be the <indexed_data_dir>
        9. input_train_dir: The directory of training and testing data, which usually is <data_dir>/query_split/
        10. train_dir: Model directory & output directory
        11. similarity_func: The function to compute the ranking score for an item with the joint model of query and user embeddings. Default “product”.
            1. “product”: the dot product of two vectors.
            2. “cosine”: the cosine similarity of two vectors.
            3. “bias_product”: the dot product plus a item-specific bias
        12. net_struct:  Network structure parameters. Different parameters are separated by “_” (e.g. ). Default “simplified_fs”
            1. “LSE”: the latent space entity model proposed by Gysel et al. [1]
            2. “simplified”: simplified embedding-based language models without modeling for each review [2]
            3. “pv”: embedding-based language models with review modeling using paragraph vector model. [3]
            4. “hdc”: regularized embedding-based language models with word context. [4]
            5. “mean”: average word embeddings for query embeddings [5]
            6. “fs”: average word embeddings with non-linear projection for query embeddings [1]
            7. “RNN”: recurrent neural network encoder for query embeddings
        13. embed_size: Size of each embedding. Default 100.
        14. window_size: Size of context window for hdc model. Default 5.
        15. max_train_epoch: Limit on the epochs of training (0: no limit). Default 5.
        16. steps_per_checkpoint: How many training steps to do per checkpoint. Default 200
        17. seconds_per_checkpoint: How many seconds to wait before storing embeddings. Default 3600
        18. negative_sample: How many samples to generate for negative sampling. Default 5.
        19. decode: Set to “False" for training and “True" for testing. Default “False"
        20. test_mode: Test modes. Default “product_scores"
            1. “product_scores”: output ranking results and ranking scores; 
            2. “output_embedding": output embedding representations for users, items and words.
            3. “explain": start interactive explanation mode. Specify product, user, and query id to find the nearest neighbors of each entity in different entity space. Read interactive_explain_mode() in ./ProductSearch/main.py for more information.
        21. rank_cutoff: Rank cutoff for output rank lists. Default 100.
    2. Evaluation
        1. After training with "--decode False”, generate test rank lists with "--decode True”.
        2. TREC format rank lists for test data will be stored in <train_dir> with name “test.<similarity_func>.ranklist”
        3. Evaluate test rank lists with ground truth <input_train_dir>/test.qrels using trec_eval or galago eval tool.

### Example Parameter Settings ###

| Hyper-parameters  | Electronics | Kindle Store | CDs & Vinyl | Cell Phones & Accessories |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| min\_count | 5 | 5 | 5 | 5 |
| embed\_size  | 400 | 400 | 400 | 400 |
| negative\_sample  | 5 | 5 | 5 | 5 |
| learning\_rate | 0.5 | 0.5 | 0.5 | 0.5 |
| max\_train\_epoch | 20 | 20 | 20 | 20 |
| net\_struct | fs | fs | fs | fs |
| batch\_size | 64 | 64 | 64 | 64 |
| dynamic\_weight | 0.1 | 0.5 | 0.3 | 0.2 |
| L2\_lambda | 0.005 | 0.005 | 0.005 | 0.005 |
| subsampling\_rate  | 0.0001  | 0.0001  | 0.0001  | 0.0001  |
| window\_size  | 3 | 3 | 3 | 3 |
| similarity\_func | bias\_product | bias\_product | bias\_product | bias\_product |

