# Frequent Itemset Mining for Intents Expansion and Ontology Matching
## Installing Dependencies
Run the following Command to install all dependencies
`pip install -r requirements.txt`
The required packages mainly are as follows - 
1. fuzzywuzzy
2. mlxtend
3. nltk
## Usage
Frequent Itemset Mining
1. Data preparation: Place the CSV file to be used for building rules using FIM in the `data` folder and CSV file with queries which are to be expanded in the `inputs` folder and rename both CSV files to `queries.csv`. The CSV files should have the following columns:
    * `intent`: The query strings.
    * `code`: The code snippets. (Needed in data only if you want to use the code snippets for building FIM.)
2. Run the following command to build the FIM:
    ```shell
    python3 src/fim_computation.py
    ```
    After running the above command, the `rules.csv` file will be saved in the `outputs` folder.
3. Run the following command to expand the queries:
    ```shell
    python3 src/query_expansion.py
    ```
    After running the above command, the `expanded_queries.csv` file will be saved in the `outputs` folder.

Generating Match Count for an Ontology
1. place the intents json/jsonl file and the onotology csv file in `data/Matching`
2. run `text-matching.py` to generate the `Match_Data.csv` file in `outputs/Matching`
     ```shell
    cd ./src/Matching
    python text-matching.py
    ```
3. To Combine all generated Match Files execute `combine.py` with corresponding combine paths to create the `Match_Data_Combined.csv` file in `outputs/Matching`
     ```shell
    cd ./src/Matching
    python combine.py
    ```
4. It generates a count of matches for each unigram, bigram and trigram available in the ontology against the intents present in the json/jsonl file
## Algorithm
The algorithm for Frequent Itemset Mining is based on the following steps:
1. Building of FIM. [1]
    * FIM is built using `apriori algorithm` implemented in `mlxtend`.
    * Itemsets with `high support` are used to build the FIM.
2. Expansion of the queries. [2]
    * Given a query, for each combination set of `three keywords`, s_b, and for each possible expansion s_e ∈ V_k , we rank by the confidence(s_b → s_e ). 
    * If there are no expansions, then the same process occurs using `two key-words`, and then `one`. 
    * We return only `one possible expansion`, for a conservative expansion.

The algorithm for Matching is the following -
1. We first preprocess all intents by removing `unwanted characters` and `stopwords` and return a list of preprocessed words present in each intent
2. We then create a list of `n-grams` - unigram, bigram and trigrams
3. We perform unigram , bigram and trigram matching along with a direct `fuzzy-matching`  with a `fuzz-threshold of 80`
## References:
1. [1] https://www.section.io/engineering-education/introduction-to-frequent-itemset-mining-with-python/
2. [2] https://research.facebook.com/publications/neural-query-expansion-for-code-search/
