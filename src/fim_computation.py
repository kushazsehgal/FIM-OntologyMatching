import pandas as pd
import nltk 
from string import punctuation
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from code_processing import *


CSV_PATH = './data/queries.csv'
# CSV FILE SHOULD HAVE A "intent" COLUMN or "intent" and "code" column if you want to use code for FIM expansion

# load the csv file and make list of data present on Questions column
def read_csv_file(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)
    questions_list = df['intent'].tolist()
    return questions_list#[:5]

# tokenize the questions and make a list of tokens
def tokenize_questions(questions_list):
    tokenized_questions = []
    for question in questions_list:
        tokenized_questions.append(nltk.wordpunct_tokenize(question))
    return tokenized_questions

# remove punctation and stopwords from the tokens
def remove_punctuation_and_stopwords(tokenized_questions):
    ques_removed_punct = []
    for question in tokenized_questions:
        ques_removed_punct.append([word for word in question if word not in punctuation])
    ques_removed = []
    for question in ques_removed_punct:
        ques_removed.append([word for word in question if word.lower() not in nltk.corpus.stopwords.words('english')])
    return ques_removed

# only keep alphabetic tokens
def keep_alphabetic_tokens(questions):
    questions_final = []
    for question in questions:
        questions_final.append([word for word in question if word.isalpha()])
    return questions_final

# split camel case words
def split_camel_case_words(questions):
    questions_and_split_camel_case = []
    for question in questions:
        # print("DEBUG:", question)
        question_split_camel_case = []
        for words in question:
            # print("DEBUG2:", words)
            if(words.islower()):
                question_split_camel_case.append(words)
            else:
                words = re.sub("([a-z])([A-Z])","\g<1> \g<2>",words)
                token_list = words.split(" ")
                for tokens in token_list:
                    question_split_camel_case.append(tokens)
            # print("DEBUG3:", re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', words))
        questions_and_split_camel_case.append(question_split_camel_case)
    return questions_and_split_camel_case

# split snake case words
def split_snake_case_words(codes):
    codes_without_snake_case = []
    for code in codes:
        snake_case_less_code = []
        for word in code:
            words = word.split("_")
            snake_case_less_code += words
        codes_without_snake_case.append(snake_case_less_code)
    return codes_without_snake_case

# lower case all tokens
def lower_case_all_tokens(questions):
    tokenized_questions_lower_case = []
    for question in questions:
        tokenized_questions_lower_case.append([word.lower() for word in question])
    return tokenized_questions_lower_case

# function with input list of questions and output tokenized questions lower case
def process_questions(questions_list):
    questions_list = read_csv_file(CSV_PATH)
    tokenized_questions = tokenize_questions(questions_list)
    tokenized_questions_without_punctuation_and_stopwords = remove_punctuation_and_stopwords(tokenized_questions)
    tokenized_questions_without_punctuation_and_stopwords_and_alphabetic = keep_alphabetic_tokens(tokenized_questions_without_punctuation_and_stopwords)
    tokenized_questions_without_punctuation_and_stopwords_and_alphabetic_and_split_camel_case = split_camel_case_words(tokenized_questions_without_punctuation_and_stopwords_and_alphabetic)
    tokenized_questions_lower_case = lower_case_all_tokens(tokenized_questions_without_punctuation_and_stopwords_and_alphabetic_and_split_camel_case)
    return tokenized_questions_lower_case

# data encoding required for apriori algorithm
def encode_data(tokenized_questions_lower_case):
    te = TransactionEncoder()
    te_ary = te.fit(tokenized_questions_lower_case).transform(tokenized_questions_lower_case)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

def save_df_to_csv(df, file_name):
    df.to_csv(file_name, index=False)

if __name__ == '__main__':
    
    # read the csv file and make list of data present on intent column
    questions_list = read_csv_file(CSV_PATH)
    # tokenize the questions and make a list of tokens
    processed_questions = process_questions(questions_list)
    ############EXECUTE THE IN BETWEEN LINES FOR CODE + INTENT FIM#############################
    # # reading the csv file and make a list of data present on code column
    # df = pd.read_csv(CSV_PATH)
    # codes = df['code'].tolist()
    # # procsess the codes and make a list of tokens
    # codes = preprocess_code(codes)
    # for i in range(len(codes)):
    #     processed_questions[i].extend(codes[i])
    #     processed_questions[i] = list(pd.unique(processed_questions[i]))
    ###########################################################################################
    df = encode_data(processed_questions)
    print("Encoded!\n")
    # We use mlxtend library to implement FIM
    
    #creating the itemsets while keeoing a minimum threshold for support
    frq_items = apriori(df, min_support = 0.005, use_colnames = True)
    # storing the itemsets in a dataframe and rejecting the itemsets with confidence less than threshold
    rules = association_rules(frq_items, metric ="lift", min_threshold = 0.1)
    # sorting the association rules
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
    print("Itemsets Created")
    # rules are saved to pickle as it maintains the datatype of stored objects
    # csv format stores the sets in a string format, thus losing their datatype, but still useful for viewing and analysis
    save_df_to_csv(rules, "./outputs/rules.csv")
    rules.to_pickle("./outputs/rules.pkl")