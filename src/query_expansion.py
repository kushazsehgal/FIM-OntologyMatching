import pandas as pd
import nltk
from string import punctuation
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

CSV_PATH = './inputs/queries.csv' # SHOULD HAVE A "intent" AND A "code" COLUMN
PKL_FILE = './outputs/rules.pkl' # PATH TO PICKLE FILE STORING ITEMSETS
SAVE_FILE = './outputs/expanded_queries.csv'# FINAL SAVE LOCATION
# tokenize the questions and make a list of tokens
def tokenize_question(question):
    return nltk.wordpunct_tokenize(question)

# remove punctation and stopwords from the tokens
def remove_punctuation_and_stopwords(question):
    question_without_punctuation = [word for word in question if word not in punctuation]
    question_without_punctuation_and_stopwords= [word for word in question_without_punctuation if word.lower() not in nltk.corpus.stopwords.words('english')]
    return question_without_punctuation_and_stopwords

# only keep alphabetic tokens
def keep_alphabetic_tokens(question_removed):
    questions_removed_and_alphabetic = [word for word in question_removed if word.isalpha()]
    return questions_removed_and_alphabetic

# split camel case words
def split_camel_case_words(question):
    question_split_camel_case = []
    for words in question:
        if(words.islower()):
            question_split_camel_case.append(words)
        else:
            words = re.sub("([a-z])([A-Z])","\g<1> \g<2>",words)
            token_list = words.split(" ")
            for tokens in token_list:
                question_split_camel_case.append(tokens)
    return question_split_camel_case

# lower case all tokens
def lower_case_all_tokens(questions):
    return [word.lower() for word in questions]

# given a input intent, will do the entire preprocessing required
def process_input(input):
    tokenized_question = tokenize_question(input)
    tokenized_question_without_punctuation_and_stopwords = remove_punctuation_and_stopwords(tokenized_question)
    tokenized_question_without_punctuation_and_stopwords_and_alphabetic = keep_alphabetic_tokens(tokenized_question_without_punctuation_and_stopwords)
    tokenized_question_without_punctuation_and_stopwords_and_alphabetic_and_split_camel_case = split_camel_case_words(tokenized_question_without_punctuation_and_stopwords_and_alphabetic)
    return lower_case_all_tokens(tokenized_question_without_punctuation_and_stopwords_and_alphabetic_and_split_camel_case)

if __name__ == '__main__':

    data = pd.read_pickle(PKL_FILE)
    df = pd.read_csv(CSV_PATH)
    intents = df['intent']
    # codes = df['code']
    expanded_queries = []
    count = 0
    not_expanded_count = 0
    for intent in intents:
        # preprocess the input
        processed_input = process_input(intent)

        """
        The nqe research paper first looks for a 3 word phrase in the itemset, then 2 and lastly 1 word phrase.
        It takes the first phrase it finds and uses it to expand the query.
        Note that itemsets are stored with decreasing confidence, so first itemset found from the top will also have the highest confidence.
        """
        
        # Create 3 phrase combinations from intent
        all_3combinations = []
        for i in range(len(processed_input)):
            for j in range(i+1, len(processed_input)):
                for k in range(j+1, len(processed_input)):
                    all_3combinations.append(frozenset((processed_input[i], processed_input[j], processed_input[k])))

        isExpansionFound = False
        
        # looking for a 3 phrase match
        for i in range(len(data['antecedents'])):
            if(isExpansionFound):
                break
            for j in range(len(all_3combinations)):       
                if(all_3combinations[j] == data['antecedents'][i]):
                    consequents = data['consequents'][i]
                    expanded_query = processed_input.copy()
                    for words in consequents:
                        expanded_query.append(words)
                    expanded_queries.append(expanded_query)
                    isExpansionFound = True
                    break

        # if a 3 phrase match is found, continue
        if(isExpansionFound):
            continue

        # Create 2 phrase combinations from intent
        all_2combinations = []
        for i in range(len(processed_input)):
            for j in range(i+1, len(processed_input)):
                all_2combinations.append(frozenset((processed_input[i], processed_input[j])))


        isExpansionFound = False
        # looking for a 2 phrase match
        for i in range(len(data['antecedents'])):
            if(isExpansionFound):
                break
            for j in range(len(all_2combinations)):
                if(all_2combinations[j] == data['antecedents'][i]):
                    consequents = data['consequents'][i]
                    expanded_query = processed_input.copy()
                    for words in consequents:
                        expanded_query.append(words)
                    expanded_queries.append(expanded_query)
                    isExpansionFound = True
                    break
        # if a 2 phrase match is found, continue
        if(isExpansionFound):
            continue

        # Create 1 phrase combinations from intent
        all_1combinations = []
        for i in range(len(processed_input)):
            all_1combinations.append(frozenset([processed_input[i]]))
        

        isExpansionFound = False
        # looking for a 1 phrase match
        for i in range(len(data['antecedents'])):
            if(isExpansionFound):
                break
            for j in range(len(all_1combinations)):
                if(all_1combinations[j] == data['antecedents'][i]):
                    consequents = data['consequents'][i]
                    expanded_query = processed_input.copy()
                    for words in consequents:
                        expanded_query.append(words)
                    expanded_queries.append(expanded_query)
                    isExpansionFound = True
                    break
        # if a phrase match is not found, we simply add the intent without expansion to the list
        if(isExpansionFound) == False:
            not_expanded_count += 1
            expanded_query = processed_input.copy()
            expanded_queries.append(expanded_query)
        # some print statement to check progress
        # count += 1
        # if count%10 == 0:
        #     print(count)
    print("No expansion found for following number of queries: ", not_expanded_count)
    # Store final df with 3 columns,intents, expanded intents and codes
    final_df = pd.DataFrame({'intent':intents,'expanded_intents': expanded_queries})
    final_df.to_csv(SAVE_FILE, index=False)
            


