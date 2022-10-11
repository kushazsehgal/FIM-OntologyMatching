import json
import itertools
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

json_path = '../../Matching/data/part2.json'
csv_path = '../../Matchingdata/ontology.csv'
STOREPATH = '../../outputs/Matching'
stopwords_character = ['.',',','?','!',':',';','(',')','[',']','{','}','\\','\"','\n','\t','\r','\v','\f','\b','\a','\0','\1','\2','\3','\4','\5','\6','\7','_','-','+','=','*','&','%','$','#','@','~','`','|','^','\\','<','>','/','\'','!']

global df
fuzzy_thresh = 80

def load_json_data(file_path):
    json_content = open(file_path)
    result = json.load(json_content)
    return result

def load_jsonl_data(file_path):
    jsonl_content = open(file_path)
    result = []
    for line in jsonl_content.readlines():
        result.append(json.loads(line))
    return result

def create_match_lists(csv_path):
    """
    Returns unigram,bigram and trigram lists.
    """
    df1 = pd.read_csv(csv_path)
    unigram = []
    bigram = []
    trigram = []
    for words in df1['concepts']:
        word_list = get_words(words)
        if len(word_list) == 1:
            unigram.append(words.lower())
        elif len(word_list) == 2:
            bigram.append(words.lower())
        else:
            trigram.append(words.lower())

    onto_words = unigram+bigram+trigram
    df = pd.DataFrame({'Ontology_Words':np.array(onto_words),'Code_Count':np.zeros((len(onto_words))),'Intent_Count':np.zeros(len(onto_words)),'Fuzzy_Code_Count':np.zeros(len(onto_words)),'Fuzzy_Intent_Count':np.zeros(len(onto_words))})
    df.index = list(df['Ontology_Words'])
    return unigram,bigram,trigram, df

def process_string(string):
    """
    From a given string, removes the unwanted characters and returns the list of words present in the string. 
    """
    newstring = ''
    for i in range(len(string)):
        char = string[i]
        if char in stopwords_character:
            if i != 0  and i != (len(string) - 1) and (string[i-1] == ' ' or string[i+1] == ' '):
                char = ''
            else:
                char = ' '
        newstring = newstring + (char)
    return get_words(newstring)

def get_words(string):
    words = [word for word in (string.lower()).split(" ")]
    try:
        while True:
            words.remove('')
    except ValueError:
        pass
    return words

def process_data(file_path, unigram, bigram, trigram, df):
    result = load_json_data(file_path)
    count = 1
    for dic in result:
        if count <= 1250:
            code_words = process_string(dic['code'])
            intent_words = process_string(dic['docstring'])
            match(code_words, intent_words, unigram, bigram, trigram, df)
            print(count)
            count += 1
        else:
            return

def match(code_words, intent_words, unigram, bigram, trigram, df):
    """
    Executes Unigram,Bigram and Trigram Matching of intent and code snippet. 
    """
    code_bigrams = generate_N_grams(code_words,2)
    intent_bigrams = generate_N_grams(intent_words,2)
    code_trigrams = generate_N_grams(code_words,3)
    intent_trigrams = generate_N_grams(intent_words,3)
    
    do_match(code_words,intent_words,unigram, df)
    do_match(code_bigrams,intent_bigrams,bigram, df)
    do_match(code_trigrams,intent_trigrams,trigram, df)
    return

def do_match(code_list,intent_list,match_list, df):
    """
    Appends the count of each entity in the code snippet and intent to the corresponding list in the dictionary.
    """
    code_sum = 0
    intent_sum = 0
    fuzzy_code_sum = 0
    fuzzy_intent_sum = 0
    for entity in match_list:
        df.at[entity, 'Code_Count'] += code_list.count(entity)
        df.at[entity, 'Intent_Count'] += intent_list.count(entity)
        for code_entity in code_list:
            code_entity = code_entity.lower()
            if fuzz.token_set_ratio(entity, code_entity) > fuzzy_thresh:
                df.at[entity, 'Fuzzy_Code_Count'] += 1
        for intent_entity in intent_list:
            intent_entity = intent_entity.lower()
            if fuzz.token_set_ratio(entity, code_entity) > fuzzy_thresh:
                df.at[entity, 'Fuzzy_Intent_Count'] += 1
    return 

def generate_N_grams(words,ngram=1):
    """
    Generates n-gram concatenations from a list of words and amount of words to be concatenated.
    """
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def store_dataframe(storepath,dataframe,name): 
    dataframe.to_csv(f'{storepath}/{name}')
    return

def main():
    unigram, bigram, trigram, df = create_match_lists(csv_path)
    process_data(json_path, unigram, bigram, trigram, df)
    store_dataframe(STOREPATH, df, 'Match_Data_1.csv')

if __name__ == '__main__':
    main()