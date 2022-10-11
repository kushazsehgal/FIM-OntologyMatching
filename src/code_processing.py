import nltk
import re
import pandas as pd


"""
Helper file to preprocess codes to be used in fim_computation.py
"""
# splitting camel case words
def split_camel_case_words(questions):
    questions_and_split_camel_case = []
    for question in questions:
        question_split_camel_case = []
        # if the question is empty, then we don't need to do anything
        if type(question) != list:
            questions_and_split_camel_case.append([""])
            continue
        for words in question:
            if(words.islower()):
                question_split_camel_case.append(words)
            else:
                # regex expression to split camel case words
                words = re.sub("([a-z])([A-Z])","\g<1> \g<2>",words)
                token_list = words.split(" ")
                for tokens in token_list:
                    question_split_camel_case.append(tokens)
        questions_and_split_camel_case.append(question_split_camel_case)
    return questions_and_split_camel_case

# split snake case words
def split_snake_case_words(codes):
    codes_without_snake_case = []
    for code in codes:
        snake_case_less_code = []
        for word in code:
            # replacing _ with space
            words = word.split("_")
            snake_case_less_code += words
        codes_without_snake_case.append(snake_case_less_code)
    return codes_without_snake_case

def remove_integers(codes):
    """
    Remove integers from the list of codes.
    """
    for n, code in enumerate(codes):
        code = [word for word in code if not word.isdigit()]
        codes[n] = code
    return codes
def preprocess_code(codes):
    """
    Preprocess the list of codes.
    """
    for n, code in enumerate(codes):
        # Replace characters which are not in [a-zA-Z0-9_] with spaces
        if(pd.isna(code)):
            codes[n] = [""]
            continue
        code = re.sub(r'[^a-zA-Z0-9_]', ' ', code)
        # Replace multiple spaces with a single space
        code = re.sub(r'\s+', ' ', code)
        # Remove leading and trailing spaces
        code = code.strip()
        # Tokenize the code
        code = nltk.wordpunct_tokenize(code)
        codes[n] = code
    
    # split camel case words into separate words
    codes = split_camel_case_words(codes)
    preprocessed_codes = split_snake_case_words(codes)
    preprocessed_codes = remove_integers(preprocessed_codes)
    for n, code in enumerate(preprocessed_codes):
        # Convert all words to lowercase
        code = [word.lower() for word in code]
        # Remove words of length 1
        code = [word for word in code if len(word) > 1]
        preprocessed_codes[n] = code
        
    return preprocessed_codes
        


