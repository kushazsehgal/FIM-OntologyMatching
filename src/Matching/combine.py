import pandas as pd
from operator import add
combine_paths = ['../../output/Matching/Match_Data_1.csv','../../output/Matching/Match_Data_2.csv','../../output/Matching/output/Match_Data_3.csv','../../output/Matching/output/Match_Data_4.csv']
output_path = '../../output/Matchingoutput/Match_Data_Combined_final.csv'
def do(combine_paths,output_path):
    
    Code_Count = []
    Intent_Count = []
    Fuzzy_Code_Count = []
    Fuzzy_Intent_Count = []
    Ontology_Words = []
    for i in range(len(combine_paths)):
        
        df = pd.read_csv(combine_paths[i])
        if i == 0:
            Ontology_Words = df['Ontology_Words']
            Code_Count = df['Code_Count']
            Intent_Count = df['Intent_Count']
            Fuzzy_Code_Count = df['Fuzzy_Code_Count']
            Fuzzy_Intent_Count = df['Fuzzy_Intent_Count']
        else:
            Code_Count = list( map (add, Code_Count, list(df['Code_Count'])))  
            Intent_Count = list( map (add, Intent_Count, list(df['Intent_Count'])))  
            Fuzzy_Code_Count = list( map (add, Fuzzy_Code_Count, list(df['Fuzzy_Code_Count'])))  
            Fuzzy_Intent_Count = list( map (add, Fuzzy_Intent_Count, list(df['Fuzzy_Intent_Count'])))         
    
    dataframe = pd.DataFrame({'Ontology_Words':Ontology_Words,'Code_Count':Code_Count,'Intent_Count':Intent_Count,'Fuzzy_Code_Count':Fuzzy_Code_Count,'Fuzzy_Intent_Count':Fuzzy_Intent_Count})
    print(dataframe.head())
    dataframe.to_csv(output_path)
if __name__ == '__main__':
    do(combine_paths,output_path)