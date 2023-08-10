import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

table=pd.read_excel('./data/new.xlsx')
clue=['收购','虚增','借款','转让','子公司','关联','赔偿','担保','融资','解聘','减持','修正','更正','贷款']

# Iterate over each row
for index, row in table.iterrows():
    text = str(row['behave'])  # Convert the cell value to string
    #words = jieba.cut()  
    words = jieba.cut(text, cut_all=False)
    words = list(words)


    # Split the text into words

    # Find the combined recognized phrases
    combined_phrases = []
    i = 0
    while i < len(words):
        phrase = words[i]
        if phrase == '更正':
            phrase = '修正'
            if phrase in clue and phrase not in combined_phrases:
                combined_phrases.append(phrase)
        else:
            if phrase in clue and phrase not in combined_phrases:
                combined_phrases.append(phrase)
        i += 1

    # Find the recognized words with words in-between less than 10 characters


    # Assign the recognized words to the 'mark' column
    table.at[index, 'clue'] = ', '.join(combined_phrases)

# Save the updated table to a new Excel file
table.to_excel('clue.xlsx', index=False)



read_table = pd.read_excel('clue.xlsx')


         

split_clue = read_table['clue'].str.split(',', expand=True).stack().str.strip().reset_index(level=1, drop=True)
split_clue.name = 'phrase'

# Count the occurrences of each phrase
phrase_counts = split_clue.value_counts()



print(phrase_counts)

