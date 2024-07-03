import pandas as pd

df = pd.read_csv('IMDB Dataset.csv', encoding='latin-1')

for index, text in enumerate(df['review']):
    rtext = text.replace('\n', ' ')
    rtext = rtext.replace('\r', ' ')
    df.iloc[index, 0] = rtext
    
df.to_csv('imdb.csv', index=False)

