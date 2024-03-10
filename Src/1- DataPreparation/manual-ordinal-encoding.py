import pandas as pd

df = pd.read_csv('test.csv', converters={'Eğitim Durumu': lambda s: {'İlkokul': 0, 'Ortaokul': 1, 'Lise': 2, 'Üniversite': 3}[s]})
print(df, end='\n\n')