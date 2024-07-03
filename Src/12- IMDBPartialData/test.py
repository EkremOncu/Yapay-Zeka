BATCH_SIZE = 32

offsets = [0, ]

f = open('IMDB Dataset.csv', encoding='latin-1')

i = 0
while True:
    line = f.readline()
    if line == '':
        break
  
    if i % BATCH_SIZE == 0:
        offsets.append(f.tell())
    i += 1
    
f.seek(offsets[5], 0)    
fifth_line = f.readline()
print(fifth_line)
        
import random

random.shuffle(offsets)

