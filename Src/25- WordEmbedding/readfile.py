with open('test.txt', encoding='utf-8') as f:
    f.readline()
    
    data = {}
    for index, line in enumerate(f):
        words = line.split()
        if (len(words) == 0):
            continue
        data[words[0]] = index, [float(word) for word in words[1:]]
        
        
        
        