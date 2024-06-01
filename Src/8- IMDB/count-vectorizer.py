from sklearn.feature_extraction.text import CountVectorizer

texts = ["film güzeldi ve senaryo iyidi", "film berbattı, tam anlamıyla berbattı", "seyretmeye değmez", "oyuncular güzel oynamışlar", "senaryo berbattı, böyle senaryo olur mu?", "filme gidin de bir de siz görün"]

cv = CountVectorizer(dtype='uint8', stop_words=['de', 'bir', 've', 'mu'])
cv.fit(texts)

print(cv.vocabulary_)
print("-----------------------")

dataset_x = cv.transform(texts).todense()

print(dataset_x)



"""
" film güzeldi ve senaryo iyidi", 

" film berbattı, tam anlamıyla berbattı", 

"seyretmeye değmez",  

"oyuncular güzel oynamışlar",  

"senaryo berbattı,  böyle senaryo olur mu?",  

"filme gidin de bir de siz görün"

"""