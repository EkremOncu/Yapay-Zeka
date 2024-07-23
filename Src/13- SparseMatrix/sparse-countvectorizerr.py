from sklearn.feature_extraction.text import CountVectorizer

texts = ['this film is very very good', 'I hate this film', 'It is good', 'I don\'t like it']

cv = CountVectorizer()
cv.fit(texts)
result = cv.transform(texts)

print(result)
print()
print(result.todense())