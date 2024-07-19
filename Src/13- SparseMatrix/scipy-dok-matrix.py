from scipy.sparse import dok_matrix

dok1 = dok_matrix((5, 5), dtype='int32')
dok1[1, 2] = 10
dok1[0, 1] = 20


a = dok1.todense()
print(dok1)
print()
print(a)
print('-' * 20)


dok2 = dok_matrix((5, 5), dtype='int32')
dok2[3, 2] = 10
dok2[4, 1] = 20
dok2[1, 2] = 20


result = dok1 + dok2
print(result)

print('-' * 20)

result = dok1 * dok2
print(result)

