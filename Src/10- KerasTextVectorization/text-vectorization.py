from tensorflow.keras.layers import TextVectorization

txt = ['film çok güzeldi', 'film güzeldi', 'film çok kötüydü', 'film ortalama bir filmdi']

tvv = TextVectorization(output_mode='count')
tvv.adapt(txt)

r = tvv(['film güzeldi, film', 'film kötüydü'])
print(r)

print("----------------------------------------")

result = tvv.get_vocabulary()
print(result)
