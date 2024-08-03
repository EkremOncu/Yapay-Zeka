"""
---------------------------------------------------------------------------
C ve Sistem Programcıları Derneği Sınıfta Yapılan Örnekler ve Özet Notlar
                                
Bu notlar Kaan ASLAN'ın notlarından yararlanılarak oluşturulmuştur. 
---------------------------------------------------------------------------
"""

#  ----------------------------- NumPy  -----------------------------
"""
------------------------------------------------------------------------------------
Python'a vektörel işlem yapma yeteneğini kazandırmak için bazı üçüncü parti kütüphaneler 
oluşturulmuştur. Bunların en çok kullanılanı "NumPy" isimli kütüphanedir. Numpy 
C'de yazılmış bir kütüphanedir.

!!! NumPy dizileri  MUTABLE 'dir. (değiştirilebilir)

------------------------------------------------------------------------------------
import numpy  as np

a = np.array([  [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
print(a, type(a))

------------------------------------------------------------------------------------
Anımsanacağı gibi Python'daki list, tuple, gibi veri yapıları aslında değerlerin i
kendisini tutmamaktadır. Değerlerin tutulduğu nesnelerin adreslerini tutmaktadır. 
Bu biçimdeki çalışma yoğun sayısal işlemlerde oldukça hantal hale gelmektedir. Bu 
nedenle NumPy dizileri değerleri Python'un listeleri gibi değil C Programlama 
Dilindeki diziler gibi tutmaktadır. Yani NumPy dizilerinin elemanları genel olarak 
aynı türdendir.Örneğin:

a = np.array([1, 2, 3, 4, 5])
a.dtype

Burada a değişkenin gösteridği NumPy dizisi tamamen C'deki gibi bir dizidir. Yani 
NumPy dizisinin elemanları adresleri değil doğrudan değerlerin kendisini tutmaktadır. 
Bu nedenle NumPy dizileri (yani ndarray nesneleri) birkaç istisna durum dışında 
homojendir.

------------------------------------------------------------------------------------
array fonksiyonuyla NumPy dizisi (ndarray nesnesi) yaratılırken yaratılacak NumPy 
dizisinin C'deki dtype türü array fonksiyonun dtype parametresi ile açıkça 
belirlenebilir.

a = np.array([1, 2, 3, 4, 5], dtype= 'float64' )
a = np.array([1, 2, 3, 4, 5], dtype= np.float64 )
#  array([1., 2., 3., 4., 5.])

 NumPy dizilerinde kullanabileceğimiz dtype türlerinin önemli olanları şunlardır:  

bool_ /bool8        : bool türü
byte / int8         : bir byte'lık işaretli tamsayı türü
ubyte / uint8       : bir byte'lık işaretsiz tamsayı türü
short / int16       : iki byte'lık işaretli tamsayı türü
ushort / uint16     : iki byte'lık işaretsiz tamsayı türü
int32               : dört byte'lık işaretli tamsayı türü
uint32              : dört byte'lık işaretsiz tamsayı türü
int64               : sekiz byte'lık işaretli tamsayı türü
uint64              : sekiz byte'lık işaretsiz tamsayı türü
float32 / single    : dört byte'lık gerçek sayı türü

float64 / float / double : sekiz byte'lık gerçek sayı türü


işaretli tamsayı -> Örneğin : (-128, 127)
işaretsiz tamsayı -> Örneğin : (0, 255)

-> işaretli tamsayı = bitlerden biri işarete ayrılmış demek

!!! Bir NumPy dizisinin dtype'ı demek -> Onun C dilindeki gerçek(orjinal) türü demek

------------------------------------------------------------------------------------
String sınıfı C'deki string sınıfı gibi davranır, iterable değil.

import numpy  as np
a = np.array('ankara')
print(a)

a = np.array(list('ankara'))
print(a)

------------------------------------------------------------------------------------
İçi sıfırlarla dolu numpy dizilerinin oluşturulması gerekebilmektedir. Bunun için 
zeros fonksiyonu kullanılmaktadır. zeros fonksiyonun yine birinci parametresi 
oluşturulacak Numpy dizisinin boyutlarını (shape) belirtir.
 
import numpy  as np
a = np.zeros(10, dtype='int8') # tek boyutlu dizi
print(a)
print()

a = np.zeros( (10,10) , dtype='int8') # iki boyutlu dizi
print(a)

------------------------------------------------------------------------------------
ones isimli fonksiyon içi 1'lerle dolu bir NumPy dizisi oluşturmaktadır. Yine 
fonksiyonun birinci parametresi oluşturulacak dizinin boyutlarını belirtir. dtype 
parametresi ise dtype türünü belirtir.  Örneğin:

a = np.ones(10, dtype='int32')
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

------------------------------------------------------------------------------------
full isimli fonksiyon NumPy dizisini bizim istediğimiz değerle doldurarak yaratır. 
(Yani zeros ve ones fonksiyonlarının genel biçimidir.) Bu fonksiyonun yine birinci 
parametresi NumPy dizisinin boyutlarını, ikinci parametresi doldurulacak değerleri 
belirtmektedir. Fonksiyonda yine dtype belirtilebilir.

a = np.full(10, 5, dtype='int8')
# array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int8)

b = np.full((5, 5), 1.2, dtype='float32')
print(b)
------------------------------------------------------------------------------------
"""

# ---------- random ------------
"""
------------------------------------------------------------------------------------
Rastgele değerlerden NumPy dizisi oluşturabilmek için numpy.random modülünde çeşitli 
fonksiyonlar bulundurulmuştur. Örneğin numpy.random.random fonksiyonu belli bir 
boyutta 0 ile 1 arasında rastgele gerçek sayı değerleri oluşturmaktadır. Bu fonksiyon 
dtype parametresine sahip değildir. Her zaman float64 olarak numpy dizisini 
yaratmaktadır. Fonksiyonun boyut belirten bir parametresi vardır

------------------------------------------------------------------------------------
import numpy  as np

a = np.random.random(5)
#  array([0.16421352, 0.73812732, 0.78583484, 0.37765506, 0.25084917])

a = np.random.random((5, 5))
# array([[0.90589859, 0.41192256, 0.48235128, 0.57842162, 0.50297967],
       [0.98173552, 0.80423981, 0.4047245 , 0.85172597, 0.32276178],
       [0.85301294, 0.26888055, 0.14672257, 0.67893162, 0.06198328],
       [0.80185088, 0.52359397, 0.35513698, 0.84171051, 0.34803395],
       [0.15082236, 0.20827871, 0.70286441, 0.46469883, 0.18069756]])

------------------------------------------------------------------------------------
numpy.random.randint fonksiyonu [low, high) aralığında rastgele tamsayı 
değerlerinden oluşan NumPy dizisi oluşturmaktadır
                                 
a = np.random.randint(10, 20, (5, 5), dtype='int8')
# array([[15, 14, 12, 13, 14],
       [16, 13, 13, 10, 12],
       [12, 15, 15, 15, 19],
       [18, 13, 13, 14, 13],
       [14, 12, 18, 19, 14]], dtype=int8)
------------------------------------------------------------------------------------
"""

# ---------- arange ------------
"""
------------------------------------------------------------------------------------
arange fonksiyonu Python'ın built-in range fonksiyonuna benzemektedir. Ancak arange 
bize dolaşılabilir bir nesne vermez. Doğrudan bir NumPy dizisi verir. start, stop, 
step parametreleri range fonksiyonunda olduğu gibidir. Ancak Python range fonksiyonunda 
start, stop ve step değerleri int türünden olmak zorundayken arange fonksiyonunda 
float türünden de olabilir. Böylelikle biz arange ile noktasal artırımlarla bir 
numpy dizisi oluşturabiliriz. arange fonksiyonu dtype parametresi de alabilmektedir. 

Ancak bu fonksiyon her zaman tek boyutlu bir diziyi bize verir.

------------------------------------------------------------------------------------
import numpy  as np

a = np.arange(-1, 1, 1)
# array([-1,  0])

a = np.arange(-1, 1, 0.25)
#  array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75])

------------------------------------------------------------------------------------
arange fonksiyonunu kullanırken dikkat etmek gerekir. Çünkü noktasal artırımlar, 
noktasal start ve stop değerleri yuvarlama hatalarından dolayı beklenenden fazla 
ya da az sayıda eleman üretebilir. (Örneğin 0.1 artırımlarla ilerleken yuvarlama 
hatasından dolayı stop değerine çok yakın ama ondan küçük değer elde edilebilir 
ve bu değer de dizi içinde bulunabilir.) Zaten Python'daki built-in range sınıfının 
tamsayı değerler almasının nedeni de budur. 

------------------------------------------------------------------------------------
"""

# ---------- linspace ------------
"""
------------------------------------------------------------------------------------
arange fonksiyonun yukarıda belirtilen probleminden dolayı noktasal artırım için 
genellikle programcılar linspace fonksiyonunu tercih ederler. Bu fonksiyon start, 
stop ve num parametrelerine sahiptir. Fonksiyon her zaman start ve stop değerlerini 
de içerecek biçimde eşit aralıklı num tane değeri oluşturarak onu bir NumPy dizisi 
olarak vermektedir. linspace ile elde edilecek eleman sayısı belli olduğu için 
arange fonksiyonu yerine genellikle programcılar bunu tercih etmektedir. linspace 
fonksiyonun dtype parametresi de vardır. Bu parametre için argüman girilmezse 
default detype np.float64 olacak biçimde belirlenir. 

import numpy  as np

a = np.linspace(0, 10, 10)
# array([ 0.        ,  1.11111111,  2.22222222,  3.33333333,  4.44444444,
        5.55555556,  6.66666667,  7.77777778,  8.88888889, 10.        ])
------------------------------------------------------------------------------------
"""

# ---------- shape ------------
"""
------------------------------------------------------------------------------------
Bir NumPy dizisinin boyutlarını (yani kaça kaçlık olduğunu) shape isimli özniteliği 
ile elde edebiliriz. shape özniteliği bize boyutları belirten bir demet vermektedir.
Örneğin:

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a.shape) # (3,4)
------------------------------------------------------------------------------------
"""

# ---------- reshape ------------
"""
------------------------------------------------------------------------------------
Bir NumPy dizisinin boyutları değiştirilebilir. Bunun için ndarray sınıfının 
reshape metodu ya da reshape fonksiyonu kullanılabilemktedir. Default durumda 
ndarray elemanları C Programlama Dilindeki gibi satırsal biçimde belleğe tek 
boyutlu olarak yerleştirilmektedir.

Örneğin aşağıdaki gibi bir NumPy dizisi bulunuyor olsun:

1 2 3 4 
5 6 7 8 

Bu aslında bu dizi bellekte şu sırada tek boyutlu biçimde tutulmaktadır:

1 2 3 4 5 6 7 8 

Şimdi biz bu diziyi 4x2 olarak reshape yaparsak dizi şu hale gelir:

1 2
3 4
5 6
7 8

Yani reshape işlemini şöyle düşünmelisiniz: Sanki önce çok boyutlu dizi tek boyuta 
dönüştürülüp, yeniden diğer boyuta dönüştürülmektedir. 

reshape işleminden yeni bir NumPy dizisi elde edilmektedir.
------------------------------------------------------------------------------------

import numpy  as np
a = np.random.randint(0, 100, (5, 4))
print(a)
print()

b = a.reshape(2,10)
print(b)

------------------------------------------------------------------------------------
Çok boyutlu dizilerin tek boyutlu hale getirilmesi çokça gereksinim duyulan bir 
işlemdir. Örneğin elimizde 5x4'lük bir NumPy dizisi olsun. Biz bunu tek boyutlu 
bir dizi haline getirmeye çalışalım. Tabii bu işlemi reshap metoduyla ya da 
fonksiyonuyla yapabiliriz. Örneğin;

a = np.random.randint(0, 100, (5, 4))
b = a.reshape(-1)                       # a.reshape(20)

Ancak bu işlem için ravel isimli bir metot ve global bir fonksiyon da 
bulundurulmuştur. ravel bize reshape işleminde olduğu gibi bir view nesnesi 
vermektedir. Örneğin:

c = a.ravel()

------------------------------------------------------------------------------------
"""

# ---------- slicing (dilimleme)------------
"""
------------------------------------------------------------------------------------
NumPy dizileri üzerinde dilimleme (slicing) yapılabilir. Dilimleme işleminde tamamen
Python listelerindeki semantik uygulanmaktadır. Dilimleme her boyut için ayrı ayrı 
yapılabilmektedir. Dilimleme işleminden yeni bir NumPy dizisi view nesnesi olarak 
elde edilmektedir. Dilimleme ilk boyuttan başlanarak boyut gerçekleştirilmektedir.

import numpy as np

a = np.random.randint(102,357, (7,7), dtype= 'int32')
print(a)
print()

print(a[1,2])
print()

print(a[2])
print()

print(a[2, 3:7])
print()

print(a[:, 2:5])
print()


a[:, 2:5] = 0
print(a)
print()
------------------------------------------------------------------------------------
"""

# bool indeksleme 
"""
------------------------------------------------------------------------------------
Bir NumPy dizisine bool indeksleme uygulanabilir. bool indeksleme için dizi uzunluğu 
ve boyutu kadar bool türden dolaşılabilir bir nesne girilir. Bu dolaşılabilir 
nesnedeki True olan elemanlara karşı gelen dizi elemanları elde edilmektedir.

import numpy as np
a = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ])

b = [True, False, True, False, True, True, False, True, True]

c = a[b]

print(c)
------------------------------------------------------------------------------------
"""

# Matris çarpımı
"""
------------------------------------------------------------------------------------
import numpy as np

a = np.random.randint(1,23,(5,5))
b = np.random.randint(1,23,(5,5))
print(a)
print()
print(b)
print()

print(a@b)
print()

print(np.matmul(a,b)) 
------------------------------------------------------------------------------------
"""


#  ----------------------------- Pandas  -----------------------------

"""
------------------------------------------------------------------------------------
Pandas kütüphanesi NumPy kütüphanesinin üzerine kurulmuş durumdadır. Yani seviye 
bakımından NumPy'dan biraz daha yüksek seviyededir. 

Pandas'ta sütunlardan ve satırlardan oluşan veri kümeleri DataFrame isimli bir 
sınıf ile temsil edilmektedir. Veri kümesindeki belli bir sütun ise Series isimli 
sınıfla temsil edilir. DataFrame sınıfını Series nesnelerini tutan bir sınıf 
olarak düşünebiliriz. 

------------------------------------------------------------------------------------
Series nesnelerinin de tıpkı NumPy dizilerinde olduğu gibi bir dtype türü vardır. 
Bir Series nesnesi yaratılırken nesnenin dtype türü dtype parametresiyle belirtilebilir. 
Pandas içerisinde ayrı dtype sınıfları yoktur. Aslında Pandas Series bilgilerini 
NumPy dizisi olarak saklamaktadır. Dolayısıyla Series nesnesi yaratılırken dtype 
bilgisi NumPy dtype türü olarak belirtilir. Örneğin:

import pandas as pd
s = pd.Series([10, 20, 30, 40, 50], dtype=np.float32)
print(s)

!!! Series nesneleri MUTABLE nesnelerdir. Değiştirilebilir.

------------------------------------------------------------------------------------
Series nesnelerinin elemanlarına erişmek için üç yol vardır. Series nesnesi s olmak üzere:

1) Doğrudan köşeli parantez operatörü ile. Yani s[...] biçiminde.
2) loc örnek özniteliği ve köşeli parantez operatörü ile. Yani s.loc[...] biçiminde
3) iloc örnek özniteliği ve köşeli parantez operatörü ile. Yani s.iloc[...] biçiminde

Seeries nesneleri "değiştirilebilir (mutable)" nesnelerdir. Bir Series nesnesine 
erişip onu değiştirebiliriz. 

Series nesnesinin index ile belirtilen (ya da index belirten değerlerine) 
"etiket (label)" da denilmektedir. Örneğin:

------------------------------------------------------------------------------------
s.iloc[...] biçimindeki erişimde ise köşeli parantez içerisine her zaman sıra 
numarası yerleştirilmek zorundadır. Biz burada köşeli parantez içerisine etiket 
yerleştiremeyiz.
    
Yani loc örnek özniteliği ile erişimde köşeli parantez içerisinde her zaman etiket 
bulundurulması gerekmektedir. Bu bakımdan s[...] erişimi ile s.loc[...] erişimi
birbirine benzemektedir fakat aralarında farklılık vardır.     

------------------------------------------------------------------------------------
Doğrudan indekslemede hem etiket hem de sıra numarası bir arada kullanılamz. 
Ancak bunlardan biri kullanılabilir. Örneğin:

import pandas as pd

s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'], dtype='float32')
k = s[['b', 'a', 'e']]
print(k)
print()

k= s.iloc[[1, 3, 2]]
print(k)

------------------------------------------------------------------------------------
Series nesnesinin içerisindeki değerler values isimli örnek özniteliği ile bir 
NumPy dizisi olarak elde edilebilmektedir. Aslında Series nesnesi zaten değerleri 
NumPy dizisi içerisinde tutmaktadır. values elemanı da bize doğrudan aslında bu 
diziyi verir. Bu dizide değişiklik yaptığımızda Series nesnesinin elemanında 
değişiklik yapmış oluruz. Örneğin:

s = pd.Series([10, 20, 30, 40, 50], dtype='float32')
a = s.values
print(a)

Series sınıfının to_numpy metodu values örnek özniteliği gibidir. Ancak to_numpy 
değişik seçeneklere de sahiptir. Default durumda to_numpy metodu ile values örnek 
özniteliği aynı Numpy dizisini vermektedir. Fakat örneğin to_numpy metodunda 
copy=True geçilirse metot bize kopyalama yaparak başka bir Numpy dizisi verir. 

s = pd.Series([10, 20, 30, 40, 50], dtype='float32')

a = s.values
print(id(a))
print(a)

a = s.to_numpy()
print(id(a))
print(a)

print('-----')

a = s.to_numpy(copy=True)
print(a)
print(id(a))

Özetle bir Series nesnesi içerisindeki değerleri NumPy dizisi olarak almak istersek 
values örnek özniteliğini ya da to_numpy metodunu kullanabiliriz.

a = s.to_numpy()
print(a)


Eğer Series nesnesi içerisindeki değerleri bir Python listesi biçiminde elde etmek 
istersek Series sınıfının to_list metodunu kullanabiliriz. 

Tabii to_list her çağrıldığında aslında bize farklı bir list nesnesi verecektir.

------------------------------------------------------------------------------------
Series nesnesinden eleman silmek için Series sınıfının drop metodu kullanılmaktadır 
(drop isimli bir fonksiyon yoktur). Bu metot her zaman etiket temelinde çalışır. 
Hiçbir zaman sıra numarasıyla çalışmaz. Biz tek bir etiket de kullanabiliriz. Bir 
grup etiketi dolaşılabilir bir nesne biçiminde de metoda verebiliriz. Metot default 
durumda "inplace" silme işlemi yapmaz. Bize silinmiş yeni bir Series nesnesi verir. 
Örneğin:
        
s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'], dtype='float32')
print(s)
result = s.drop(['a', 'e'])
print()
print(result)


Metodun inplace parametresi True geçilirse silme işlemi nesne üzerinde yapılır. 
Bu durumda metot None değerine geri döner.

s.drop(['a', 'e'], inplace=True)
print(s)            

------------------------------------------------------------------------------------
Series sınıfının pek çok faydalı metodu vardır. Bu metotlar bize yaptıklar işlem 
sonucunda yeni bir Series nesnesi verirler. Aslında bu metotlar NumPy metotlarına 
çok benzemektedir. NumPy'da pek çok işlem hem metotlarla hem de fonksiyonlarla 
yapılabilmektedir. Ancak Pandas'ta ağırlıklı olarak metotlar bulundurulmuştur. 
Yani pek çok fonksiyon yalnızca metot biçiminde bulundurulmuştur.

abs metodu elemanların mutlak değerlerini elde eder. add metodu karşılıklı elemanları 
toplar (yani + operatörü ile yapılanı yapar). argmax, argmin, argsort metotoları 
sırasıyla en büyük elemanın indeksini, en küçük elemanın indeksini ve sort 
edilme durumundaki indeksleri vermektedir.

s = pd.Series([12, 8, -4, 2, 9], dtype='float32')

print(s.abs())
print('-----')

print(s.argmin())
print(s.argmax())
print('-----')

print(s.argsort()) 
print('-----')
print(s[s.argsort()])

------------------------------------------------------------------------------------
dropna metodu eksik verileri atmak için kullanılmaktadır. Yani NaN değerleri 
Series nesnesinden silinir.

s = pd.Series([3, None, 7, 9, None, 10], dtype='float32')
print(s)
print()
result = s.dropna()
print(result)
print()

fillna isimli metot eksik verileri (yani NaN olan elemanları) spesifik bir değerle 
doldurmaktadır. Örneğin biz eksik verileri aşağıdaki gibi ortalamayla doldurabiliriz:

reul= s.fillna(s.mean())
print(reul)        
------------------------------------------------------------------------------------
"""

# DataFrame
"""
------------------------------------------------------------------------------------
Pandas'taki en önemli veri yapısı DataFrame denilen veri yapısıdır. DataFrame tipik 
olarak istatistiksel veri kümesini temsil etmek için düşünülmüştür. DataFrame 
nesnesinin sütunlardan oluşan matriksel bir yapısı vardır. Aslında DataFrame nesnesi 
Series nesnelerinden oluşmaktadır. Yani DataFrame nesnelerinin sütunları Series 
nesneleridir. 

NumPy dizilerinin elemanları aynı türden olur. Her ne kadar elemanları aynı türden 
olmayan NumPy dizileri de oluşturulabiliyorsa da (örneğin dtype='object' diyerek) 
bu biçimde NumPy dizilerinin uygulamada kullanımı yoktur. O halde Pandas kütüphanesi 
aslında sütunları farklı türlerden olabilen DataFrame denilen bir veri yapısı 
sunmaktadır. İstatistik ve veri bilimindeki "veri kümeleri (datasets)" ham durumda 
böyle bir yapıya sahiptir. Bu bakımdan veri kümeleri veritabanlarındaki tablolara 
da benzetilebilir.

DataFrame nesnesi iki boyutlu bir Python listesi ile oluşturulabilir. Eğer index 
parametresi ve columns parametresi belirtilmezse oluşturulan DataFrame nesnesinin 
satır etiketleri ve sütun etiketleri 0, 1, 2, ... biçiminde atanır

------------------------------------------------------------------------------------
import pandas as pd
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'b', 'c'], columns=['x', 'y', 'z'])
print(df)

------------------------------------------------------------------------------------
Bir DataFrame nesnesi iki boyutlu bir NumPy dizisi ile de yaratılabilir. Örneğin:

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
df = pd.DataFrame(a, columns=['a', 'b', 'c'], index= [3,7,4])
print(df)
print()

print(df.loc[7]) # df.loc[1] -> indexError
print()

print(df.iloc[1]) # df.iloc[7] -> indexError

------------------------------------------------------------------------------------
DataFrame nesnesi bir sözlük ile de yaratılabilir. Bu durumda sözlüğün anahtarları 
sütun isimlerini, değerleri de sütunlardaki değerleri belirtir. Örneğin:
    
d = {'Adı Soyadı': ['Kaan Aslan', 'Ali Serçe', 'Ayşe Er'], 'Boy': [182, 174, 168], 'Kilo': [78, 69, 56]}
df = pd.DataFrame(d)
print(df)

------------------------------------------------------------------------------------
DataFrame üzerinde bir sütun insert etmek için DataFrame sınıfının insert metodu 
metodu kullanılabilmektedir. insert metodunun birinci parametresi her zaman insert 
edilecek sütunun indeks numarasını alır. İkinci parametre indeks edilecek sütunun 
ismini (ayni etiketini), üçüncü parametre ise sütun bilgilerini almaktadır. insert 
metodu "in-place" insert işlemi yapmaktadır. Yani DatFrame nesnesinin kendi 
üzerinde ekleme yapılmaktadır.

import pandas as pd
d = {'Adı': ['Ali', 'Veli', 'Selami', 'Ayşe', 'Fatma'], 'Kilo': [48.3, 56.7, 92.3, 65.3, 72.3], 'Boy': [172, 156, 182, 153, 171]}
df = pd.DataFrame(d)
print(df)
print('----------')

bmi = df['Kilo'] / (df['Boy'] / 100) ** 2
print(bmi)
print(type(bmi))
print('----------')

df.insert(3, 'Vücut Kitle Endeksi', bmi)
print(df)
------------------------------------------------------------------------------------
"""


#  ----------------------------- Statistics  -----------------------------

"""
------------------------------------------------------------------------------------
İstatistikte verilerin merkezine ilişkin bilgi veren ölçülere "merkezi eğilim ölçüleri 
(measures of central tendency)" denilmektedir. Merkezi eğilim ölçülerinin en yaygın 
kullanılanı "aritmetik ortalamadır". Aritmetik ortalama (mean) değerlerin toplanarak 
değer sayısına bölünmesiyle elde edilmektedir. 

Aritmetik ortalama hesaplamak için çeşitli kütüphanelerde çeşitli fonksiyonlar hazır 
olarak bulunmaktadır. Örneğin Python'ın standart kütüphanesindeki statistics modülünde 
bulunan mean fonksiyonu aritmetik ortalama hesaplamaktadır. 

import statistics
a = [1, 2, 7, 8, 1, 5]
statistics.mean(a)

axis=0  -> satır
axis=1  -> sutun

mean fonksiyonu herhangi bir dolaşılabilir nesneyi parametre olarak alabilmektedir. 

NumPy kütüphanesindeki mean fonksiyonu axis temelinde (yani satırsal ve sütunsal biçimde) 
ortalama hesaplayabilmektedir. Örneğin:

import numpy as np
a = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
np.mean(a, axis=0)
# array([4.66666667, 5.66666667, 6.66666667])


Pandas kütüphanesinde Series ve DataFrame sınıflarının mean metotları aritmetik 
ortalama hesabı yapmaktadır. DataFrame sınıfının mean metodunda default axis 0 
biçimindedir. Yani sütunsal ortalamalar elde edilmektedir. Örneğin:

import pandas as pd
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df)
print()
print(df.mean())
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Diğer bir merkezi eğilim ölçüsü de "medyan (median)" denilen ölçüdür. Medyan sayıların 
küçükten büyüğe sıraya dizildiğinde ortadaki değerdir. Ancak sayılar çift sayıda 
ise sayıların tam ortasında bir değer olmadığı için ortadaki iki değerin aritmetik 
ortalaması medyan olarak alınmaktadır. Medyan işlemi uç değerlerden (outliers) 
etkilenmez. Ancak medyan işlemi aritmetik ortalamadan daha fazla zaman alan bir 
işlemdir. Çünkü medyan için önce değerlerini sıraya dizilmesi gerekmektedir. 
Dolayısıyla  medyan işlemi O(N log N) karmaşıklıta bir işlemdir.

import statistics
a = [1, 23, 56, 12, 45, 21]
statistics.median(a)

NumPy kütüphanesinde medyan işlemi eksensel biçimde median fonksiyonuyla yapılabilmektedir.

import numpy as np
a = np.random.randint(1, 100, (10, 10))

np.median(a, axis=0)
# array([33.5, 56.5, 69. , 48. , 57.5, 39. , 54.5, 32. , 61.5, 40.5])


df = pd.DataFrame(a)
df.median()

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Merkezi eğilim ölçülerinin bir diğeri de "mod" denilen ölçüdür. Bir grup verideki 
en çok yinelenen değere "mod (mode)" denilmektedir. Mod özellikle kategorik ve 
sıralı ölçeklerde ortalamanın yerini tutan bir işlem olarak kullanılmaktadır. 
Mod işlemi genel olarak O(N log N) karmaşıklıkta yapılabilmektedir. (Tipik mod 
algoritmasında değerler önce sıraya dzilir. Sonra yan yana aynı değerlerden kaç 
tane olduğu tespit edilir.)

import statistics
a = [1, 3, 3, 4, 2, 2, 5, 2, 7, 9, 5, 3, 5, 7, 5]
statistics.mode(a)

NumPy kütüphanesinde mod işlemini yapan bir fonksiyon bulunmamaktadır. Ancak SciPy 
kütüphanesinde mod işlemi için stats modülü içerisindeki mode fonksiyonu kullanılabilir. 
Bu fonksiyon yine eksensel işlemler yapabilmektedir. mode fonksiyonu ModeResult 
isimli bir sınıf türünden tuple sınıfından türetilen bir sınıf türünden bir nesne 
verir. Bu sınııfın mode ve count örnek öznitelikleri en çok yinelenen değerleri ve 
onların sayılarını bize vermektedir. ModeResult sınıfı bir çeşit demet özelliği 
gösterdiği için demet gibi de kullanılabilir. 

import scipy.stats
import numpy as np

a = np.random.randint(1, 10, (20, 10))
mr = scipy.stats.mode(a, axis=0)
print(type(mr))  # <class 'scipy.stats._stats_py.ModeResult'>

mr.mode
#  array([[4, 1, 2, 3, 5, 8, 4, 6, 8, 1]])

mr.count
# array([[4, 4, 6, 4, 4, 4, 4, 4, 5, 4]])



Pandas kütüphanesinde Series ve DataFrame sınıflarının mode metotları da mode 
işlemi yapmaktadır.

import pandas as pd
a = np.random.randint(1, 10, (2, 10))
df = pd.DataFrame(a)
df.mode()

DataFrame sınıfının mode metodu bize bir DataFrame nesnesi vermektedir. Uygulamacı 
genellikle bunun ilk satırı ile ilgilenir. Diğer satırlar eşit miktarda tekrarlanan 
elemanlardan oluşmaktadır. Tabii belli bir sütunda eşit miktarda tekrarlanan elemanların
sayısı az ise artık geri döndürülen DataFrame'in o sütuna ilişkin satırlarında 
NaN değeri bulunacaktır.

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Değerlerin merkezine ilişkin bilgiler dağılım hakkında iyi bir fikir vermeyebilir. 
Örneğin iki ülkede kişi başına düşen ortalama yıllık gelir (gayri safi milli hasıla) 
15000 dolar olabilir. Ancak bu iki ülke arasında gelir dağılımında önemli farklılıklar 
bulunuyor olabilir. O halde değerlerin ortalamasının yanı sıra onların ortalamaya 
göre nasıl yayıldıkları da önemlidir. İstatistikte değerlerin ortalamaya göre 
yayılımı için "merkezi yayılım ölçüleri (measures of dispersion)" denilen bazı 
ölçüler kullanılmaktadır. Merkezi yayılım ölçüleri aslında değerlerin ortalamadan 
ortalama uzaklığını belirlemeyi hedeflemektedir

Merkezi yayılım ölçüsü olarak "değerlerin ortalamadan ortalama mutlak uzaklığına" 
başvurulabilir. Burada mutlak değer alınmasının nedeni uzaklıkları negatif olmaktan 
kurmak içinidir. Bu durumda ortalama 0 çıkmaz. Ancak bu yöntem de aslında çok iyi 
bir yöntem değildir. Aşağıda aynı dilimin ortalamadan ortalama mutlak uzaklığı 
hesaplanmışır.

import numpy as np

a = np.array([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
mean = np.mean(a)
print(mean)                # 5

result = np.mean(np.abs(a - mean))
print(result)               # 2.5

------------------------------------------------------------------------------------
Aslında ortalamadan ortalama uzaklık için "standart sapma (standard deviation)" 
denilen ölçü tercih edilmektedir. Standart sapmada ortalamadan uzaklıkların mutlak 
değeri değil kareleri alınarak negatiflikten kurtulunmaktadır. Kare alma işlemi 
değerleri daha fazla farklılaştırmaktadır. Yani aynı değerlerin oluşma olasılığı 
kare alma sayesinde daha azalmaktadır. Aynı zamanda bu işlem bazı durumlarda başka 
faydalara da yol açmaktadır. (Örneğin ileride bu kare alma işlemlerinin optimizasyon
problemlerinde uygun bir işlem olduğunu göreceğiz.)

import numpy as np

def sd(a, ddof = 0):
    return np.sqrt(np.sum((a - np.mean(a)) ** 2)  / (len(a)  - ddof))

a = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]
result = sd(a)
print(result)           # 2.7688746209726918

------------------------------------------------------------------------------------
Python'ın Standart Kütüphanesinde statistics modülü içerisinde standart sapma hesabı 
yapan stdev ve pstdev fonksiyonları bulunmaktadır. stdev fonksiyonu (n - 1)'e bölme 
yaparken, pstdev  (buradaki 'p' harfi "population" sözcüğünden gelmektedir) fonksiyonu 
n'e bölme yapmaktadır. 

import statistics

a = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]

std = statistics.stdev(a)
print(std)                          # 2.891995221924885  

std = statistics.pstdev(a)          # 2.7688746209726918
print(std) 

------------------------------------------------------------------------------------
NumPy kütüphanesinde std isimli fonksiyon eksensel standart sapma hesaplayabilmektedir. 
Fonksiyonun ddof parametresi default durumda 0'dır. Yani default durumda fonksiyon 
n'e bölme yapmaktadır.

import numpy as np

a = np.array([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
result = np.std(a, ddof=0)
print(result)                       # 2.7688746209726918 

------------------------------------------------------------------------------------
Pandas kütüphanesinde de Series ve DataFrame sınıflarının std isimli metotları
eksensel standart sapma hesabı yapabilmektedir. Ancak bu metotlarda ddof parametresi 
default 1 durumundadır. Yani bu metotlar default durumda (n - 1)'e bölme yapmaktadır.

import pandas as pd

s = pd.Series([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
result = s.std()
print(result)                       # 2.891995221924885        

s = pd.Series([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
result = s.std(ddof=0)
print(result)                       # 2.7688746209726918 

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Standart sapmanın karesine "varyans (variance)" denilmektedir. Varyans işlemi standart 
kütüphanedeki statistics modülünde bulunan variance ve pvariance fonksiyonlarıyla 
yapılmaktadır. NumPy kütüphanesinde varyans işlemi var fonksiyonuyla ya da ndarray 
sınıfının var metoduyla, Pandas kütüphenesinin Series ve DataFrame sınıflarındaki 
var metoduyla yapılmaktadır. Yine NumPy'daki variance fonksiyonundaki ddof default 
olarak 0, Pandas'taki ddof ise 1'dir.

Pekiyi neden standart sapma varken ayrıca onun karesi için varyans terimi uydurulmuştur? 
İşte istatistikte pek çok durumda aslında doğrudan ortalamadan farkların karesel 
ortalamaları (yani standart sapmanın karesi) kullanılmaktadır. Bu nedenle bu hesaba 
ayrı bir isim verilerek anlatımlar kolaylaştırılmıştır.


import statistics
import numpy as np
import pandas as pd

data = [1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8]

result = statistics.pvariance(data)
print(result)               # 7.666666666666667

result = np.var(data)
print(result)               # 7.666666666666667

s = pd.Series(data)
result = s.var(ddof=0)
print(result)               # 7.666666666666667

------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------
Bir deney sonucunda oluşacak durum baştan tam olarak belirlenemiyorsa böyle deneylere 
"rassal deney (random experiment)" denilmektedir. Örneğin bir paranın atılması 
deneyinde para "yazı" ya da "tura" gelebilir. O halde "paranın atılması" rassal 
bir deneydir. Benzer biçimde bir zarın atılması, bir at yarışında hangi atın birinci 
olacağı gibi deneyler rassal deneylerdir. 

Bir deneyin sonucu öndecen bilinebiliyorsa bu tür deneylere "deterministik deneyler" 
de denilmektedir. 

Bir rassal deney sonucunda oluşabilecek tüm olası durumların kümesine "örnek uzayı 
(sample space)" denilmektedir.

Olasılığın (probablity) değişik tanımları yapılabilmektedir. Olasılığın en yaygın 
tanımlarından birisi ------>>  "göreli sıklık (relative frequency)" tanımıdır. 

Bu tanıma göre bir rassal olay çok sayıda yinelendikçe elde edilen olasılık değeri 
belli bir değere yakınsamaya başlar. Örneğin bir paranın 100 kere atılmasında 
50 kere yazı 50 tura gelmeyebilir. 

Ancak para sonsuz sayıda atılırsa (ya da çok fazla sayıda atılırsa) tura gelme 
sayısının paranın atılma sayısına oranı 0.5'e yakınsayacaktır. 

Buna istatistike ---->> "büyük sayılar yasası (law of large numbers)" da denilmektedir. 

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
Olasılıkta ve istatistikte en çok kullanılan temel kavramlardan biri "rassal değişken 
(random variable)" denilen kavramdır. Her ne kadar "rassal değişken" isminde bir 
"değişken" geçiyorsa da aslında rassal değişken bir fonksiyon belirtmektedir. 
Rassal değişken bir rassal deney ile ilgilidir. Bir rassal deneyde örnek uzayın 
her bir elemanını (yani basit olayını) reel bir değere eşleyen bir fonksiyon 
belirtmektedir. Rassal değişkenler genellikle "sözel biçimde" ifade edilirler. 
Ancak bir fonksiyon belirtirler. Rassal değişkenler matematiksel gösterimlerde 
genellikle büyük harflerle belirtilmektedir. Örneğin:

- R rassal değişkeni "iki zar atıldığında zarların üzerindeki sayıların toplamını"
belirtiyor olsun. Burada aslında R bir fonksiyondur. Örnek uzayın her bir elemanını 
bir değere eşlemektedir. Matematiksel gösterimle R rassal değişkeni şöyle belirtilebilir:

R: S -> R

Burada R'nin  örnek uzayından R'ye bir fonksiyon belirttiği anlaşılmalıdır. Burada 
R fonksiyonu aşağıdaki gibi eşleme yapmaktadır:

(1, 1) -> 2
(1, 2) -> 3
(1, 3) -> 4
...
(6, 5) -> 11
(6, 6) -> 12

K rassal değişkeni "rastgele seçilen bir kişinin kilosunu belirtiyor" olsun. Bu 
durumda örnek uzayı aslında dünyaki tüm insanlardır. Burada K fonksiyonu da her 
insanı onun kilosuna eşleyen bir fonksiyondur. 

Rassal değişkenler kümeler üzerinde işlemler yapmak yerine gerçek sayılar üzerinde 
işlem yapmamızı sağlayan, anlatımlarda ve gösterimlerde kolaylık sağlayan bir kavramdır. 

Rassal değişkenler tıpkı matematiksel diğer fonksiyonlarda olduğu gibi "kesikli 
(discrete)" ya da "sürekli (continuous)" olabilmektedir. eğer bir rassal değişken 
(yani fonksiyon) teorik olarak belli bir aralıkta tüm gerçek sayı değerlerini 
alabiliyorsa böyle rassal değişkenlere "sürekli (continous)" rassal değişkenler 
denilmektedir. Ancak bir rassal değişken belli bir aralıkta yalnızca belli gerçek 
sayı değerlerini alabiliyorsa bu rassal değişkenlere "kesikli (discrete)" rassal
değişkenler denilmektedir.

Örneğin "iki zarın atılmasında üste gelen sayılar toplamını belirten R rassal 
değişkeni" kesiklidir. Çünkü yalnızca belli değerleri alabilmektedir. Ancak 
"rastgele seçilen bir kişinin kilosunu belirten" K rassal değişkeni süreklidir. 
Çünkü teorik olarak belli bir aralıkta tüm gerçek değerleri alabilir.(Biz kişilerin 
kilolarını yuvarlayarak ifade etmekteyiz. Ancak aslında onların kiloları belli 
aralıktaki tüm gerçek değerlerden biri olabilir.)

------------------------------------------------------------------------------------
Yapay zeka ve makine öğrenmesinde sürekli rassal değişkenler daha fazla karşımıza 
çıkmaktadır. Bu nedenle biz sürekli rassal değişkenler ve onların olasılıkları 
üzerinde biraz daha duracağız. 

Sürekli bir rassal değişkenin aralıksal olasılıklarını hesaplama aslında bir 
"intergral" hesabı akla getirmektedir. İşte sürekli rassal değişkenlrin aralıksal 
olasılıklarının hesaplanması için kullanılan fonksiyonlara -----> 

"olasılık yoğunluk fonksiyonları (probability density functions)" denilmektedir. 

Birisi bize bir rassal değişkenin belli bir aralıktaki olasılığını soruyorsa o 
kişiin bize o rassal değişkene ilişkin "olasılık yoğunluk fonksiyonunu" vermiş 
olması gerekir. Biz de örneğin P{x0 < X < x1} olasılığını x0'dan x1'e f(x)'in 
integrali ile elde ederiz. 

Bir fonksiyonun olasılık yoğunluk fonksiyonu olabilmesi için -sonsuzdan + sonsuza 
integralinin (yani tüm eğri altında kalan alanın) 1 olması gerekir. Bir rassal 
değişkenin olasılık yoğunluk fonksiyonuna "o rassal değişkenin dağılımı" da 
denilmektedir.

------------------------------------------------------------------------------------
Değişik ortalama ve standart sapmaya ilişkin sonsuz sayıda Gauss eğrisi çizilebilir. 
Ortalaması 0, standart sapması 1 olan normal dağılıma "standart normal dağılım" da 
denilmektedir. Genellikle istatistiktre standart normal dağılımdaki X değerlerine
"Z değerleri" denilmektedir. Aşağıdaki örnekte Gauss eğrisi çizdirilmiştir.

import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu = 0, std = 1):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((x - mu) / std) ** 2)

# yukarıdaki formül -->  'probability denstiy function'    

x = np.linspace(-5, 5, 1000)
y = gauss(x)

plt.title('Gauss Function')
plt.plot(x, y)
plt.show()

------------------------------------------------------------------------------------
Yukarıdaki çizimde eksenleri kartezyen koordinat sistemindeki gibi de gösterbiliriz. 
 
import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu = 0, std = 1):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((x - mu) / std) ** 2)
  
    
def draw_gauss(mu = 0, std = 1):
    x = np.linspace(-5 * std + mu, 5 * std + mu, 1000)
    y = gauss(x, mu, std)
    
    mu_y = gauss(mu, mu, std)
    
    plt.figure(figsize=(10, 4))
    plt.title('Gauss Function', pad=10, fontweight='bold')
    axis = plt.gca()
    
    axis.set_ylim([-mu_y * 1.1, mu_y * 1.1])
    axis.set_xlim([-5 * std + mu, 5 * std + mu])
    axis.set_xticks(np.arange(-4 * std + mu, 5 * std + mu, std))
    # axis.set_yticks(np.round(np.arange(-mu_y, mu_y, mu_y / 10), 2))
    axis.spines['left'].set_position('center')
    axis.spines['top'].set_color(None)
    axis.spines['bottom'].set_position('center')
    axis.spines['right'].set_color(None)
    axis.plot(x, y)
    plt.show()

draw_gauss(100, 15)

------------------------------------------------------------------------------------
Normal dağılımda eğri altında kalan toplam alanın 1 olduğunu belirtmiştik. Bu dağılımda 
toplaşmanın ortalama civarında olduğu eğirinin şeklinden anlaşılmaktadır. Gerçektende 
normal dağılımda ortalamadan bir standart sapma soldan ve sağdan kaplanan alan 
yani P{mu - std < X < mu + std} olasılığı 0.6827, ortalamadna iki standart sapma 
soldan ve sağdan kaplanan alan yani P{mu - std * 2< X < mu + std * 2} olasılığı 
0.9545 biçimindedir. 

Matplotlib'te bir eğrinin altındaki alanı boyamak için fill_between isimli fonksiyon 
kullanılmaktadır. Bu fonksiyon axis sınıfının bir metodu olarak da bulundurulmuştur. 
Aşağıdaki örnekte eğrinin altındaki belli bir alan fill_between metodu ile boyanmıştır.

import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu = 0, std = 1):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((x - mu) / std) ** 2)
      
def draw_gauss(mu = 0, std = 1, fstart= 0, fstop = 0):
    x = np.linspace(-5 * std + mu, 5 * std + mu, 1000)
    y = gauss(x, mu, std)
    
    mu_y = gauss(mu, mu, std)
    
    plt.figure(figsize=(10, 4))
    plt.title('Gauss Function', pad=10, fontweight='bold')
    axis = plt.gca()
    
    axis.set_ylim([-mu_y * 1.1, mu_y * 1.1])
    axis.set_xlim([-5 * std + mu, 5 * std + mu])
    axis.set_xticks(np.arange(-4 * std + mu, 5 * std + mu, std))
   # axis.set_yticks(np.round(np.arange(-mu_y, mu_y, mu_y / 10), 2))
    axis.spines['left'].set_position('center')
    axis.spines['top'].set_color(None)
    axis.spines['bottom'].set_position('center')
    axis.spines['right'].set_color(None)
    axis.plot(x, y)
    
    x = np.linspace(fstart, fstop, 1000)
    y = gauss(x, mu, std)
    axis.fill_between(x, y)
    plt.show()

draw_gauss(100, 15, 85, 115)

------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
Kümülatif dağılım fonksiyonu (cummulative distribution function) belli bir değere 
kadar tüm birikimli olasılıkları veren fonksiyondur. Genellikle F harfi gösterilmektedir. 
Örneğin F(x0) aslında P{X < x0} anlamına gelmektedir. Normal dağılımda F(x0) değeri 
aslında eğride X değerinin x0 olduğu noktadan soldaki tüm eğri altında kalan alanı 
belirtmektedir. (Başka bir deyişle sürekli dağılımlarda F(x0) değeri "-sonsuzdan 
x0'a kadar olasılık yoğunluk fonksiyonunun integraline eşittir)
------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
Normal dağılımla ilgili işlemleri yapabilmek için Python standart kütüphanesinde 
statistics modülü içerisinde NormalDist isimli bir sınıf bulundurulmuştur. Programcı 
bu sınıf türünden bir nesne yaratır. İşlemlerini bu sınıfın metotlarıyla yapar.
NormalDist nesnesi yaratılırken __init__ metodu için ortalama ve standart sapma 
değerleri girilir. (Bu değerler girilmezse ortalama için sıfır, standart sapma 
için 1 default değerleri kullanılmaktadır.)

------------------------------------------------------------------------------------
NormalDist sınıfının cdf (cummulative distribution function) isimli metodu verilen 
x değeri için eğrinin solunda kalan toplam alanı yani kümülatif olasılığı bize 
vermektedir. Örneğin standart normal dağılımda x = 0'ın solunda alan 0.5'tir.

import statistics

nd = statistics.NormalDist()

result = nd.cdf(0)
print(result)           # 0.5

------------------------------------------------------------------------------------
Örneğin biz ortalaması 100, standart sapması 15 olan bir normal dağılımda 
P{130 < x < 140} olasılığını aşağıdaki gibi elde edebiliriz:

nd = statistics.NormalDist(100, 15)
result = nd.cdf(140) - nd.cdf(130)
print(result)   # 0.018919751380589434

Şöyle bir soru sorulduğunu düşünelim: "İnsanların zekaları ortalaması 100, standart 
sapması 15 olan normal dağılıma uygundur. Bu durumda zeka puanı 140'ın yukarısında 
olanların toplumdaki yüzdesi nedir?". Bu soruda istenen şey aslında normal dağılımdaki
P{X > 140} olasılığıdır. Yani x ekseninde belli bir noktanın sağındaki kümülatif 
alan sorulmaktadır. Bu alanı veren doğrudan bir fonksiyon olmadığı için bu işlem 
1 - F(140) biçiminde ele alınarak sonuç elde edilebilir. Yani örneğin:

nd = statistics.NormalDist(100, 15)
result = 1 - nd.cdf(140)

------------------------------------------------------------------------------------
Belli bir kümülatif olasılık değeri için x değerinin bulunması işlemi de NormalDist 
sınıfının inv_cdf metoduyla yapılmaktadır. Örneğin standart normal dağılımda 0.99 
olan kümülatif olasılığın Z değeri aşağıdaki gibi bulunabilir:

nd = statistics.NormalDist()
result = nd.inv_cdf(0.99)
print(result)                       # 2.3263478740408408

result = nd.cdf(2.3263478740408408) # 0.99
print(result)

------------------------------------------------------------------------------------
Belli bir x değeri için Gauss fonksiyonunda ona karşı gelen y değeri sınıfın pdf 
metduyla elde edilmektedir. Örneğin x = 0 için standart normal dağılımda Gauss 
fonksiyonu değerini aşağıdaki gibi elde edebiliriz:

nd = statistics.NormalDist()
result = nd.pdf(0)
print(result)   # 0.3989422804014327

------------------------------------------------------------------------------------
!!!! Normal dağılmış !!!! rastgele n tane sayı üretmek için NormalDist sınıfının 
samples isimli metodu kullanılmaktadır. Bu metot bize bir liste olarak n tane 
float değer verir. Örneğin:

nd = statistics.NormalDist()
result = nd.samples(10)
print(result) 

Bu işlemden biz normal dağılmış 10 tane rasgele değerden oluşan bir liste elde ederiz. 

Biz normal dağılmış rastgele sayılardan histogram çizersek histogramımızın 
Gauss eğrisine benzemesi gerekir.


import statistics

nd = statistics.NormalDist()

result = nd.samples(10000)

import matplotlib.pyplot as plt

plt.hist(result, bins=30)
plt.show()

------------------------------------------------------------------------------------
Python'ın statistics modülündeki NormalDist sınıfı vektörel işlemler yapamamaktadır. 
Maalesef NumPy ve Pandas kütüphanelerinde normal dağılım üzerinde vektörel işlem 
yapan öğeler yoktur. Ancak SciPy kütüphanesi içerisinde pek çok dağılım üzerinde 
vektörel işlemler yapan sınıflar bulunmaktadır. Bu nedenle pratikte Python kütüphanesi 
yerine bu tür işlemler için SciPy kütüphanesi tercih edilmektedir. 
------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
scipy.stats modülü içerisindeki norm isimli singleton nesne normal dağılım üzerinde 
vektörel işlem yapan metotlara sahiptir. norm bir sınıf nesnesidir ve zaten yaratılmış 
bir biçimde bulunmaktadır. Dolayısıyla programcı doğrudan bu nesne ile ilgili sınıfın 
metotlarını çağırabilir. Genellikle programcılar bu tür nesneleri kullanmak için 
from import deyimini tercih ederler:

from scipy.stats import norm
 
norm nesnesine ilişkin sınıfın cdf isimli metodu üç parametre almaktadır:

cdf(x, loc=0, scale=1)

Buradaki x bir NumPy dizisi ya da Python dolaşılabilir nesnesi olabilir. Bu durumda 
tüm x değerlerinin kümülatif olasılıkları hesaplanıp bir NumPy dizisi olarak 
verilmektedir. Burada loc ortalamayı, scale ise standart sapmayı belirtmektedir. 
Örneğin:

from scipy.stats import norm
result = norm.cdf([100, 130, 140], 100, 15)
print(result)

------------------------------------------------------------------------------------
norm nesnesinin ilişkin olduğu sınıfın ppf (percentage point function) isimli metodu 
cdf işleminin tersini yapmaktadır. Yani kümülatif olasılığı bilindiği durumda bize 
bu kümalatif olasılığa karşı gelen x değerini verir. (Yani ppf NormalDist sınıfındaki 
inv_cdf metoduna karşılık gelmektedir.):

ppf(q, loc=0, scale=1)

from scipy.stats import norm

result = norm.ppf([0.50, 0.68, 0.95], 100, 15)
print(result)

ppf (percentage point function) ismi size biraz tuhaf gelebilir. Bu isim birikimli 
dağılım fonksiyonunun tersini belirtmek için kullanılmaktadır. ppf aslında 
"medyan (median)" kavramının genel biçimidir. Anımsanacağı gibi medyan ortadan ikiye 
bölen noktayı belirtiyordu. Örneğin standart normal dağılımda medyan 0'dır. 
Yani ortalamaya eşittir. 

İstatistikte tam ortadan bölen değil de diğer noktalardan bölen değerler için 
"percentage point" de denilmektedir. Örneğin normal dağılımda 1/4 noktasından bölen 
değer aslında birikimli dağılım fonksiyonunun 0.25 için değeridir. 

------------------------------------------------------------------------------------
norm nesnesinin ilişkin olduğu sınıfın pdf (probability density function) isimli 
metodu yine x değerlerinin Gaus eğrisindeki Y değerlerini vermektedir. Metodun 
parametrik yapısı şöyledir:

pdf(x, loc=0, scale=1) 

------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(40, 160, 1000)
y = norm.pdf(x, 100, 15)

plt.plot(x, y)

x = np.full(200, 100)       # 200 tane 100'lerden oluşan dizi
yend = norm.pdf(100, 100, 15)
y = np.linspace(0, yend, 200)
plt.plot(x, y, linestyle='--')

plt.show()

------------------------------------------------------------------------------------
norm nesnesinin ilişkin olduğu sınıfın rvs metodu ise normal dağılıma ilişkin rassal 
sayı üretmek için kullanılmaktadır. Metodun parametrik yapısı şöyledir:

    rvs(loc=0, scale=1, size=1)

statistics.NormalDist()'in sample metodunun bezeri

------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = norm.rvs(100, 15, 10000)

plt.hist(x, bins=20)
plt.show()

------------------------------------------------------------------------------------
Normal dağılımda ortalamadan birer standart sapma arasındaki bölgenin olasılığı, 
yani P{mu - sigma < x < mu + sigma} olasılığı 0.68 civarındadır.


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

result = norm.cdf(1) - norm.cdf(-1)
print(result)

x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

plt.title('Ortalamadan 1 Standart Sapma Arası Bölge', fontweight='bold')
axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(-4, 5))
axis.text(2.5, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-1, 1, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()

------------------------------------------------------------------------------------
Normal dağılımda ortalamadan iki standart sapma arasındaki bölgenin olasılığı, yani 
P{mu - 2 * sigma < x < mu + 2 * sigma} olasılığı 0.95 civarındadır.


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

result = norm.cdf(2) - norm.cdf(-2)
print(result)

x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

plt.title('Ortalamadan 2 Standart Sapma Arası Bölge', fontweight='bold')
axis = plt.gca()

axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(-4, 5))
axis.text(2, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-2, 2, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()

------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------
Diğer çok karşılaşılan sürekli dağılım "sürekli düzgün dağılım (continous uniform distribution)" 
denilen dağılımdır. Burada dağılımı temsil eden a ve b değerleri vardır. Sürekli 
düzgün dağılımın olasılık yoğunluk fonksiyonu dikdörtgensel bir alandır. Dolayısıyla 
kümülatif dağılım fonksiyonu x değeriyle orantılı bir değer vermektedir. Sürekli 
düzgün dağılımın olasılık yoğunluk fonksiyonu şöyle ifade edilebilir:

f(x) = {
            1 / (b - a)     a < x < b
            0               diğer durumlarda    
       }

------------------------------------------------------------------------------------
Sürekli düzgün dağılım için Python'ın standart kütüphanesinde bir sınıf bulunmamaktadır. 
NumPy'da da böyle bir sınıf yoktur. Ancak SciPy içerisinde stats modülünde uniform 
isimli bir singleton nesne bulunmaktadır. Bu nesneye ilişkin sınıfın yine cdf, ppf, 
pdf ve rvs metotları vardır. Bu metotlar sırasıyl a değerini ve a'dan uzunluğu 
parametre olarak almaktadır. Örneğin:

result = uniform.pdf(15, 10, 10)

Burada aslında a = 10, b = 20 olan bir sürekli düzgün dağılımdaki olasılık yoğunluk 
fonksiyon değeri elde edilmektedir. Tabii aslında 10 ile 20 arasındaki tüm olasılık 
yoğunluk fonksiyon değerleri 1 / 10 olacaktır.


result = uniform.cdf(15, 10, 10)

Burada a = 10, b = 20 olan bir sürekli düzgün dağılımda 15'in solundaki alan elde 
edilecektir. 15 burada orta nokta olduğunda göre elde edilecek bu değer 0.5'tir.
uniform nesnesinin metotlarındaki ikinci parametrenin (loc) a değeri olduğuna ancak 
üçüncü parametrenin a'dan uzaklık belirttiğine (scale) dikkat ediniz.

------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

A = 10
B = 20

x = np.linspace(A - 5, B + 5, 1000)
y = uniform.pdf(x, A, B - A)

plt.title('Continupos Uniform Distribution', fontweight='bold')
plt.plot(x, y)

x = np.linspace(10, 12.5, 1000)  # 12.5'in cdf'sini boyamak icin
y = uniform.pdf(x, A, B - A)
plt.fill_between(x, y)
plt.show()

result = uniform.cdf(12.5, A, B - A)
print(result)                               # 0.25

result = uniform.ppf(0.5, A, B - A)
print(result)                               # 15

------------------------------------------------------------------------------------
Düzgün dağılmış rastgele sayı aslında bizim aşina olduğumuz klasik rastgele üretimidir. 
Örneğin Python standart kütüphanesindeki random modülünde bulnan random fonksiyonu 
0 ile 1 arasında rastgele bir sayı veriyordu. Aslında bu fonksiyon a = 0, b = 1 olan
düzgün dağılımda rastegele sayı veren fonksiyonla tamamen aynıdır. Benzer biçimde 
NumPy'daki random modülündeki random fonksiyonu 0 ile 1 arasında düzgün dağılmış 
rastgele sayı üretmektedir. Örneğin:

result = uniform.rvs(10, 10, 10)
print(result)  

Burada a = 10, b = 20 olan sürekli düzgün dağılımda 10 tane rastgele noktalı sayı 
elde edilecektir. 

Örneğin 100 ile 200 arasında rastgele 10 tane gerçek sayı üretmek istesek bu işlemi 
şöyle yapmalıyız:

uniform.rvs(100, 200, 10)
------------------------------------------------------------------------------------
"""


# t Dağılımı

"""   
------------------------------------------------------------------------------------
Özellikle güven aralıklarında (confidence interval) ve hipotez testlerinde kullanılan 
diğer önemli bir sürekli dağılım da "t dağılımı (t distribution)" denilen dağılımdır.

t dağılımı standart normal dağılıma oldukça benzemektedir. Bu dağılımın ortalaması 
0'dır. Ancak standart sapması "serbestlik derecesi (degrees of freedom)" denilen 
bir değere göre değişir. t dağılımının standart sapması sigma = karekök(df / (df - 2)) 
biçimindedir. t dağılımın olasılık yoğunluk fonksiyonu biraz karmaşık bir görüntüdedir. 
Ancak fonksiyon standart normal dağılıma göre "daha az yüksek ve biraz daha şişman" 
gibi gözükmektedir. t dağılımının serbestlik derecesi artırıldığında dağılım standart 
normal dağılıma çok benzer hale gelir. Serbestlik derecesi >= 30 durumunda standart 
normal dağılımla oldukça örtüşmektedir. Yani serbestlik derecesi >= 30 durumunda 
artık t dağılımı kullanmakla standart normal dağılım kullanmak arasında önemli bir 
farklılık kalmamaktadır.

t dağılımı denildiğinde her zaman ortalaması 0 olan standart sapması 1 olan (bu 
konuda bazı ayrıntılar vardır) dağılım anlaşılmaktadır. Tabii t dağılımı da eksende 
kaydırılabilir ve standart sapma değiştirilebilir.

t dağılımı teorik bir dağılımdır. Yukarıda da belirttiğimiz gibi özellikle "güven 
aralıklarının oluşturulması" ve "hipotez testlerinde" kullanım alanı bulmaktadır. 

!!!!!
Bu tür durumlarda anakütle standart sapması bilinmediği zaman örnek standart sapması
anakütle standart sapması olarak kullanılmakta ve t dağılımından faydalanılmaktadır.
!!!!!

t dağılımının önemli bir parametresi "serbestlik derecesi (degrees of freedom)" 
denilen parametresidir. Serbestlik derecesi örneklem büyüklüğünden bir eksik değeri 
belirtir. Örneğin örneklem büyüklüğü 10 ise serbestlik derecesi 9'dur. 

------------------------------------------------------------------------------------
 t dağılımına ilişkin Python standart kütüphanesinde bir sınıf yoktur. NumPy 
kütüphanesinde de t dağılımına ilişkin bir öğe bulunmamaktadır. Ancak SciPy 
kütüphanesindeki stats modülünde t isimli singleton nesne t dağılımı ile işlem yapmak 
için kullanılmaktadır. t isimli singleton nesnenin metotları norm nesnesinin 
metotlarıyla aynıdır. 

Bu fonksiyonlar genel olarak önce x değerlerini sonra serbestlik derecesini, 
sonra da ortalama değeri ve standart sapma değerini parametre olarak almaktadır. 
Ancak yukarıda da belirttiğimiz gibi t dağılımı denildiğinde genel olarak ortalaması 
0, standart sapması 1 olan t dağılımı anlaşılır.

------------------------------------------------------------------------------------
Aşağıdaki programda standart normal dağılım ile 5 serbestlik derecesi ve 30 
serbestlik derecesine ilişkin t dağılımlarının olasılık yoğunluk fonksiyonları 
çizdirilmiştir. Burada özellikle 30 serbestlik derecesine ilişkin t dağılımının 
grafiğinin standart normal dağılım grafiği ile örtüşmeye başladığına dikkat ediniz. 


import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_xticks(range(-4, 5))
plt.plot(x, y)

y = t.pdf(x, 1)                 # 1 --> serbestlik derecesi
plt.plot(x, y)

plt.legend(['Standart Normal Dağılım', 't Dağılımı (DOF = 5)', 't dağılımı (DOF = 30)'])

y = t.pdf(x, 30)                # 30 --> serbestlik derecesi
plt.plot(x, y, color='red')     
                    
                # >= 30'dan sonra normal dağılımla hemen hemen aynı grafik oluyor  
plt.show()

------------------------------------------------------------------------------------
Değişik Serbestlik Derecelerine İlişkin t Dağılımı Grafikleri


from scipy.stats import norm,  t
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(15, 10))
plt.title('Değişik Serbestlik Derecelerine İlişkin t Dağılımı Grafikleri', fontweight='bold')
axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

y_norm = norm.pdf(x)
plt.plot(x, y_norm, color='blue')

df_info = [(2, 'red'), (5, 'green'), (10, 'black')]

for df, color in df_info:
    y_t = t.pdf(x, df)
    plt.plot(x, y_t, color=color)
    
plt.legend(['Standart Normal Dağılım'] + [f'{t[0]} Serbestlik Derecesi' for t in df_info], fontsize=14)

plt.show()

------------------------------------------------------------------------------------
Tabii standart normal dağılımla t dağılının olasılık yoğunluk fonksiyonları farklı 
olduğuna göre aynı değerlere ilişkin kümülatif olasılık değerleri de farklı olacaktır. 


from scipy.stats import norm,  t

x = [-0.5, 0, 1, 1.25]
result = norm.cdf(x)
print(result)               # [0.30853754 0.5        0.84134475 0.89435023]

x = [-0.5, 0, 1, 1.25]
result = t.cdf(x, 5)    
print(result)               # [0.31914944 0.5        0.81839127 0.86669189]

------------------------------------------------------------------------------------
"""


# Kesikli (discrete) dağılım
"""  
------------------------------------------------------------------------------------
 Kesikli dağılımlarda X değerleri her gerçek değeri almamaktadır. Dolayısıyla bunların 
fonksiyonları çizildiğinde sürekli fonksiyonlar elde edilemeyecek yalnızca noktalar 
elde edilecektir. Kesikli dağılımlarda X değerlerini onların olasılıklarına eşleyen 
fonksiyonlara "olasılık kütle fonksiyonu (probability mass function)" denilmektedir. 
Sürekli rassal değişkenlerin olasılık yoğunluk fonksiyonları integral hesap için 
kullaılırken kesikli rassal değişkenler için olasılık kütle fonksiyonları doğrudan 
rassal değişkeninin ilgili noktadaki olasılığını vermektedir. Tabii olasılık kütle 
fonksiyonu f(x) olmak üzere her x için f(x) değerlerinin toplamının yine 1 olması 
gerekmektedir.

------------------------------------------------------------------------------------
# Poisson dağılımı

En çok karşılaşılan kesikli dağılımlardan biri "poisson (genellikle "puason" biçiminde 
okunuyor)" dağılımıdır. Bu kesikli dağılım adeta normal dağılımın kesikli versiyonu 
gibidir. Poisson dağılımının olasılık kütle fonksiyonu şöyledir:

    P(X = x) = (e^-lambda * lambda^x) / x!

Buradaki x olasılığını hesaplamak istediğimiz kesikli değeri belirtmektedir. Lamda 
ise ortalama olay sayısını belirtmektedir. Lambda değeri ortalama belirttiği için 
gerçek bir değer olabilir. Ancak x değerleri 0, 1, 2, ... n biçiminde o ve pozitif 
tamsayılardan oluşmaktadır.

Yukarıda da belirttiğimiz gibi poisson dağılımı adeta normal dağılımın kesikli hali 
gibidir. Dolayısıyla doğada da çok karşılaşılan kesikli dağılımlardandır.

Poisson dağılımı için de Python standart kütüphanesinde ya da Numpy ve Pandas 
kütüphanelerinde özel fonksiyonlar ve sınıflar bulunmamaktadır. Ancak SciPy 
kütüphanesinin stats modülü içerisinde poisson isimli bir single nesne ile bu 
dağılımla ilgili işlemler kolaylıkla yapılabilmektedir. 

SciPy'da kesikli dağılımlar üzerinde işlemler yapan singleton nesneler sürekli 
dağılımlarla işlemler yapan single nesnelere kullanım bakımından oldukça benzemektedir. 
Ancak kesikli dağılımlar için fonksiyonun ismi "pdf değil pmf" biçimindedir. Burada 
"pmf" ismi "probability mass function" sözcükleridnen kısaltılmıştır. 

SciPy'daki poisson nesnesinin fonksiyonları genel olarak bizden x değerini ve lambda 
değerini parametre olarak istemektedir. Örneğin maçlardaki gol sayısının poisson 
dağılıma uyduğunu varsayalım. Maçlardaki ortalama gol sayısının 2 olduğunu kabul 
edelim. Bu durumda bir maçta 5 gol olma olasılığı aşağıdaki gibi elde edilebilir:

from scipy.stats import poisson    
result = poisson.pmf(5, 2)
print(result)           # 0.03608940886309672


Yukarıdaki gibi poisson dağılımı sorularında genellikle soruyu soran kişi belli bir 
olayın ortalama gerçekleşme sayısını verir. Sonra kişiden bazı değerleri bulmasını 
ister. Peki soruda "bir maçta 2'den fazla gol olma olasılığı sorulsaydı biz soruyu 
nasıl çözerdik? poisson nesnesi ile cdf fonksiyonunu çağırdığımızda bu cdf fonksiyonu 
bize x değerine kadarki (x değeri de dahil olmak üzere) kümülatif olasılığı verecektir. 
Bu değeri de 1'den çıkartırsak istenen olasılığı elde edebiliriz:

result = 1 - poisson.cdf(2, 2)
print(result)   # 0.3233235838169366

------------------------------------------------------------------------------------
Poisson dağılımında lamda değeri yüksek tutulursa saçılma grafiğinin Gauss 
eğrisine benzediğine dikkat ediniz. 

from scipy.stats import poisson
import matplotlib.pyplot as plt

plt.title('Poisson Distribution with Lambda 100', fontweight='bold')
x = range(0, 200)
y = poisson.pmf(x, 100)

plt.scatter(x, y)
plt.show()

result = poisson.pmf(3, 4)
print(result)

------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------

# Bernoulli dağılımı

Bernoulli dağılımında X değeri 0 ya da 1 olabilir. X = 0 durumu bir olayın olumsuz 
olma ya da gerçekleşmeme olasılığını, X = 1 durumu ise bir olayın olumlu olma ya 
da gerçekleşme olasılığını belirtmektedir. Bu rassal değişkenin yalnızca iki değer 
aldığına dikkat ediniz. Sonucu iki değerden biri olan rassal deneylere 
"Bernoulli deneyleri" de denilmektedir. Bernoulli dağılımının olasılık kütle 
fonksiyonu şöyle ifade edilebilir:

P{X = x} =  {
                p           X = 1 ise
                1 - p       X = 0 ise

            }

Ya da bu olasılık kütle fonksiyonunu aşağıdaki gibi de ifade edebiliriz:

P{X = x} = p^x * (1 - p)^(1 - x)

Burada X = 0 için 1 - p değerinin X = 1 için p değerinin elde edildiğine dikkat ediniz. 


Bernoulli dağılımı için de SciPy kütüphanesinde stats modülü içerisinde bernoulli 
isimli bir singleton nesne bulundurulmuştur. Tabii bu dağılım çok basit olduğu 
için bu nesnenin kullanılması da genellikle gereksiz olmaktadır. bernoulli nesnesinin
ilikin olduğu sınıfın metotları bizden X değerini (0 ya da 1 olabilir) ve X = 1 
için p değerini almaktadır. Örneğin:

from scipy.stats import bernoulli
bernoulli.pmf(0, 0.7)               # 0.3


buradan 0.3 değeri elde edilecektir. 
------------------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------------------

# binom dağılım

Binom dağılımında bir Bernoulli deneyi (yani iki sonucu olan bir deney) toplam 
n defa yinelenmektedir. Bu n defa yinelenmede olayın tam olarak kaç defa olumlu
sonuçlanacağının olasılığı hesaplanmak istenmektedir. Dolayısıyla binom dağılımının 
olasılık kütle fonksiyonunda X değerleri 0, 1, 2, ... gibi tamsayı değerler alır. 

Örneğin bir para 5 kez atılıyor olsun. Biz de bu 5 kez para atımında tam olarak 
3 kez Tura gelme olasılığını hesaplamak isteyelim. Burada paranın atılması bir 
Bernoulli deneyidir. Bu deney 5 kez yinelenmiş olumlu kabul ettiğimiz Tura gelme 
durumunun toplamda 3 kez olmasının olasılığı elde edilmek istenmiştir. 
 
Binom dağılımının olasılık kütle fonksiyonu şöyledir:

    P{X = x} = C(n, x) *  p^x  *  (1 - p)^(n - x)


Binom dağılımı için SciPy kütüphanesindeki stats modülünde bulunan binom isimli 
singleton nesne kullanılabilir. Nesnenin kullanılması diğer singleton nesnelere 
benzemektedir. Örneğin ilgili sınıfın pmf fonksiyonu olasılık kütle fonlsiyonunu 
belirtmektedir. 


Örneğin yukarıda belirttiğimiz paranın 5 kez atılmasında tam olarak 3 kez tura 
gelme olsaılığı şöyle elde edilebilir:

from scipy.stats import binom
binom.pmf(3, 5, 0.5)    

Bu fonksiyonlarda birinci parametre "olumlu gerçekleşme sayısını", ikinci parametre 
"toplam deney sayısını"  ve üçüncü parametre de "olumlu gerçekleşme olasılığını" 
belirtmektedir. 
------------------------------------------------------------------------------------
"""

#   Merkezi limit teoremi (central limit theorem)

"""
------------------------------------------------------------------------------------
Bu teoreme göre bir "anakütleden (population)" çekilen belli büyüklükteki örneklerin 
ortalamaları normal dağılmaktadır. Örneğin elimizde 100,000 elemanlı bir anakütle 
olsun. Bu anakütleden 50'lik tüm alt kümeleri yani örnekleri elde edip bunların 
ortalamalarını hesaplayalım. İşte bu ortalamalar normal dağılmaktadır. Bir anakütleden 
alınan alt kümelere "örnek (sample)" denilmektedir. Bu işleme de genel olarak 
"örnekleme (sampling)" denir. Örneğimizdeki 100,000 elemanın 50'li alt kümelerinin 
sayısı çok fazladır. O halde deneme için 100,000 elemanlı anakütlenin tüm alt 
kümelerini değil belli sayıda alt kümelerini elde ederek histogram çizebiliriz. Bu 
histogramın teoreme göre Gauss eğrisine benzemesi gerekir. 

------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = np.random.randint(0, POPULATION_RANGE, POPULATION_SIZE)
samples = np.random.choice(population, (NSAMPLES, SAMPLE_SIZE))
samples_means = np.mean(samples, axis=1)

plt.hist(samples_means, bins=50)

plt.show()

------------------------------------------------------------------------------------
Tabii yukarıdaki örneği hiç NumPy kullanmadan tamamen Python standart kütüphanesi 
ile de yapabilirdik.

import random
import statistics

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = random.sample(range(POPULATION_RANGE), POPULATION_SIZE)

samples_means = [statistics.mean(random.sample(population, SAMPLE_SIZE)) for _ in range(NSAMPLES)]

samples_means_mean = statistics.mean(samples_means)

import matplotlib.pyplot as plt

plt.title('Central Limit Theorem', fontweight='bold')
plt.hist(samples_means, bins=50)
plt.show()

------------------------------------------------------------------------------------
Merkezi limit teoremine göre örnek ortalamalarına ilişkin normal dağılımın 
ortalaması anakütle ortalaması ile aynıdır. (Yani bir anakütleden çekilen örnek 
ortalamalarının ortalaması anakütle ortalaması ile aynıdır.)
Aşağıdaki programda bu durum gösterilmiştir.


import numpy as np
import matplotlib.pyplot as plt

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = np.random.randint(0, POPULATION_RANGE, POPULATION_SIZE)
samples = np.random.choice(population, (NSAMPLES, SAMPLE_SIZE))

population_mean = np.mean(population)
samples_means = np.mean(samples, axis=1)
samples_means_mean = np.mean(samples_means)

plt.hist(samples_means, bins=50)

plt.show()

print(f'Anakütle ortalaması: {population_mean}')
print(f'Örnek ortalamalarının Ortalaması: {samples_means_mean}')

------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
Peki bir anakütleden alınan örnek ortalamaları normal dağılmaktadır ve bu normal 
dağılımın ortalaması anakütle ortalamasına eşittir. Peki örnek ortalamalarına 
ilişkin normal dağılımın standart sapması nasıldır? İşte merkezi limit teoremine 
göre örnek ortalamalarının standart sapması ------------->

"anakütle kütle standart sapması / karekök(n)" biçimindedir.   n->(örnek büyüklüğü)

Yani örnek ortalamalarının standart sapması anakütle sapmasından oldukça küçüktür 
ve örnek büyüklüğüne bağlıdır. Buradaki n örnek büyüklüğünü belirtmektedir. Örnek 
ortalamalarının standart sapmasına "standard error" de denilmektedir. Görüldüğü 
gibi eğer örnek ortalamalarına ilişkin normal dağılımın standart sapması düşürülmek 
isteniyorsa (başka bir deyişle standart hata azaltılmak isteniyorsa) örnek büyüklüğü
artırılmalıdır. 

------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

POPULATION_RANGE = 1_000_000_000
POPULATION_SIZE = 1_000_000
NSAMPLES = 10000
SAMPLE_SIZE = 50

population = np.random.randint(0, POPULATION_RANGE, POPULATION_SIZE)
samples = np.random.choice(population, (NSAMPLES, SAMPLE_SIZE))

population_mean = np.mean(population)
population_std = np.std(population)

samples_means = np.mean(samples, axis=1)
samples_means_mean = np.mean(samples_means)
sample_means_std = np.std(samples_means)

plt.hist(samples_means, bins=50)

plt.show()

print(f'Anakütle ortalaması: {population_mean}')
print(f'Örnek ortalamalarının Ortalaması: {samples_means_mean}')
print(f'Fark: {np.abs(population_mean - samples_means_mean)}')
print("------------------")

print(f'Merkezi limit teroreminden elde edilen örnek ortalamalarının standart sapması: {population_std / np.sqrt(SAMPLE_SIZE)}')
print()
print(f'Örnek ortalamalarının standart sapması: {sample_means_std}')

------------------------------------------------------------------------------------
Merkezi limit teoremine göre eğer anakütleden çekilen örnekler büyükse yani tipik 
olarak n örnek büyüklüğü N ise anakütle büyüklüğü olmak üzere n / N >= 0.05 ise 
bu durumda örneklem dağılımın standart sapması için "anakütle standart sapması / kök(n)" 
değeri "düzeltme faktörü (correction factor)" denilen bir çarpanla çarpılmalıdır. 
Düzeltme faktörü karekök((N - n)/(N -1)) biçimindedir. Bu konunun ayrıntıları için 
başka kaynaklara başvurabilirsiniz. Örneğin ana kütle 100 elemandan oluşuyor olsun. 
Biz 30 elemanlı örnekler çekersek bu düzeltme faktörünü kullanmalıyız. Tabii 
paratikte genellikle n / N değeri 0.05'ten oldukça küçük olmaktadır.

------------------------------------------------------------------------------------
Merkezi limit teoreminde anakütlenin normal dağılmış olması gerekmez. Nitekim 
yukarıdaki örneklerimizde bir anakütleyi "düzgün dağılıma (uniform distribution)" 
ilişkin olacak biçimde oluşturduk. Ancak eğer anakütle normal dağılmamışsa örneklem 
ortalamalarının dağılımının normal olması için örnek büyüklüklerinin belli bir 
değerden büyük olması gerekmektedir. Bu değer tipik olarak >= 30 biçimindedir.
Özet olarak:
    
- Eğer anakütle normal dağılmışsa örnek büyüklüğü ne olursa olsun örneklem 
ortalamalarının dağılımı normaldir. 

- Eğer anakütle normal dağılmamışsa örneklem ortalamalarının dağılımının normal 
olması için örnek büyüklüğünün >= 30 olması gerekir. Tabii n < 30 durumunda yine 
örneklem dağılımı normale benzemektedir ancak kusurlar oluşmaktadır.

Özetle anakütle normal dağılmamışsa örnek büyüklüklerinin artırılması gerekmektedir. 

------------------------------------------------------------------------------------
"""

# Normalliğin Test Edilmesi

"""
------------------------------------------------------------------------------------
Sonuç çıkartıcı istatistikte (inferential statisics) bazen anakütlenin normal dağılıp 
dağılmadığını örneğe dayalı olarak test etmek gerebilir. Bunun için anakütleden 
bir örnek alınır. Sonra bu örneğe bakılarak anakütlenin normal dağılıp dağılmadığı 
belli bir "güven düzeyinde (confidence level)" belirlenir. 

Buna "normallik testleri" denilmektedir. Aslında normallik testi gözle de üstünkörü 
yapılabilmektedir. Anakütle içerisinden bir örnek çekip onun histogramını çizersek 
eğer bu histogram Gauss eğrisine benziyorsa biz anakütlenin de normal dağılmış 
olduğu sonucunu gözle tespit edebiliriz. Ancak anakütlenin normal dağılıp 
dağılmadığının tespit edilmesi için aslında "hipotez testleri (hypothesis testing)" 
denilen özel testler kullanılmaktadır. Normal dağılıma ilişkin iki önemli hipotez 
testi vardır: 

"Kolmogorov-Smirnov" testi ve "Shapiro-Wilk" testidir. Bu testlerin istatistiksel 
açıklaması biraz karmaşıktır ve bizim konumuz içerisinde değildir. Ancak bu testler 
SciPy kütüphanesindeki stats modülü içerisinde bulunan fonksiyonlarla yapılabilmektedir. 

Hipotez testlerinde bir hipotez öne sürülür ve bu hipotezin belirli güven düzeyi 
içerisinde doğrulanıp doğrulanmadığına bakılır. Genel olarak bizim doğrulanmasını 
istediğimiz hipoteze H0 hipotezi, bunun tersini belirten yani arzu edilmeyen durumu 
belirten hipoteze de H1 hipotezi denilmektedir. Örneğin normal testindeki H0 ve H1 
hipotezleri şöyle oluşturulabilir:

    H0: Seçilen örnek normal bir anakütleden gelmektedir. 
    H1: Seçilen örnek normal dağılmış bir anakütleden gelmemektedir.
    
------------------------------------------------------------------------------------
scipy.stats modülündeki kstest fonksiyonunun ilk iki parametresi zorunlu parametrelerdir. 
Birinci parametre anakütleden rastgele seçilen örneği alır. İkinci parametre testi 
yapılacak dağılımın kümülatif dağılım fonksiyonunu parametre olarak almaktadır. 
Ancak bu parametre kolaylık olsun diye yazısal biçimde girilebilmektedir. Normallik 
testi için bu parametre 'norm' ya da norm.cdf biçiminde girilebilir. 

------------------------------------------------------------------------------------
kstest fonksiyonu çağrıldığktan sonra bize "isimli bir demet (named tuple) verir. 
Demetin birinci elemanı test istatistiğini, ikinci elemanı p değerini belirtir. 
Bizim burada yapmamız gereken bu p değerinin kendi seçtiğimiz belli bir kritik
değerden büyük olup olmadığına bakmaktır. Bu kritik değer tipik olarak 0.05 olarak 
alınmaktadır. Ancak testi daha katı yapacaksanız bu değeri 0.01 gibi küçük tutabilirsiniz.

Eğer "p değeri belirlediğimiz kritik değereden (0.05) büyükse H0 hipotezi kabul edilir, 
H1 hipotezi reddedilir. Eğer p değeri bu kritik değerden küçükse H0 hipotezi reddelip, 
H1 hipotezi kabul edilmektedir. Yani özetle bu p değeri 0.05 gibi bir kritik değerden 
büyükse örnek normal dağılmış bir anakütleden gelmektedir, 0.05 gibi bir kritik 
değerden küçükse örnek normal dağılmamış bir anakütleden gelmektedir.

Aşağıdaki örnekte normal dağılmış bir anakütleden ve düzgün dağılmış bir anakütleden 
rastgele örnekler çekilip kstest fonksiyonuna sokulmuştur.
    
    
from scipy.stats import norm, uniform, kstest

sample_norm = norm.rvs(size=1000)

result = kstest(sample_norm, 'norm')
print(result.pvalue)        # 1'e yakın bir değer             

sample_uniform = uniform.rvs(size=1000) 

result = kstest(sample_uniform, norm.cdf)
print(result.pvalue)        # 0'a çok yakın bir değer

------------------------------------------------------------------------------------    
kstest fonksiyonunda ikinci parametreye 'norm' girildiğinde birinci parametredeki 
değerler ortalaması 0, standart sapması 1 olan standart normal dağılıma uygun değerler 
olmalıdır. Eğer ortalama ve standart sapması farklı bir normal dağılım testi 
yapılacaksa dağılımın parametreleri de ayrıca args parametresiyle verilmelidir. 
Örneğin biz ortalaması 100 standart sapması 15 olan bir dağılıma ilişkin test yapmak 
isteyelim. Bu durumda test aşağıdaki gibi yapılmalıdır. 


from scipy.stats import norm, uniform, kstest

sample_norm = norm.rvs(100, 15, size=1000)  # ort=100 std'si 15 olan rastgele değerler

result = kstest(sample_norm, 'norm', args=(100, 15))
print(result.pvalue)                   

sample_uniform = uniform.rvs(100, 100, size=1000) # 100 ile 200 arasında rastgele değerler

result = kstest(sample_uniform, norm.cdf, args=(100, 100))
print(result.pvalue)

------------------------------------------------------------------------------------  
Shapiro-Wilk testi de tamamen benzer biçimde uygulanmaktadır. Ancak bu fonksiyonun 
kullanılması daha kolaydır. Bu fonksiyonun tek bir parametresi vardır. Bu parametre 
anakütleden çekilen örneği belirtir. Buradaki normal dağılım herhangi bir ortalama ve 
standart sapmaya ilişkin olabilir. Yani bizim dağılım değerlerini ortalaması 0, 
standart sapması 1 olacak biçimde ölçeklendirmemiz gerekmemektedir. 

Ayrıca anakütleden çekilen örnekler küçükse (tipik olarak <= 50) Shapiro-Wilk testi 
Kolmogorov-Simirnov testine göre daha iyi bir sonucun elde edilmesine yol açmaktadır. 
Yani örneğiniz küçükse Shapiro-Wilk testini tercih edebilirsiniz. 

Aşağıdaki örnekte ortalaması 100, standart sapması 15 olan normal dağılmış ve düzgün 
dağılmış bir anakütleden 100'lük bir örnek seçilip Shapiro-Wilk testine sokulmuştur. 
Buradan elde edine pvalue değerlerine dikkat ediniz. 


from scipy.stats import norm, uniform, shapiro

sample_norm = norm.rvs(100, 15, size=100)

result = shapiro(sample_norm)
print(result.pvalue)                   

sample_uniform = uniform.rvs(100, 100, size=100) 

result = shapiro(sample_uniform)
print(result.pvalue)

------------------------------------------------------------------------------------  
"""

# Örnekten Hareketle Anakütle Parametrelerinin Tahmin Edilmesi

"""
------------------------------------------------------------------------------------  
İstatistikte örneğe dayalı olarak anakütlenin ortalamasını (ve/veya standart sapmasını) 
tahmin etmeye "parametre tahmini (parameter estimation)" denilmektedir. Parametre 
tahmini "noktasal olarak (point estimate)" ya da "aralıksal olarak (interval estimate)" 
yapılabilmektedir. Anakütle ortalamasının aralıksal tahminine "güven aralıkları 
(confidence interval)" da denilmektedir. Güven aralıkları tamamen merkezi limit 
teroremi kullanılarak oluşturulmaktadır.

------------------------------------------------------------------------------------  
Bizim anakütle ortalamasını bilmediğimizi ancak anakütle standart sapmasını bildiğimizi 
varsayalım. (Genellikle aslında anakütle standart sapmasını da bilmeyiz. Ancak 
burada bildiğimizi varsayıyoruz.) Bu anakütleden rastgele bir örnek seçtiğimizde 
o örneğin ortalamasına bakarak anakütle ortalamasını belli bir aralıkta belli bir 
güven düzeyinde tahmin edebiliriz. 

Şöyle ki: Örneğin seçtiğimiz güven düzeyi %95 olsun. Bu durumda bizim örneğimiz 
örnek ortalamalarının dağılımında en kötü olasılıkla soldan 0.025 ve sağdan 0.975 
kümülatif olasılığa karşı gelen x değerlerinden biri olabilir. Tabii seçtiğimiz 
örneğin ortalamasının bu aralıkta olma olasılığı %95'tir Bu durumda yapacağımız 
şey örnek ortalamasının örneklem dağılımına göre seçtiğimiz örneğin ortalamasının 
%47.5 soluna ve %47.5 sağına ilişkin değerlerin elde edilmesidir. Bu durumda anakütle 
ortalaması %95 güven düzeyi içerisinde bu aralıkta olacaktır. Tabii aslında bu 
işlemi daha basit olarak "rastgele elde ettiğimiz örneğin ortalamasını normal 
dağılımın merkezine alarak soldan 0.025 ve sağdan 0.975 kümülatif olasılık 
değerlerine karşı gelen noktaların elde edilmesi yoluyla da yapabiliriz.  

------------------------------------------------------------------------------------  
Örneğin standart sapması 15 olan bir anakütleden rastgele 60 elemanlık bir örnek 
elde etmiş olalım. Bu örneğin ortalamasının 109 olduğunu varsayalım. Bu durumda 
%95 güven düzeyi içerisinde anakütle ortalamasına ilişkin güven aralıkları 
aşağıdaki gibi elde edilebilir:

    
import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109
sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)

print(f'{lower_bound}, {upper_bound}')              # 105.20454606435501, 112.79545393564499


Burada biz anakütlenin standart sapmasını bildiğimiz için örnek ortalamalarına 
ilişkin normal dağılımın standart sapmasını hesaplayabildik. Buradan elde ettiğimiz 
güven aralığı şöyle olmaktadır:

105.20454606435501, 112.79545393564499

------------------------------------------------------------------------------------ 
Güven düzeyini yükseltirsek güven aralığının genişleyeceği açıktır. Örneğin bu 
problem için güven düzeyini %99 olarak belirlemiş olalım:

import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109
sampling_mean_std = population_std / np.sqrt(sample_size)

lower_bound = norm.ppf(0.005, sample_mean, sampling_mean_std)
upper_bound = norm.ppf(0.995,  sample_mean, sampling_mean_std)

print(f'{lower_bound}, {upper_bound}')     # 104.01192800234102, 113.98807199765896


Burada güven aralığının aşağıdaki gibi olduğunu göreceksiniz:

104.01192800234102, 113.98807199765896

Gördüğünüz gibi aralık büyümüştür.

------------------------------------------------------------------------------------  
Anımsanacağı gibi örnek ortalamalarına ilişkin dağılımın standart sapmasın "standart hata 
(standard error)" deniliyordu. Örnek ortalamalarına ilişkin dağılımın standart sapması 
azaltılırsa (yani standart hata düşürülürse) değerler ortalamaya yaklaşacağına göre 
güven aralıkları da daralacaktır. O halde anakütle ortalamasını tahmin ederken 
büyük örnek seçmemiz güven aralıklarını daraltacaktır. Aşağıdaki örnekte yukarıdaki 
problemin 30'dan 100'e kadar beşer artırımla örnek büyüklükleri için %99 güven 
düzeyinde güven aralıkları elde edilmiştir. Elde edilen aralıklar şöyledir:

import numpy as np
from scipy.stats import norm

population_std = 15
sample_mean = 109

for sample_size in range(30, 105, 5):
    
    sampling_mean_std = population_std / np.sqrt(sample_size)
    
    lower_bound = norm.ppf(0.025, sample_mean, sampling_mean_std)
    upper_bound = norm.ppf(0.975,  sample_mean, sampling_mean_std)
    
    print(f'sample size: {sample_size}: [{lower_bound}, {upper_bound}]')  

    
sample size: 30: [103.63241756884852, 114.36758243115148]
sample size: 35: [104.03058429805395, 113.96941570194605]
sample size: 40: [104.35153725771579, 113.64846274228421]
sample size: 45: [104.61738729711709, 113.38261270288291]
sample size: 50: [104.84228852695097, 113.15771147304903]
sample size: 55: [105.03577765357056, 112.96422234642944]
sample size: 60: [105.20454606435501, 112.79545393564499]
sample size: 65: [105.3534458105975, 112.6465541894025]
sample size: 70: [105.48609245861904, 112.51390754138096]
sample size: 75: [105.60524279777148, 112.39475720222852]
sample size: 80: [105.71304047283782, 112.28695952716218]
sample size: 85: [105.81118086644236, 112.18881913355763]
sample size: 90: [105.9010248384772, 112.0989751615228]
sample size: 95: [105.98367907149301, 112.01632092850699]
sample size: 100: [106.06005402318992, 111.93994597681008]


Buradan da gördüğünüz gibi örneği büyüttüğümüzde güven aralıkları daralmakta ve 
anakütle ortalaması daha iyi tahmin edilmektedir. Örnek büyüklüğünün artırılması 
belli bir noktaya kadar aralığı iyi bir biçimde daraltıyorsa da belli bir noktadan 
sonra bu daraltma azalmaya başlamaktadır. Örneklerin elde edilmesinin belli bir 
çaba gerektirdiği durumda örnek büyüklüğünün makul seçilmesi önemli olmaktadır.

------------------------------------------------------------------------------------  
Anımsanacağı gibi anakütle normal dağılmamışsa merkezi limit teoreminin yeterli 
bir biçimde uygulanabilmesi için örneklerin büyük olması (tipik olarak >= 30) 
gerekiyordu. O halde güven aralıklarını oluştururken eğer anakütle normal dağılmamışsa
bizim örnekleri >= 30 biçiminde seçmemiz uygun olur.

------------------------------------------------------------------------------------  
Örnekten hareketle anakütle ortalamaları aslında tek hamlede norm nesnesinin ilişkin 
olduğu sınıfın interval metoduyla da elde edilebilmektedir. interval metodunun 
parametrik yapısı şöyledir:

interval(confidence, loc=0, scale=1)

Metodun birinci parametresi olan confidence güven düzeyini belirtmektedir. Örneğin 
%95 güven düzeyi için bu parametre 0.95 girilmelidir. Metodun ikinci ve üçüncü 
parametreleri örnek ortalamalarının dağılımına ilişkin ortalama ve standart sapmayı 
belirtir. Tabii ikinci parametre elde etmiş olduğumuz örneğin ortalaması olarak 
girilmelidir. Metot güven aralığını belirten bir demetle geri döner. Demetin ilk 
elemanı lower_bound ikinci elemanı upper_bound değerlerini vermektedir.

Bu durumda yukarıdaki problemi interval metoduyla aşağıdaki gibi de çözebiliriz:

import numpy as np
from scipy.stats import norm

sample_size = 60
population_std = 15
sample_mean = 109

sampling_mean_std = population_std / np.sqrt(sample_size)
lower_bound, upper_bound = norm.interval(0.95, sample_mean, sampling_mean_std)
print(f'{lower_bound}, {upper_bound}')     

------------------------------------------------------------------------------------  
Aşağıdaki örnekte ortalaması 100, standart sapması 15 olan normal dağılıma uygun 
rastgele 1,000,000 değer üretilmiştir. Bu değerlerin anakütleyi oluşturduğu varsayılmıştır. 
Sonra bu anakütle içerisinden rastgele 60 elemanlık bir örnek elde edilmiştir. Bu 
örneğe dayanılarak ana kütle ortalaması norm nesnesinin interval metoduyla elde 
edilip ekrana yazdırılmıştır.


import numpy as np
from scipy.stats import norm

POPULATION_SIZE = 1_000_000
SAMPLE_SIZE = 60

population = norm.rvs(100, 15, POPULATION_SIZE)

population_mean = np.mean(population)
population_std = np.std(population)

print(f'population mean: {population_mean}')
print(f'population std: {population_std}')
print("----------------------")

sample = np.random.choice(population, SAMPLE_SIZE)
sample_mean = np.mean(sample)
sampling_mean_std = population_std / np.sqrt(SAMPLE_SIZE)

print(f'sample mean: {sample_mean}')


lower_bound, upper_bound = norm.interval(0.95, sample_mean, sampling_mean_std)

print(f'[lower_bound= {lower_bound}, upper_bound= {upper_bound}]')

------------------------------------------------------------------------------------   
"""
"""
------------------------------------------------------------------------------------  
Biz yukarıdaki örneklerde güven aralıklarını oluştururken anakütle standart sapmasının 
bilindiğini varsaydık. Halbuki genellikle anakütle ortalamasının bilinmediği durumda 
anakütle standart sapması da bilinmemektedir. Peki bu durumda örnekten hareketle 
anakütle ortalamasının aralık tahmini nasıl yapılacaktır? İşte bu durumda çektiğimiz 
örneğin standart sapması anakütlenin standart sapması gibi işleme sokulmaktadır.

Ancak dağılım olarak normal dağılım değil t dağılımı kullanılmaktadır. Zaten Gosset 
t dağılımını tamamen böyle bir problemi çözerken geliştirmiştir. Yani t dağılımı 
zaten "anakütle standart sapmasının bilinmediği durumda örneğin standart sapmasının 
anakütle standart sapması olarak alınmasıyla" elde edilen bir dağılımdır. t dağılımının 
serbestlik derecesi denilen bir değere sahip olduğunu anımsayınız. Serbestlik 
derecesi örnek büyüklüğünün bir eksik değeridir. Ayrıca 30 serbestlik derecesinden 
sonra zaten t dağılımının normal dağılıma çok benzediğini de belirtmiştik.

Çektiğimiz örneğin standart sapmasını anakütle standart sapması olarak kullanırken 
örneğin standrat sapması N'e değil (N - 1)'e bölünerek hesaplanmalıdır. Burada 
bölmenin neden (N - 1)'e yapıldığının açıklaması biraz karmaşıktır. Burada bu 
konu üzerinde durmayacağız. Ancak Internet'te bu konuyu açıklayan pek çok kaynak 
bulunmaktadır.İstatistikte çekilen örneklerin standart sapmaları genellikle sigma 
sembolü ile değil s harfiyle belirtilmektedir.

Anımsanacağı gibi pek çok kütüphanede standart sapma ya da varyans hesaplanırken 
bölmenin neye yapılacağına "ddof (delta degrees of freedom)" deniyordu. Standart 
sapma ya da varyans hesabı yapan fonksiyonların ddof parametreleri vardı. NumPy'da 
bu ddof parametresi default 0 iken Pandas'da 1'dir. Bu ddof parametresi (N - değer)
'deki değeri belirtmektedir. Yani ddof = 0 ise bölme N'e ddof = 1 ise bölme 
(N - 1)'e yapılmaktadır. 

------------------------------------------------------------------------------------  
sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  87.99167058,  95.16206704,
    101.78211828,  80.71674993, 126.3793041 , 105.07860807, 98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 
    117.31491413, 105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 81.77519378, 117.61282039, 
    109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

Anakütlein standart sapmasının da bilinmediğini varsayalım. Bu değerlerden hareketle 
%95 güven düzeyinde güven aralığını şöyle oluşturabiliriz:

import numpy as np
from scipy.stats import t

sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  
                87.99167058,  95.16206704, 101.78211828,  80.71674993, 126.3793041 , 105.07860807, 
                98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 117.31491413, 
                105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 
                81.77519378, 117.61282039, 109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 
                100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
sampling_mean_std = sample_std / np.sqrt(len(sample))

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sampling_mean_std)
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')


Burada örneğin standart sapmasını hesaplarken ddof=1 kullandığımıza dikkat ediniz. 
Güven aralıkları normal dağılım kullanılarak değil t dağılımı kullanılarak elde 
edilmiştir. t dağılımındaki serbestlik derecesinin (ppf fonksiyonun ikinci parametresi) 
örnek büyüklüğünün bir eksik değeri olarak alındığını anımsayınız.

Serbestlik derecesi 30'dan sonra artık t dağılımın normal dağılımla örtüşmeye 
başladığını anımsayınız. Buradaki örneğimizde örnek büyüklüğü 35'tir. Örnek 
büyüklüğü >= 30 durumunda t dağılı ile normal dağılım birbirine çok benzediği 
için aslında bu örnekte t dağılımı yerine normal dağılım da kullanabilirdi.

------------------------------------------------------------------------------------  
Tıpkı sscipy.stats modülündeki norm nesnesinde olduğu gibi t nesnesinin de ilişkin 
olduğu snıfın interval isimli bir metodu bulunmaktadır. Bu metot zaten doğrudan 
t dağılımını kullanarak güven aralıklarını hesaplamaktadır. interval metodunun 
parametrik yapısı şöyledir:

interval(confidence, df, loc=0, scale=1)

Buradaki confidence parametresi yine "güven düzeyini (confidence level)" belirtmektedir. 
df parametresi serbestlik derecesini belirtir. loc ve scale parametreleri de sırasıyla 
ortalama ve standart sapma değerlerini belirtmektedir. Burada loc parametresine 
biz örneğimizin ortalamasını, scale parametresine de örneklem dağılımının standart 
sapmasını girmeliyiz. Tabii örneklem dağılımının standart sapması yine örnekten 
hareketle elde edilecektir. Metot yine güven aralığının alt ve üst değerlerini bir 
demet biçiminde geri döndürmektedir.


import numpy as np
from scipy.stats import t

sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948, 87.1273706 ,  76.37932669,  
                   87.99167058,  95.16206704, 101.78211828,  80.71674993, 126.3793041 , 105.07860807, 
                   98.4475209 , 124.47749601,  82.79645255,  82.65166373, 92.17531189, 117.31491413, 
                   105.75232982,  94.46720598, 100.3795159 ,  94.34234528,  86.78805744,  97.79039692, 
                   81.77519378, 117.61282039, 109.08162784, 119.30896688, 98.3008706 ,  96.21075454, 
                   100.52072909, 127.48794967, 100.96706301, 104.24326515, 101.49111644])

sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
sampling_mean_std = sample_std / np.sqrt(len(sample))

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sampling_mean_std)
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sampling_mean_std)

print(f'[{lower_bound}, {upper_bound}]')
print("---------------------------------")

lower_bound, upper_bound = t.interval(0.95, len(sample) - 1, sample_mean, sampling_mean_std)
print(f'[{lower_bound}, {upper_bound}]')

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
Merkezi limit teoremine göre eğer ana kütle normal dağılmamışsa ancak n >= 30 koşulunu 
sağlayan örneklem dağılımlarının normal dağıldığı kabul edilmektedir. Yani 
örneklerimizdeki gibi anakütle ortalamasının tahmin edilmesi ve güven aralıklarının 
oluşturulması için şu iki koşuldan en az biri sağlanmalıdır:

1) Anakütle normal dağılmıştır ve örneklem dağılımı için n < 30 durumu söz konusudur.
2) Anakütle normal dağılmamıştır ve örneklem dağılımı için n >= 30 durumu söz konusudur.

                                
Pekiyi örneğimiz küçükse (tipik olarak < 30) ve ana kütle normal dağılmamışsa güven 
aralıklarını oluşturamaz mıyız? İşte bu tür durumlarda güven aralıklarının oluşturulması 
ve bazı hipotez testleri için "parametrik olmayan (nonparametric) yöntemler 
kullanılmaktadır. Ancak genel olarak parametrik olmayan yöntemler parametrik 
yöntemlere göre daha daha az güvenilir sonuçlar vermektedir. 
------------------------------------------------------------------------------------  
"""



#  --------------------- Verilerin Kullanıma Hazır Hale Getirilmesi ---------------------

"""
------------------------------------------------------------------------------------  
Veriler toplandıktan sonra hemen işleme sokulamayabilir. Veriler üzerinde çeşitli 
ön işlemler yapmak gerekebilir. Bu ön işlemlere "verilerin hazır hale getirilmesi 
(data preparation)" denilmektedir. Biz bu bölümde verilerin hazır hale getirilmesi 
için bazı temel işlemler üzerinde duracağız. Diğer bazı işlemler başka bölümlerde 
ele alınacaktır. Ancak verilerin kullanıma hazır hale getirilmesi tipik olarak 
aşağıdaki gibi süreçleri içermektedir:

1) Verilerin Temizlenmesi (Data Cleaning): Veriler eksiklikler içerebilir ya da 
geçersiz değerler içerebilir. Bazen de aşırı uç değerlerden (outliers) kurtulmak 
gerekebilir. Bu faaliyetlere verilerin temizlenmesi denilmektedir. Kusurlu veriler 
yapılan analizleri olumsuz yönde etkilemektedir.


2) Özellik seçimi (Feature Selection): Veri kümelerindeki tüm sütunlar bizim için 
anlamlı ve gerekli olmayabilir. Gereksiz sütunların atılıp gereklilerin alınması 
faaliyetine "özellik seçimi" denilmektedir. Örneğin bir veri tablosundaki kişinin 
"adı soyadı" sütunu veri analizi açısından genellikle (ama her zaman değil) bir 
fayda sağlamamaktadır. Bu durumda bu sütunların atılması gerekir. Bazen veriler 
tamamen geçersiz bir durum halinde de karşımıza gelebilmektedir. Örneğin oransal 
bir ölçeğe sahip sütunda yanlışlıkla kategorik bir veri bulunabilir.


3) Verilerin Dönüştürülmesi (Data Transformation): Kategorik veriler, tarih ve 
zaman verileri gibi veriler, resimler gibi veriler doğrudan işleme sokulamazlar. 
Bunların sayısal biçime dönüştürülmesi gerekir. Bazen veri kümesindeki sütunlarda 
önemli skala farklılıkları olabilmektedir. Bu skala farklılıkları algoritmaları 
olumsuz etkileyebilmektedir. İşte sütunların skalalarını birbirine benzer hale 
getirme sürecine "özellik ölçeklemesi (feature scaling)" denilmektedir.


4) Özellik Mühendisliği (Feature Engineering): Özellik mühendisliği veri tablosundaki 
sütunlardan olmayan başka sütunların oluşturulması sürecine denilmektedir. Yani 
özellik mühendisliği var olan bilgilerden hareketle önemli başka bilgilerin elde
edilmesidir. Örneğinin kişinin boy ve kilosu biliniyorsa biz vücut kitle endeksini 
tabloya ekleyebiliriz.


5) Boyutsal Özellik İndirgemesi (Dimentionality Feature Reduction): Veri kümesinde 
çok fazla sütun olmasının pek çeşitli dezavantajı olabilmektedir. Örneğin bu tür 
durumlarda işlem yükü artabilir. Gereksiz sütunlar kestirimi sürecini olumsuz
biçimde etkileyebilir. Fazla sayıda sütun kursumuzun ilerleyen zamanalarında sıkça 
karşılaşacağımız "overfitting" denilen yanlış öğrenmelere yol açabilir. O zaman 
sütunların sayısının azaltılması gerekebilir. İşte n tane sütunun k < n olmak üzere
k tane sütun haline getirilmesi sürecine boyutsal özellik indirgemesi denilmektedir. 
Bu konu kursumuzda ileride ayrı bir bölümde ele alınacaktır.


6) Verilerin Çoğaltılması (Data Augmentation): Elimizdeki veriler (veri kümesindeki 
satırlar) ilgili makine öğrenmesi yöntemini uygulayabilmek için sayı bakımından 
ya da nitelik bakımından yetersiz olabilir.  Eldeki verilerle (satırları kastediyoruz)
yeni verilerin oluşturulması (yeni satırların oluşturulması) sürecine "verilerin 
çoğaltılması (data augmentation)" denilmektedir. Örneğin bir resimden döndürülerek 
pek çok resim elde edilebilir. Benzer biçimde örneğin bir resmin çeşitli kısımlarından
yeni resimler oluşturulabilir. Özellik mühendisliğinin "sütun eklemeye yönelik", 
verilerin çoğaltılmasının ise "satır eklemeye yönelik" bir süreç olduğuna dikkat ediniz.

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
CSV dosyalarında iki virgül arasında hiçbir değer yoksa bu eksik veri anlamına 
geliyor olabilir. Böyle CSV dosyalarını Pandas'ın read_csv fonksiyonuyla okursak 
NaN (Not a Number) denilen özel numpy.float64 değerini elde ederiz. Örneğin "person.csv" 
isimli dosya şu içeriği sahip olsun:

Adı Soyadı,Kilo,Boy,Yaş,Cinsiyet
Sacit Bulut,78,172,34,Erkek
Ayşe Er,67,168,45,Kadın
Ahmet San,,182,32,Erkek
Macit Şen,98,156,65,Erkek
Talat Demir,85,,49,Erkek

Bu  dosyayı şöyle okuyalım:

import pandas as pd

df = pd.read_csv('person.csv')

Şöyle bir çıktı elde ederiz:

        Adı Soyadı  Kilo    Boy  Yaş Cinsiyet
    0  Sacit Bulut  78.0  172.0   34    Erkek
    1      Ayşe Er  67.0  168.0   45    Kadın
    2    Ahmet San   NaN  182.0   32    Erkek
    3    Macit Şen  98.0  156.0   65    Erkek
    4  Talat Demir  85.0    NaN   49    Erkek

Bir CSV dosyasında özel bazı sözcükler de eksik veri anlamına gelebilmektedir. 
Ancak hangi özel sözcüklerin eksik veri anlamına geldiği CSV okuyucuları arasında 
farklılıklar gösterebilmektedir. Örneğin Pandas'ın read_csv fonksiyonu şu özel 
sözcükleri "eksik veri" gibi ele almaktadır: NaN, '', '#N/A', '#N/A' 'N/A', '#NA', 
'-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 
'NULL', 'NaN', 'None', 'n/a', 'nan', ‘null’. CSV dosyalarında en çok karşımıza çıkan 
eksik veri gösterimlri şunlardır: '', NaN, nan, NA, null. 

read_csv fonksiyonu ayrıca na_values isimli parametresi yoluyla programcınn istediği 
yazıları da eksik veri olarak ele alabilmektedir. Bu parametreye yazılardan oluşan 
dolaşılabilir bir nesne girilmeldir. Örneğin:

df = pd.read_csv('person.csv', na_values=['NE'])

Burada read_csv yukarıdakilere ek olarak 'NE' yazısını da eksik olarak ele alacaktır. 

read_csv fonksiyonunun keep_default_na parametresi False olarak girilirse 
(bu parametrenin default değeri True biçimdedir) bu durumda yukarıda belirttiğimiz eksik 
veri kabul edilen yazılar artık eksik veri olarak kabul edilmeyecek onlara normal 
yazı muamalesi yapılacaktır. 

read_csv fonksiyonu bir süredir yazısal olan sütunların dtype özelliğini "object" 
olarak tutmaktadır. Yani bu tür sütunların her elemanı farklı türlerden olabilir. 
Bu tür sütunlarda NaN gibi eksik veriler söz konusu olduğunda read_csv fonksiyonu 
yine onu np.float64 NaN değeri olarak ele almaktadır.

Eksik veri işlemleri NumPy kütüphanesi ile loadtxt fonksiyonu kullanılarak da yapılabilir. 
Ancak loadtxt fonksiyonunun eksik verileri ele almak için kullanılması çok zahmetlidir. 
Bu nedenle biz kursumuzda CSV dosyalarını genellikle Pandas'ın read_csv fonksiyonu 
ile okuyacağız. 

------------------------------------------------------------------------------------  
Eksik verilerle çalışırken ilk yapılması gereken şey "eksikliğin analiz edilmesidir". 
Yani kaç tane eksik veri vardır? Kaç satırda ve hangi sütunlarda eksik veriler bulunmaktadır? 
Eksik veriler toplam verilerin yüzde kaçını oluşturmaktadır? Gibi sorulara yanıtlar 
aranmalıdır.

Eksik verilerin ele alınmasında iki temel strateji vardır:

1) Eksik verilerin bulunduğu satırı (nadiren de sütunu) tamamen atmak
2) Eksik verilerin yerine başka değerler yerleştirmek (imputation). 
                                                                   
------------------------------------------------------------------------------------  
Eksik verilerin bulunduğu satırın atılması yönteminde şunlara dikkat edilmelidir:

a) Eksik verili satırlar atıldığında elde kalan veri kümesi çok küçülecek midir?
b) Eksik verili satırlar atıldığında elde kalan veri kümesi yanlı (biased) hale gelecek midir?

Eğer bu soruların yanıtı "hayır" ise eksik verilerin bulunduğu satırlar tamamen atılabilir. 

Eksik veriler yerine bir verinin doldurulması işlemine İngilizce "imputation" denilmektedir. 
Eğer eksik verilerin bulunduğu satır (nadiren de sütun) atılamıyorsa "imputation" 
uygulanmalıdır.

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
 DataFrame nesnesi df olmak üzere, eksik veri analizinde şu kalıpları kullanabilirsiniz:

1) Sütunlardaki eksik verilerin miktarları şöyle bulunabilir:

df.isna().sum() ya da pd.isna(df).sum()

Pandas'taki isna fonksiyonu aynı zamanda DataFrame ve Series sınıflarında bir metot 
biçiminde de bulunmaktadır. isna bize bool türden bir DataFrame ya da Series nesnesi 
vermektedir. bool üzerinde sum işlemi yapıldığında False değerler 0 olarak, True
değerler 1 olarak işleme girer. Dolayısıyla yularıdaki işlemlerde biz sütunlardaki 
eksik veri sayılarını elde etmiş oluruz. isna fonksiyonunun diğer bir ismi isnull 
biçimindedir.

------------------------------------------------------------------------------------  
2) Eksik verilerin toplam sayısı şöyle bulunabilir:

df.isna().sum().sum() ya da pd.isna(df).sum().sum() 

isna fonksiyonu (ya da metodu) sütunsal temelde eksik verilerin sayılarını verdiğine 
göre onların toplamı da toplam eksik verileri verecektir.

------------------------------------------------------------------------------------  
3) Eksik verilerin bulunduğu satır sayısı şöyle elde edilebilir:

pd.isna(df).any(axis=1).sum() ya da df.isna().any(axis=1).sum()

any fonksiyonu ya da metodu bir eksen parametresi alarak satırsal ya da sütunsal 
işlem yapabilmektedir. any "en az bir True değer varsa True değerini veren hiç 
True değer yoksa False değerini veren" bir fonksiyondur. 

Yukarıdaki ifadede biz önce isna fonksiyonu ile eksik verileri matrissel bir biçimde 
DataFrame olarak elde ettik. Sonra da onun satırlarına any işlemi uyguladık. 
Dolayısıyla "en az bir eksik veri olan satırların" sayısını elde etmiş olduk.

------------------------------------------------------------------------------------  
4) Eksik verilerin bulunduğu satır indeksleri şöyle elde edilebilir:

df.index[pd.isna(df).any(axis=1)] ya da df.loc[df.isna().any(axis=1)].index

------------------------------------------------------------------------------------  
5) Eksik verilerin bulunduğu sütun isimleri şöyle elde edilebilir:

missing_columns = [name for name in df.columns if df[name].isna().any()]

Burada liste içlemi kullandık. Önce sütun isimlerini elde edip o sütun bilgilerini 
doğrudan indesklemeyle elde ettik. Sonra o sütunda en az bir eksik veri varsa o 
sütunun ismini listeye ekledik.

------------------------------------------------------------------------------------  
------------------------------------------------------------------------------------  
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

missing_columns = [colname for colname in df.columns if df[colname].isna().any()]
print(f'Eksik verilen bulunduğu sütunlar: {missing_columns}', end='\n\n')

missing_column_dist = df.isna().sum()
print('Eksik verilerin sütunlara göre dağılımı:')
print(missing_column_dist, end='\n\n')

missing_total = df.isna().sum().sum()
print(f'Eksik verilen toplam sayısı: {missing_total}')

missing_ratio = missing_total / df.size
print(f'Eksik verilen oranı: {missing_ratio}')

missing_rows = df.isna().any(axis=1).sum()
print(f'Eksik veri bulunan satırların sayısı: {missing_rows}')

missing_rows_ratio = missing_rows / len(df)
print(f'Eksik veri bulunan satırların oranı: {missing_rows_ratio}')


---------- Elde Edilen Çıktı ----------

Eksik verilen bulunduğu sütunlar: ['Car', 'BuildingArea', 'YearBuilt', 'CouncilArea']

Eksik verilerin sütunlara göre dağılımı:
Suburb              0
Address             0
Rooms               0
Type                0
Price               0
Method              0
SellerG             0
Date                0
Distance            0
Postcode            0
Bedroom2            0
Bathroom            0
Car                62
Landsize            0
BuildingArea     6450
YearBuilt        5375
CouncilArea      1369
Lattitude           0
Longtitude          0
Regionname          0
Propertycount       0
dtype: int64

Eksik verilen toplam sayısı: 13256
Eksik verilen oranı: 0.04648292306613367
Eksik veri bulunan satırların sayısı: 7384
Eksik veri bulunan satırların oranı: 0.543740795287187

------------------------------------------------------------------------------------  

------------------------------------------------------------------------------------  
Eksik verileri DataFrame nesnesinden silmek için DataFrame sınıfının dropna metodu 
kullanılabilir. Bu metotta default axis = 0'dır. Yani default durumda satırlar 
atılmaktadır. Ancak axis=1 parametresiyle sütunları da atabiliriz. Metot default 
durumda bize eksik verilerin atıldığı yeni bir DataFrame nesnesi vermektedir. Ancak 
metodun inplace parametresi True yapılırsa  nesne üzerinde atım yapılmaktadır. 


import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

print(f'Veri kümesinin boyutu: {df.shape}')
print("---------")

df_deleted_rows = df.dropna(axis=0)
print(f'Satır atma sonucundaki yeni boyut: {df_deleted_rows.shape}')

df_deleted_cols = df.dropna(axis=1)
print(f'Sütun atma sonucundaki yeni boyut: {df_deleted_cols.shape}')

-----------------------------------------------------------------------------------  
Eksik verilerin yerine başka değerlerin yerleştirilmesi işlemine "imputation" denilmektedir.
Kullanılan tipik imputation stratejiler şunlardır:

- Sütun sayısal ise Eksik verileri sütun ortalaması ile doldurma

- Sütun kategorik ya da sırasal ise eksik verileri mode değeri ile doldurma

- Eksik verilerin  sütunlarda uç değeler varsa medyan değeri ile doldurulması 

- Eksik verilerin yerine belli aralıkta ya da dağılımda rastgele değer yerleştirme yöntemi

- Eksik verilerin zaman serileri tarzında sütunlarda önceki ya da sonraki sütun değerleriyle doldurulması

- Eksik değerlerin regresyonla tahmin edilmesi yoluyla doldurulması

- Eksik değerlerin KNN (K-Nearest Neighbours) Yöntemi ile doldurulması

En çok uygulanan yöntem basitliği nedeniyle sütun ortalaması, sütun modu ya da 
sütun medyanı ile doldurma yöntemidir. 

-----------------------------------------------------------------------------------  
Bu veri kümesinde eksik veriler şu sütunlarda bulunmaktaydı: "Car", "BuildingArea", ,
"YearBuilt", "CouncilArea". Inputation işlemi için bu sütunların incelenmesi gerekir. 
"Car" sütunu ev için ayrılan otopark alanının kaç arabayı içerdiğini belirtmektedir. 
Bu sütunda ayrık küçük tamsayı değerler vardır. Burada imputation için sütun ortalaması 
alınabilir. Ancak bunların yuvarlanması daha uygun olabilir. Bu işlem şöyle yapılabilir:

impute_val = df['Car'].mean().round()
df['Car'] = df['Car'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)


Pandas'taki DataFrame ve Series sınıflarının mode metotları sonucu Series nesnesi 
biçiminde vermektedir. (Aynı miktarda yinelenen birden fazla değer olabileceği 
için bu değerlerin hepsinin verilmesi tercih edilmiştir.) Dolayısıyla biz bu Series 
nesnesinin ilk elemanını alarak değeri elde ettik.

impute_val = df['CouncilArea'].mode()
df['CouncilArea'] = df['CouncilArea'].fillna(impute_val[0])

-----------------------------------------------------------------------------------  
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

print(df.isna().sum())
print("-----------------------------")

impute_val = round( df['Car'].mean())
df['Car'] = df['Car'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)

impute_val = round(df['BuildingArea'].mean())
df['BuildingArea'] = df['BuildingArea'].fillna(impute_val)    # eşdeğeri df['Car'].fillna(impute_val, inplace=True)

impute_val = round(df['YearBuilt'].median())
df['YearBuilt'] = df['YearBuilt'].fillna(impute_val)    # eşdeğeri df['YearBuilt'].fillna(impute_val, inplace=True)

impute_val = df['CouncilArea'].mode()
df['CouncilArea'] = df['CouncilArea'].fillna(impute_val[0])

print(f'kontrol için: {df.isna().sum()}')

-----------------------------------------------------------------------------------  
"""

# scikit-learn

"""
-----------------------------------------------------------------------------------  
NumPy ve Pandas genel amaçlı kütüphanelerdir. SciPy ise matematik ve lineer cebir 
konularına odaklanmış genel bir kütüphanedir. Oysa scikit-learn makine öğrenmesi 
amacıyla tasarlanmış ve bu amaçla kullanılan bir kütüphanedir.

scikit-learn kütüphanesi yapay sinir ağları ve derin öğrenme ağlarına yönelik 
tasarlanmamıştır. Ancak kütüphane verilerin kullanıma hazır hale getirilmesine 
ilişkin öğeleri de içermektedir. scikit-learn içerisindeki sınıflar matematiksel 
ve istatistiksel ağırlıklı öğrenme yöntemlerini uygulamaktadır. scikit-learn 
kütüphanesinin import ismi sklearn biçimindedir.

-----------------------------------------------------------------------------------  
-----------------------------------------------------------------------------------  
scikit-learn kütüphanesi genel olarak "nesne yönelimli" biçimde oluşturulmuştur. 
Yani kütüphane daha çok fonksiyonlar yoluyla değil sınıflar yoluyla kullanılmaktadır. 
Kütüphanenin belli bir kullanım biçimi vardır. Öğrenme kolay olsun diye bu biçim 
değişik sınıflarda uygulanmıştır. Kütüphanenin tipik kullanım biçimi şöyledir:

1) Önce ilgili sınıf türünden nesne yaratılır. Örneğin sınıf SimpleImputer isimli sınıf olsun:

from sklearn.impute import SimpleImputer

si = SimpleImputer(...)    

-----------------------------------------------------------------------------------  
2) Nesne yaratıldıktan sonra onun bir veri kümesi ile eğitilmesi gerekir. Buradaki 
eğitme kavramı genel amaçlı bir kavramdır. Bu işlem sınıfların fit metotlarıyla yapılmaktadır. 
fit işlemi sonrasında metot birtakım değerler elde edip onu nesnenin içerisinde saklar. 
Yani fit metotları söz konusu veri kümesini ele alarak oradan gerekli faydalı 
bilgileri elde etmektedir. Örneğin:

si.fit(dataset)

fit işlemi genel olarak transform için kullanılacak bilgilerin elde edilmesi işlemini 
yapmaktadır. DOlayısıyla fit işleminden sonra artık nesnenin özniteliklerini kullanabiliriz. 
    
-----------------------------------------------------------------------------------  
3) fit işleminden sonra fit işlemiyle elde edilen bilgilerin bir veri kümesine  
uygulanması gerekir. Bu işlem de transform metotlarıyla yapılmaktadır. Bir veri 
kümesi üzerinde fit işlemi uygulayıp birden fazla veri kümesini transform edebiliriz. 

result1 = si.transform(data1)
result2 = si.transform(data2)
...

fit ve transform metotları bizden bilgiyi NumPy dizisi olarak, Pandas, Series ya da 
DataFrame nesnesi olarak ya da Python listesi olarak alabilmektedir. fit metotları 
nesnenin kendisine geri dönmekte ve transform metotları da transform edilmiş
NumPy dizilerine geri dönmektedir.

-----------------------------------------------------------------------------------  
4) Bazen fit edilecek veri kümesi ile transform edilecek veri kümesi aynı olur. Bu 
durumda önce fit, sonra transform metotlarını çağırmak yerine bu iki işlem sınıfların 
fit_transform metotlarıyla tek hamlede de yapılabilir. Örneğin:

result = si.fit_transform(dataset)

-----------------------------------------------------------------------------------  
5) Ayrıca sınıfların fit işlemi sonucunda oluşan bilgileri almak için kullanılan 
birtakım örnek öznitelikleri ve metotları da olabilmektedir.

-----------------------------------------------------------------------------------  
6) Sınıfların bazen inverse_transform metotları da bulunmaktadır. inverse_transform 
metotları transform işleminin tersini yapmaktadır. Yani transform edilmiş bilgileri 
alıp onları transform edilmemiş hale getirmektedir. 

Genel olarak scikit-learn kütüphanesi hem NumPy hem de Pandas nesnelerini desteklemektedir. 
fit, transform ve fit_transform metotları genel olarak iki boyutlu bir veri kümesini 
kabul etmektedir. Bunun amacı genelleştirmeyi sağlamaktadır. Bu nedenle örneğin 
biz bu metotlara Pandas'ın Series nesnelerini değil DataFrame nesnelerini verebiliriz. 
-----------------------------------------------------------------------------------  
"""

"""
-----------------------------------------------------------------------------------  
# SimpleImputer 

Şimdi eksik verilerin doldurulması işleminde kolaylık sağlayan scikit-learn kütüphanesindeki 
SimpleImputer sınıfını görelim. Sınıfın kullanılması yukarıda ele alınan kalıba 
uygundur. Önce nesne yaratılır. Sonra fit işlemi yapılır. Asıl dönüştürme işlemini 
transform yapmaktadır. SimpluImputer sınıfı türünden nesne yaratılırken __init__ 
metodunda birtakım parametreler belirtilebilmektedir. Bunların çoğu zaten default 
değer almış durumdadır. Örneğin strategy parametresi default olarak 'mean' durumdadır. 
Bu impute işleminin sütunun hangi bilgisine göre yapılacağını belirtir. 'mean' 
dışında şu stratejiler bulunmaktadır:

'median'
'most_frequent'
'constant'

'constant' stratejisi belli bir değerle doldurma işlemini yapar. Eğer bu strateji 
seçilirse doldurulacak değerin fill_value parametresiyle belirtilmesi gerekir. 
Diğer parametreler için sınıfın dokümantasyonuna bakabilirsiniz

-----------------------------------------------------------------------------------  
SimpleImputer sınıfının fit, transform ve fit_transform metotları İKİ BOYUTLU bir 
dizi almaktadır. Yani bu metotlara bir NumPy dizisi geçirecekseniz onun iki boyutlu 
olması gerekir. Pandas'ın Series nesnelerinin tek boyutlu bir dizi belirttiğini 
anımsayınız. Bu durumda biz bu metotlara Series nesnesi geçemeyiz. Ancak DataFrame 
nesneleri iki boyutlu dizi belirttiği için DataFrame nesnelerini geçebiliriz. Eğer 
elimizde tek boyutlu bir dizi varsa onu bir sütundan n satırdan oluşan iki boyutlu 
bir diziye dönüştürmeliyiz. Bunun için NumPy ndarray sınıfının reshape metodunu 
kullanabilirsiniz.

----------------------------------------------------------------------------------- 
SimpleImputer nesnesi yaratılırken doldurma stratejisi nesnenin yaratımı sırasında 
verilmektedir. Yani nesne başta belirtilen stratejiyi uygulamaktadır. Ancak veri 
kümelerinin değişik sütunları değişik stratejilerle doldurulmak istenebilir. Bunun 
için birden fazla SimpleImputer nesnesi yaratılabilir. Örneğin:

si1 = SimpleImputer(strategy='mean')
si2 = SimpleImputer(strategy='median')
...

Ancak bunun yerine SimpleImputer sınıfının set_params metodu da kullanılabilir. 
Bu metot önceden belirlenmiş parametreleri değiştirmekte kullanılmaktadır. Örneğin:

si = SimpleImputer(strategy='mean')
...
si.set_params(strategy='median')
...

-----------------------------------------------------------------------------------      
SimpleImputer sınıfında yukarıda belirttiğimiz gibi fit metodu asıl doldurma işlemini 
yapmaz. Doldurma işlemi için gereken bilgileri elde eder. Yani örneğin:

from sklearn.impute import SimpleImputer
a = np.array([1, 1, None, 4, None]).reshape(-1, 1)

si = SimpleImputer(strategy='mean')
si.fit(a)

Burada fit metodu aslında yalnızca bu a dizisindeki sütunların ortalamalarını elde 
etmektedir. (Örneğimizde tek bir sütun var). Biz fit yaptığımız bilgiyi transform 
etmek zorunda değiliz. Örneğin:

b = np.array([1, 1, None, 4, None]).reshape(-1, 1)
result = si.transform(b)

Biz şimdi burada a'dan elde ettiğimiz ortalama 3 değeri ile bu b dizisini doldurmuş 
oluruz. Tabii genellikle bu tür durumlarda fit yapılan dizi ile transform yapılan 
dizi aynı dizi olur.

-----------------------------------------------------------------------------------  
-----------------------------------------------------------------------------------  
scikit-learn kütüphanesindeki pek çok sınıf aynı anda birden fazla sütun üzerinde 
işlem yapabilmektedir. Bu nedenle bu sınıfların fit ve transform metotları bizden 
iki boyutlu dizi istemektedir. tranform metotları da bize iki iki boyutlu dizi 
geri döndürmektedir.

Örneğin SimpleImputer sınıfına biz fit işleminde iki boyutlu bir dizi veriririz. 
Bu drumda fit metodu her sütunu diğerinden ayrı bir biçimde ele alır ve o sütunlara 
ilişkin bilgileri oluşturur. Örneğin biz fit metoduna aşağıdaki gibi iki boyutlu 
bir dizi vermiş olalım:

1       4
None    7
5       None
3       8
9       2

Stratejinin "mean" olduğunu varsayalım. Bu durumda fit metodu her iki sütunun da 
ortalamasını alıp nesnenin içerisinde saklayacaktır. Biz artık transform metoduna 
iki boyutlu iki sütundan oluşan bir dizi verebiliriz. Bu transform metodu bizim 
verdiğimiz dizinin ilk sütunununu fit ettiğimiz dizinin ilk sütunundan elde ettiği 
bilgiyle, ikinci sütununu fit ettiğimiz dizinin ikinci sütunundan elde ettiği bilgiyle 
dolduracakır. 

İşte sckit-learn sınıflarının fit ve transform metotlarına biz iki boyutlu diziler 
veririz. O da bize iki boyutlu diziler geri döndürür. Eğer elimizde tek boyutlu 
bir dizi varsa biz onu reshape metoduyla iki boyutlu hale getirerek fit ve transform 
metotlarına vermeliyiz.

-----------------------------------------------------------------------------------  
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

si = SimpleImputer(strategy='mean')
df[['Car','BuildingArea']] = np.round(si.fit_transform(df[['Car','BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = si.fit_transform(df[['YearBuilt']])

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])

-----------------------------------------------------------------------------------  
scikit-learn kütüphanesinde sklearn.impute modülünde aşağıdaki imputer sınıfları 
bulunmaktadır:

SimpleImputer
IterativeImputer
MissingIndicator
KNNImputer

Buradaki KNNImputer sınıfı "en yakın komşuluk" yöntemini kullanmaktadır. Bu konu 
ileride ele alınacaktır. IterativeImputer sınıfı ise regresyon yaparak doldurulacak 
değerleri oluşturmaktadır. Örneğin biz bu sınıfa fit işleminde 5 sütunlu bir dizi 
vermiş olalım. Bu sınıf bu sütunların birini çıktı olarak dördünü girdi olarak ele 
alıp girdilerden çıktıyı tahmin edecek doğrusal bir model oluşturmaktadır. Yani bu 
sınıfta doldurulacak değerler yalnızca doldurmanın yapılacağı sütunlar dikkate 
alınarak değil diğer sütunlar da dikkate alınarak belirlenmektedir. 

Örneğin MHS veri kümesinde evin metrakaresi bilinmediğinde bu eksik veriyi ortalama 
metrakareyle doldurmak yerine "bölgeyi", "binanın yaşını" da dikkate alarak doldurmak 
isteyebiliriz. Ancak yukarıda da belirttiğimiz gibi genellikle bu tür karmaşık 
imputation işlemleri çok fazla kullanılmamaktadır. 

-----------------------------------------------------------------------------------  
"""

"""
-----------------------------------------------------------------------------------  
Verilerin kullanıma hazır hale getirilmesi sürecinin en önemli işlemlerinden biri 
de "kategorik (nominal)" ve "sıralı (ordinal)" sütunların sayısal biçime dönüştürülmesidir. 
Çünkü makine öğrenmesi algoritmaları veriler üzerinde "toplama", "çarpma" gibi 
işlemler yaparlar. Dolayısıyla kategorik veriler böylesi işlemlere sokulamazlar. 
Bunun için önce onların sayısal biçime dönüştürülmeleri gerekir. 

Genellikle bu dönüştürme "eksik verilerin ele alınması" işleminden daha sonra 
yapılmaktadır. Ancak bazen önce kategorik dönüştürmeyi yapıp sonra imputation 
işlemi de yapılabilir.

-----------------------------------------------------------------------------------  
Kategorik verilerin sayısal biçime dönüştürülmesi genellikle her kategori (sınıf) 
için 0'dan başlayarak artan bir tamsayı karşı düşürerek yapılmaktadır. Örneğin bir 
sütunda kişilerin renk tercihleri olsun. Ve sütun içeriği aşağıdaki gibi olsun:

Kırmızı
Mavi
Kırmızı
Yeşil
Mavi
Yeşil
...

Biz şimdi burada bu kategorik alanı Kırmızı = 0, Mavi = 1, Yeşil = 2 biçiminde 
sayısal hale dönüştürebiliriz. Bu durumda bu sütun şu hale gelecektir:

0
1
0
2
1
2
....

Örneğin biz bu işlemi yapan bir fonksiyon yazabiliriz. Fonksiyonun birinci parametresi 
bir DataFrame olabilir. İkinci parametresi ise hangi sütun sayısallaştıtılacağına 
ilişkin sütun isimlerini belirten dolaşılabilir bir nesne olabilir. Fonksiyonu 
şöyle yazabiliriz:

def category_encoder(df, colnames):
for colname in colnames:
    labels = df[colname].unique()
    for index, label in enumerate(labels):
        df.loc[df[colname] == label, colname] = index

Burada biz önce sütun isimlerini tek tek elde etmek için dış bir döngü kullandık. 
Sonra ilgili sütundaki "tek olan (unique)" etiketleri (labels) elde ettik. Sonra 
bu etiketleri iç bir döngüde dolaşarak sütunda ilgili etiketin bulunduğu satırlara
onları belirten sayıları yerleştirdik. 

----------------------------------------------------------------------------------- 

Test işlemi için aşağıdaki gibi "test.csv" isimli bir CSV dosyasını kullanabiliriz:
    
AdıSoyadı,Kilo,Boy,Yaş,Cinsiyet,RenkTercihi
 Sacit Bulut,78,172,34,Erkek,Kırmızı
 Ayşe Er,67,168,45,Kadın,Yeşil
 Ahmet San,85,182,32,Erkek,Kırmızı
 Macit Şen,98,192,65,Erkek,Mavi
 Talat Demir,85,181,49,Erkek,Yeşil
 Sibel Ünlü,72,172,34,Kadın,Mavi
 Ali Serçe,75,165,21,Erkek,Yeşil

Test kodu da şöyle olabilir:

    
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')
print(df)
print('-------------------------------------------------')

def label_encode(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index
            
            
label_encode(df, ['RenkTercihi', 'Meslek'])
print(df)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df.loc[df[colname] == label, colname]: Bu kısım, colname sütunundaki değerleri 
label ile eşit olan satırları seçer. df.loc[] metodunun ilk bölümü 
(df[colname] == label) hangi satırların seçileceğini belirlerken, ikinci bölüm 
(colname) hangi sütunun değerlerinin değiştirileceğini belirtir.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Şöyle bir çıktı elde edilmiştir:

        AdıSoyadı  Kilo  Boy  Yaş Cinsiyet RenkTercihi
0  Sacit Bulut    78  172   34        0            0
1      Ayşe Er    67  168   45        1            1
2    Ahmet San    85  182   32        0            0
3    Macit Şen    98  192   65        0            2
4  Talat Demir    85  181   49        0            1
5   Sibel Ünlü    72  172   34        1            2    

----------------------------------------------------------------------------------- 
Aşağıda da MHS veri kümesi üzerinde aynı işlem yapılmıştır.

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])

def category_encoder(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index
    
category_encoder(df, ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname'])
print(df)

----------------------------------------------------------------------------------- 
# LabelEncoder

Aslında yukarıdaki işlem scikit-learn kütüphanesindeki preprocessing modülünde 
bulunan LabelEncoder sınıfıyla yapılabilmektedir. LabelEncoder sınıfının genel 
çalışma biçimi scikit-learn kütüphanesinin diğer sınıflarındaki gibidir. Ancak bu 
sınıfın fit ve transform metotları "TEK" boyutlu bir numpy dizisi ya da Series nesnesi 
almaktadır. (Halbuki yukarıda da belirttiğimiz gibi genel olarak fit ve transform 
metotlarının çoğu iki boyutlu dizileri ya da DataFrame nesnelerini alabilmektedir.)

fit metodu yukarıda bizim yaptığımız gibi unique elemanları tespit edip bunu nesne 
içerisinde saklamaktadır. Asıl dönüştürme işlemi transform metoduyla yapılmaktadır. 
Tabii eğer fit ve transform metotlarında aynı veriler kullanılacaksa bu işlemler 
tek hamlede fit_transform metoduyla da yapılabilir. Örneğin yukarıdaki "test.csv" 
veri kümesindeki "Meslek" ve "RenkTercihi" sütunlarını kategorik olmaktan çıkartıp 
sayısal biçime şöyle dönüştürebiliriz:
    
-----------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

le = LabelEncoder()

transformed_data = le.fit_transform(df['RenkTercihi'])
df['RenkTercihi'] = transformed_data

transformed_data = le.fit_transform(df['Meslek'])
df['Meslek'] = transformed_data
print(df)

-----------------------------------------------------------------------------------
Aşağıda MHS veri kümesindeki gerekli sütunlar LabelEncoder sınıfı ile sayısal 
biçime dönüştürülmüştür.


import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/melb_data.csv')
print(df, end='\n\n')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

si.set_params(strategy='median')
df[['YearBuilt']] = np.round(si.fit_transform(df[['YearBuilt']]))

si.set_params(strategy='most_frequent')
df[['CouncilArea']] = si.fit_transform(df[['CouncilArea']])
    

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for colname in ['Suburb', 'SellerG', 'Method', 'CouncilArea', 'Regionname']:
    df[colname] = le.fit_transform(df[colname])

print(df)

-----------------------------------------------------------------------------------
LabelEncoder sınıfının inverse_transform metodu ters işlemi yapmaktadır. Yani bir 
kez fit işlemi yapıldıktan sonra nesne zaten hangi etiketlerin hangi sayısal değerlere 
karşı geldiğini kendi içerisinde tutmaktadır. Böylece biz sayısal değer verdiğimizde 
onun yazısal karşılığını inverse_transform ile elde edebiliriz


import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')
print(df, end='\n\n')

le = LabelEncoder()

transformed_data = le.fit_transform(df['RenkTercihi'])
df['RenkTercihi'] = transformed_data

label_names = le.inverse_transform(transformed_data)
print(label_names, end='\n\n')


transformed_data = le.fit_transform(df['Cinsiyet'])
df['Cinsiyet'] = transformed_data

label_names = le.inverse_transform(transformed_data)
print(label_names)

-----------------------------------------------------------------------------------

-----------------------------------------------------------------------------------
Aslında kategorik verilerin 0'dan itibaren birer tamsayı ile numaralandırılması 
iyi bir teknik değildir. Kategorik verilen "one hot encoding" denilen biçimde 
sayısallaştırılması doğru tekniktir. Biz "one hot encoding" dönüştürmesini izleyen 
paragraflarda ele alacağız.

Sıralı (ordinal) verilerin sayısal biçime dönüştürülmesi tipik olarak düşük sıranın 
düşük numarayla ifade edilmesi biçiminde olabilir. Örneğin "Eğitim Durumu", "ilkokul", 
"ortaokul", "lise", "üniversite" biçiminde dört kategoride sıralı bir bilgi olabilir. 
Biz de bu bilgilere sırasıyla birer numara vermek isteyebiliriz. Örneğin:

İlkokul     --> 0
Ortaokul    --> 1
Lise        --> 2
Üniversite  --> 3

scikit-learn içerisindeki LabelEncoder sınıfı bu amaçla kullanılamamaktadır. Çünkü 
LabelEncoder etiketlere rastgele numara vermektedir. scikit-learn içerisinde bu 
işlemi pratik bir biçimde yapan hazır bir sınıf bulunmamaktadır. Gerçi scikit-learn 
içerisinde OrdinalEncoder isimli bir sınıf vardır ama o sınıf bu tür amaçları 
gerçekleştirmek için tasarlanmamıştır. 

-----------------------------------------------------------------------------------
OrdinalEncoder sınıfının kullanımı benzerdir. Önce fit sonra transform yapılır. 
Eğer fit ve transform metodunda aynı veri kümesi kullanılacaksa fit_transform metodu 
ile bu iki işlem bir arada yapılabilir. OrdinalEncoder sınıfının categories_ örnek 
özniteliği oluşturulan kategorileri NumPy dizisi olarak vermektedir. Sınıfın 
n_features_in_ örnek özniteliği ise fit işlemine sokulan sütunların sayısını vermektedir.


OrdinalEncoder sınıfı encode edilecek sütunları eğer onlar yazısal biçimdeyse 
"lexicographic" sıraya göre numaralandırmaktadır. (Yani sözlükte ilk gördüğü 
kategoriye düşük numara vermektedir. Tabii bu işlem UNICODE tabloya göre yapılmaktadır.) 
Kategorilere bizim istediğimiz numaraları vermemektedir. Ayrıca bu sınıfın fit ve 
transform metotları iki boyutlu nesneleri kabul etmektedir. Bu bağlamda OrdinalEncoder 
sınıfının LabelEncoder sınıfındne en önemli farklılığı OrdinalEncoder sınıfının 
birden fazla sütunu (özelliği) kabul etmesidir. Halbuki LabelEncoder sınıfı tek bir 
sütunu (özelliği) dönüştürmektedir.

-----------------------------------------------------------------------------------
Örneğin "test.csv" veri kümesi için:
    
Burada "Meslek" ve "RenkTercihi" kategorik (nominal) ölçekte sütunlardır. "EğitimDurumu" 
sütunu kategorik ya da sıralı olarak ele alınabilir. Eğer biz İlkokul = 0, Ortaokul = 1, 
Lise = 2, Üniversite = 3 biçiminde sıralı ölçeğe ilişkin bir kodlama yapmak istersek 
bunu LabelEncoder ya da OrdinalEncoder ile sağlayamayız. Örneğin:

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder 

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')
print(df, end='\n\n')

oe = OrdinalEncoder()
transformed_data = oe.fit_transform(df[['Meslek', 'RenkTercihi', 'Eğitim Durumu']])
df[['Meslek', 'RenkTercihi', 'EğitimDurumu']] = transformed_data 

 Buradan şöyle bir DataFrame elde edilecektir:

        AdıSoyadı  Kilo  Boy  Yaş  Cinsiyet  RenkTercihi  EğitimDurumu
0  Sacit Bulut    78  172   34       0.0           0.0            3.0
1      Ayşe Er    67  168   45       1.0           2.0            1.0
2    Ahmet San    85  182   32       0.0           0.0            3.0
3    Macit Şen    98  192   65       0.0           1.0            0.0
4  Talat Demir    85  181   49       0.0           2.0            2.0
5   Sibel Ünlü    72  172   34       1.0           1.0            1.0
6    Ali Serçe    75  165   21       0.0           2.0            3.0
                                
Bunu OridinalEncoder sınıfı le kodlamaya çalışırsak muhtemelen tam istediğimiz 
gibi bir kodlama yapamayız.   

-----------------------------------------------------------------------------------
Bu durumda eğer kategorilere istediğiniz gibi değer vermek istiyorsanız bunu manuel 
bir biçimde yapabilirsiniz. Örneğin veri kümesi read_csv fonksiyonuyla okunurken 
converters parametresi yoluyla hemen dönüştürme yapılabilir. read_csv ve load_txt 
fonksiyonlarında converters parametresi bir sözlük nesnesi almaktadır. Bu sözlük 
nesnesi hangi sütun değerleri okunurken hangi dönüştürmenin yapılacağını belirtmektedir. 
Buradaki sözlüğün her elemanının anahtarı bir sütun indeksinden ya da sütun indeksinden 
değeri ise o sütunun dönüştürülmesinde kullanılacak fonksiyondan oluşmaktadır. Tabii 
bu fonksiyon lambda ifadesi olarak da girilebilir. Örneğin:

import pandas as pd
df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv', converters={'EğitimDurumu': lambda s: {'İlkokul': 0, 'Ortaokul': 1, 'Lise': 2, 'Üniversite': 3}[s]})

Elde edilecek DataFrame nesnesi şöyle olacaktır:

        AdıSoyadı  Kilo  Boy  Yaş Cinsiyet RenkTercihi  EğitimDurumu
0  Sacit Bulut    78  172   34    Erkek      Kırmızı              0
1      Ayşe Er    67  168   45    Kadın        Yeşil              1
2    Ahmet San    85  182   32    Erkek      Kırmızı              0
3    Macit Şen    98  192   65    Erkek         Mavi              2
4  Talat Demir    85  181   49    Erkek        Yeşil              3
5   Sibel Ünlü    72  172   34    Kadın         Mavi              1
6    Ali Serçe    75  165   21    Erkek        Yeşil              0

-----------------------------------------------------------------------------------

-----------------------------------------------------------------------------------
# one hot encoding

Kategorik verilerin 0'dan itibaren LabelEncoder ya da OridnalEncoder sınıfı ile 
sayısallaştırılması iyi bir fikir değildir. Çünkü bu durumda sanki veri "sıralı (ordinal)" 
bir biçime sokulmuş gibi olur. Pek çok algoritma bu durumdan olumsuz yönde etkilenmektedir.

kategorik olguların birden fazla sütunla ifade edilmesi yoluna gidilmektedir. Kategorik 
verilerin birden fazla sütunla ifade edilmesinde en yaygın kullanılan yöntem "one hot encoding" 
denilen yöntemdir. Bu yöntemde sütundaki kategorilerin sayısı hesaplanır. Veri 
tablosuna bu sayıda sütun eklenir. "One hot" terimi "bir grup bitten hepsinin sıfır 
yalnızca bir tanesinin 1 olma durumunu" anlatmaktadır. İşte bu biçimde n tane kategori 
n tane sütunla ifade edilir. Her kategori yalnızca tek bir sütunda 1 diğer sütunlarda 
0 olacak biçimde kodlanır.

-----------------------------------------------------------------------------------

    RenkTercihi
    -----------
    Kırmızı
    Kırmızı
    Mavi
    Kırmızı
    Yeşil
    Mavi
    Yeşil
    ...

Burada 3 renk olduğunu düşünelim. Bunun "one hot encoding" dönüştürmesi şöyle olacaktır:

Kırmızı     Mavi        Yeşil
1           0           0
1           0           0
0           1           0
1           0           0
0           0           1
0           1           0
0           0           1
...

Eğer sütundaki kategori sayısı 2 tane ise böyle sütunlar üzerinde "one hot encoding" 
uygulamanın bir faydası yoktur. Bu tür ikili sütunlar 0 ve 1 biçiminde kodlanabilir. 
(Yani bu işlem LabelEncoder sınıfyla yapılabilir).

-----------------------------------------------------------------------------------
"One hot encoding" yapmanın çok çeşitli yöntemleri vardır. Örneğin scitkit-learn 
içerisindeki preprocessing modülünde bulunan OneHotEncoder sınıfı bu işlem için 
kullanılabilir. Sınıfın genel kullanımı diğer sckit-learn sınıflarında olduğu gibidir. 
Yani önce fit işlemi sonra transform işlemi yapılır. fit işleminden sonra sütunlardaki 
"tek olan (unique)" elemanlar sınıfın categories_ örnek özniteliğine NumPy dizilerinden 
oluşan bir liste biçiminde kaydedilmektedir.

"One hot encoding" yapılırken DataFrame içerisine eski sütunu silip yeni sütunlar 
eklenmelidir. DataFrame sütununa bir isim vererek atama yapılırsa nesne o sütunu 
zaten otomatik eklemektedir. "One hot encoding" ile oluşturulan sütunların isimleri 
önemli değildir. Ancak OneHotEncoder nesnesi önce sütunu np.unique fonksiyonuna 
sokmakta ondan sonra o sırada encoding işlemi yapmaktadır. NumPy'ın unique fonksiyonu 
aynı zamanda sıraya dizme işlemini de zaten yapmaktadır. Dolayısıyla OneHotEncoder 
aslında kategorik değerleri alfabetik sıraya göre sütunsal olarak dönüştürmektedir.

OneHotEncoder sınıfının fit ve transform metotları çok boyutlu dizileri kabul 
etmektedir. Bu durumda biz bu metotlara Pandas'ın Series nesnesini değil DataFrame 
nesnesini vermeliyiz.

OneHotEncoder nesnesini yaratırken "sparse_output" parametresini False biçimde vermeyi 
unutmayınız. (Bu parametrenin eski ismi yalnızca "sparse" biçimindeydi). Çünkü bu 
sınıf default olarak transform edilmiş nesneyi "seyrek matris (sparse matrix)" 
olarak vermektedir. Elemanlarının büyük çoğunluğu 0 olan matrislere "seyrek matris 
(sparse matrix)" denilmektedir. Bu tür matrisler "fazla yer kaplamasın diye" 
sıkıştırılmış bir biçimde ifade edilebilmektedir. 

Yine OneHotEncoder nesnesi yaratılırken parametre olarak dtype türünü belirtebiliriz. 
Default dtype türü np.float64 alınmaktadır. Matris seyrek formda elde edilmeyecekse 
bu dtype türünü "uint8" gibi en küçük türde tutabilirsiniz. max_categories parametresi 
kategori sayısını belli bir değerde tutmak için kullanılmaktadır. Bu durumda diğer 
tüm kategoriler başka bir kategori oluşturmaktadır

-----------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

ohe = OneHotEncoder(sparse_output=False, dtype='uint8')
transformed_data = ohe.fit_transform(df[['RenkTercihi']])

df.drop(['RenkTercihi'], axis=1, inplace=True)

df[ohe.categories_[0]] = transformed_data

# ohe.categories_  -----> [ array(['Kırmızı', 'Mavi', 'Yeşil'], dtype=object) ]
print(df)

-----------------------------------------------------------------------------------
DataFrame nesnesine yukarıdaki gibi birden fazla sütun eklerken dikkat etmek gerekir. 
Çünkü tesadüfen bu kategori isimlerine ilişkin sütunlardan biri zaten varsa o sütun 
yok edilip yerine bu kategori sütunu oluşturulacaktır. Bunu engellemek için 
oluşturacağınız kategori sütunlarını önek vererek isimlendirebilirsiniz. Ön ek 
verirken orijinal sütun ismini kullanırsanız bu durumda çakışma olmayacağı garanti 
edilebilir. Yani örneğin RenkTercihi sütunu için "Kırmızı", "Mavi" "Yeşil" isimleri 
yerine "RenkTercihi_Kırmızı", "RenkTercihi_Mavi" ve "RenkTercihi_Yeşil" isimlerini 
kullanabilirsiniz. Bu biçimde isim elde etmek "liste içlemiyle" oldukça kolaydır. 
Örneğin:

category_names = ['RenkTercihi_' + category for category in ohe.categories_[0]]


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

ohe = OneHotEncoder(sparse_output=False, dtype='uint8')

transformed_data = ohe.fit_transform(df[['RenkTercihi']])

df.drop(['RenkTercihi'], axis=1, inplace=True)

category_names = ['RenkTercihi_' + category for category in ohe.categories_[0]]

df[category_names] = transformed_data

print(df)
    
-----------------------------------------------------------------------------------
Burada "RenkTercihi"nin yanı sıra "Eğitim Durumu" de kategorik bir sütundur. 
Bunun her ikisini birden tek hamlede "one hot encoding" işlemine sokabiliriz:

ohe = OneHotEncoder(sparse=False, dtype='uint8')
transformed_data = ohe.fit_transform(df[['RenkTercihi', 'Eğitim Durumu']])

df.drop(['RenkTercihi', 'Eğitim Durumu'], axis=1, inplace=True)

categories1 = ['RenkTercihi_' + category for category in ohe.categories_[0]]
categories2 = ['Eğitim Durumu_' + category for category in ohe.categories_[1]]

df[categories1 + categories2] = transformed_data

----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
# get_dummies

One hot encoding yapmanın diğer bir yolu Pandas kütüphanesindeki get_dummies 
fonksiyonunu kullanmaktadır. get_dummies fonksiyonu bizden bir DataFrame, Series 
ya da dolaşılabilir herhangi bir nesneyi alır. Eğer biz get_dummies fonksiyonuna 
bütün bir DataFrame geçirirsek fonksiyon oldukça akıllı davranmaktadır. Bu durumda 
fonksiyon DataFrame nesnesi içerisindeki yazısal sütunları tespit eder. Yalnızca 
yazısal sütunları "one hot encoding" işlemine sokar ve bize yazısal sütunları 
dönüştürülmüş yeni bir DataFrame nesnesi verir. Pandas ile çalışırken bu fonksiyon 
çok kolaylık sağlamaktadır.

Biz aslında get_dummies fonksiyonu yoluyla yapmış olduğumuz işlemleri tek hamlede 
yapabiliriz:

    
import pandas as pd
df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')    

transformed_df = pd.get_dummies(df, dtype='uint8')
print(transformed_df )


Burada biz tek hamlede istediğimiz dönüştürmeyi yapabildik. Bu dönüştürmede yine 
sütun isimleri orijinal sütun isimleri ve kategori isimleriyle örneklendirilmiştir. 
Eğer isterse programcı "prefix" parametresi ile bu öneki değiştirebilir, "prefix_sep"
parametresiyle de '_' karakteri yerine başka birleştirme karakterlerini kullanabilir. 

transformed_df = pd.get_dummies(df, columns=['RenkTercihi', 'AdıSoyadı'], dtype='uint8', prefix=['R', 'AD'], prefix_sep='-')

get_dummies fonksiyonu default durumda sparse olmayan bool türden bir DataFrame 
nesnesi vermektedir. Ancak get_dummies fonksiyonunda "dtype" parametresi belirtilerek 
"uint8" gibi bir türden çıktı oluşturulması sağlanabilmektedir. 

----------------------------------------------------------------------------------
Biz bir DataFrame nesnesinin tüm yazısal sütunlarını değil bazı yazısal sütunlarını 
da "one hot encoding" işlemine sokmak isteyebiliriz. Bu durumda fonksiyon DataFrame 
nesnesinin diğer sütunlarına hiç dokunmamaktadır. Örneğin:

transformed_df = pd.get_dummies(df, columns=['RenkTercihi'], dtype='uint8')


Biz burada yalnızca DataFrame nesnesinin "RenkTercihi" sütununu "one hot encoding" 
yapmış olduk. get_dummies fonksiyonun zaten "one hot encoding" yapılan sütunu 
sildiğine dikkat ediniz. Bu bizim genellikle istediğimiz bir şeydir. Yukarıdaki 
örnekte "test.csv" dosyasında "AdıSoyadı" sütunu yazısal bir sütundur. Dolayısıyla 
default durumda bu sütun da "one hot encoding" işlemine sokulacaktır. Bunu engellemek 
için "columns" parametresinden faydalanabiliriz ya da baştan o sütunu atabiliriz. 
Örneğin:

transformed_df = pd.get_dummies(df.iloc[:, 1:], dtype='uint8')

----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
# to_categorical

Diğer bir "one hot encoding" uygulama yöntemi de "tensorflow.keras" kütüphanesindeki 
"to_categorical" fonksiyonudur. Bazen zaten Keras ile çalışıyorsak bu fonksiyonu 
tercih edebilmekteyiz. to_categorical fonksiyonunu kullanmadan önce kategorik sütunun 
sayısal biçime dönüştürülmüş olması gerekmektedir. Yani biz önce sütun üzerinde 
eğer sütun yazısal ise LabelEncoder işlemini uygulamalıız. to_categorical fonksiyonu 
aynı anda birden fazla sütunu "one hot encoding" yapamamaktadır. Bu nedenle diğer 
seçeneklere göre kullanımı daha zor bir fonksiyondur. to_categorical fonksiyonu 
Keras kütüphanesindeki utils isimli modülde bulunmaktadır.
----------------------------------------------------------------------------------

import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv') 

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

transformed_color = le.fit_transform(df['RenkTercihi'])
transformed_occupation = le.fit_transform(df['Meslek'])

ohe_color = to_categorical(transformed_color)
ohe_occupation = to_categorical(transformed_occupation)

color_categories = ['RenkTercihi_' + color for color in df['RenkTercihi'].unique()]
occupation_categories = ['Meslek_' + occupation for occupation in df['Meslek'].unique()]

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)

df[color_categories] = ohe_color
df[occupation_categories] = ohe_occupation

print(df)

----------------------------------------------------------------------------------
"One hot encoding" yapmanın diğer bir yolu da manuel yoldur.
 
 
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

print(df, end='\n\n')

import numpy as np

color_cats = np.unique(df['RenkTercihi'].to_numpy())
occupation_cats = np.unique(df['Meslek'].to_numpy())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['RenkTercihi'] = le.fit_transform(df['RenkTercihi'])
df['Meslek'] = le.fit_transform(df['Meslek'])

print(df, end='\n\n')

color_um = np.eye(len(color_cats))
occupation_um = np.eye(len(occupation_cats))

ohe_color = color_um[df['RenkTercihi'].to_numpy()]
ohe_occupation = occupation_um[df['Meslek'].to_numpy()]

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)
df[color_cats] = ohe_color
df[occupation_cats] = ohe_occupation

print(df, end='\n\n')

----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
# dummy variable encoding

"One hot encoding" işleminin bir versiyonuna da "dummy variable encoding" denilmektedir. 
Şöyle ki: "One hot encoding" işleminden tane kategori için n tane sütun oluşturuluyordu. 
Halbuki "dummy variable encoding" işleminde n tane kategori için n - 1 tane sütun 
oluşturulmaktadır. Çünkü bu yöntemde bir kategori tüm sütunlardaki sayının 0 olması 
ile ifade edilmektedir. Örneğin Kırmızı, Yeşil, Mavi kategorilerinin bulunduğu 
bir sütun şöyle "dummy variable encoding" biçiminde dönüştürülebilir:

Mavi Yeşil
0       0       (Kırmızı)
1       0       (Mavi)
0       1       (Yeşil)

Görüldüğü gibi kategorilerden biri (burada "Kırmızı") tüm elemanı 0 olan satırla 
temsil edilmiştir. Böylece sütun sayısı bir eksiltilmiştir.

---------------------------------------------------------------------------------
"Dummy variable encoding" işlemi için farklı sınıflar ya da fonksiyonlar kullanılmamaktadır. 
Bu işlem "one hot encoding" yapan sınıflar ve fonksiyonlarda özel bir parametreyle 
gerçekleştirilmektedir. Örneğin scikit-learn kütüphanesindeki OneHotEncoder 
sınıfının drop parametresi 'first' olarak geçilirse bu durumda transform işlemi 
"dummy variable encoding" biçiminde yapılmaktadır.

---------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

ohe = OneHotEncoder(sparse=False, drop='first')
transformed_data = ohe.fit_transform(df[['RenkTercihi']]) 

print(df['RenkTercihi'])
print()
print(ohe.categories_)
print()
print(transformed_data)

Görüldüğü gibi burada "Kırmızı" kategorisi [0, 0] biçiminde kodlanmıştır. 

---------------------------------------------------------------------------------
Pandas'ın get_dummies fonksiyonunda drop_first parametresi True geçilirse 
"dummy variable encoding" uygulanmaktadır. Örneğin:

    
transformed_df = pd.get_dummies(df, columns=['RenkTercihi', 'Meslek'], dtype='uint8', drop_first=True)
print(transformed_df)

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# Binary encoding 

Binary encoding yönteminde her kategori "ikilik sistemde bir sayıymış" gibi ifade 
edilmektedir. Örneğin sütunda 256 tane kategori olsun. Bu kategoriler 0'dan 255'e 
kadar numaralandırılabilir. 0 ile 255 arasındaki sayılar 2'lik sistemde 8 bit ile
ifade edilebilir. Örneğin bir kategorinin sayısal değeri (LabelEncoder yapıldığını 
düşünelim) 14 olsun. Biz bu kategoriyi aşağıdaki gibi 8 bit'lik 2'lik sistemde 
bir sayı biçiminde kodlayabiliriz:

0 0 0 0 1 1 1 0 

Tabii kategori sayısı tam 2'nin kuvveti kadar olmak zorunda değildir. Bu durumda 
kategori sayısı N olmak üzere gerkeli olan bit sayısı (yani sütun sayısı) 
ceil(log2(N)) hesabı ile elde edilebilir. 

!! ceil fonksiyonu float sayıyı üste yuvarlar

scikit-learn kütüphanesinin contribute girişimlerinden birinde bu işlemi yapan bir 
BinaryEncoder isminde bir sınıf bulunmaktadır. Bu sınıf category_encoders isimli 
bir paket içerisindedir ve bu paket ayrıca yüklenmelidir. Yükleme şöyle yapılabilir:

pip install category_encoders

---------------------------------------------------------------------------------
BinaryEncoder sınıfının transform fonksiyonu default durumda Pandas DataFrame nesnesi 
vermektedir. Ancak nesne yaratılırken return_df parametresi False geçilirse bu 
durumda transform fonksiyonları NumPy dizisi geri döndürmektedir.

from category_encoders.binary import BinaryEncoder

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

be = BinaryEncoder()
transformed_data = be.fit_transform(df['Meslek'])
print(transformed_data) 

Burada "Meslek" sütunu "binary encoding" biçiminde kodlanmıştır. BinaryEncode kodlamada değerleri 1'den başlatılmaktadır. 
Yukarıdaki işlemden aşağıdaki gibi bir çıktı elde edilmiştir:

   Meslek_0  Meslek_1
0         0         1
1         0         1
2         1         0
3         1         1
4         1         0
5         1         1
6         0         1

---------------------------------------------------------------------------------
import pandas as pd
  
df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

from category_encoders.binary import BinaryEncoder

be = BinaryEncoder()
transformed_data = be.fit_transform(df[['RenkTercihi', 'Meslek']])
print(transformed_data)

df.drop(['RenkTercihi', 'Meslek'], axis=1, inplace=True)
df[transformed_data.columns] = transformed_data    # pd.concat((df, transformed_data), axis=1)
print(df)

---------------------------------------------------------------------------------
Tıpkı get_dummies fonksiyonunda olduğu gibi aslında bir DataFrame bütünsel olarak 
da verilebilir. Yine default durumda tüm yazısal sütunlar "binary encoding" 
dönüştürmesine sokulmaktadır. Ancak biz BinaryEncoding sınıfının __init__ metodunda
cols parametresi ile hangi sütunların dönüştürüleceğini belirleyebiliriz.


import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Desktop/GitHub/YapayZeka/Src/1- DataPreparation/test.csv')

from category_encoders.binary import BinaryEncoder

be = BinaryEncoder(cols=['RenkTercihi', 'Meslek'])
transformed_df = be.fit_transform(df)
print(transformed_df)

---------------------------------------------------------------------------------
"""



#  -------------------  Yapay Sinir Ağları (Artificial Neural Neetworks)  -------------------

"""
---------------------------------------------------------------------------------
Yapay zeka ve makine öğrenmesi alanının en önemli yöntemlerinin başında "yapay 
sinir ağları (artificial neural networks)" ve "derin öğrenme (deep learning)" denilen 
yöntemler gelmektedir. Biz de bu bölümde belli bir derinlikte bu konuları ele alacağız. 

Yapay sinir ağları yapay nöronların birbirlerine bağlanmasıyla oluşturulmaktadır. 
Bir nöronun girdileri vardır ve yalnızca bir tane de çıktısı vardır. Nöronun girdileri 
veri kümesindeki satırları temsil eder. Yani veri kümesindeki satırlar nöronun 
girdileri olarak kullanılmaktadır. Nöronun girdilerini xi temsil edersek her girdi 
"ağırlık (weight) değeri" denilen bir değerler çarpılır ve bu çarpımların toplamları 
elde edilir. Ağırlık değerlerini wi ile gösteribiliriz. Bu durumda xi'lerle wi'ler 
karşılıklı olarak çarpılıp toplanmaktadır.

total = x1w1 + x2w2 + x3w3 + x4w4 + x5w5

İki vektörün karşılıklı elemanlarının çarpımlarının toplamına İngilizce "dot product" 
denilmektedir. (Dot product işlemi np.dot fonksiyonuyla yapılabilmektedir.) Elde 
edilen dot product "bias" denilen bir değerle toplanır. Biz bias dieğerini b ile 
temsil edeceğiz. Örneğin:

total = x1w1 + x2w2 + x3w3 + x4w4 + x5w5 + b

Bu toplam da "aktivasyon fonsiyonu (activation function)" ya da "transfer fonksiyonu 
(transfer function)" denilen bir fonksiyona sokulmaktadır. Böylece o nöronun çıktısı 
elde edilmektedir. Örneğin:

out = activation(x1w1 + x2w2 + x3w3 + x4w4 + x5w5 + b)

Biz bu işlemi vektörel olarak şöyle de gösterebiliriz:

out = activation(XW + b)

---------------------------------------------------------------------------------
Bir nöronun çıktısı başka nöronlar girdi yapılabilir. Böylece tüm ağın nihai çıktıları 
oluşur. Örneğin ağımızın bir katmanında k tane nöron olsun. Tüm girdilerin 
(yani Xi'lerin) bu nöronların hepsine bağlandığını varsayalım. Her nöronun ağırlık 
değerleri ve bias değeri diğerlerinden farklıdır.

out = activation(XW + b)

X = [x1, x2, x3, ..., xn]

W matrisi aşağıdaki görünümdedir:

w11  w21 w31 ... wk1
w12  w22 w31 ... wk2
w13  w23 w33 ... wk3
...  ... ... ... ...
w1n  w2n w3n ... wkn


Buradaki X matrisi ile W matrisi matrisi, matris çarpımına sokulduğunda 1XK boyutunda 
bir matris elde edilecektir. Gösterimimizdeki b matrisi şöyle temsil edilebilir:

b = [b1, b2, b2, ...., bk]

Böylece XW + b işleminden 1XK boyutunda bir matris elde edilecektir. 

---------------------------------------------------------------------------------
Bir sinir ağının amacı bir "kestirimde" bulunmaktır. Yani biz ağa girdi olarak Xi 
değerlerini veririz. Ağdan hedeflediğimiz çıktıyı elde etmeye çalışırız. Ancak 
bunu yapabilmemiz için nöronlardaki w değerlerinin ve b değerlerinin biliniyor 
olması gerekir. Önce biz ağımızı mevcut verilerle eğitip bu w ve b değerlerinin 
uygun biçimde oluşturulmasını sağlarız. Ondan sonra kestirim yaparız. Tabi ağ ne 
kadar iyi eğitilirse ağın yapacağı kestirim de o kadar isabetli olacaktır. 

---------------------------------------------------------------------------------
İstatistikte girdi değerlerinden hareketle çıktı değerinin belirlenmesine (tahmin 
edilmesine) yönelik süreçlere "regresyon (regression)" denilmektedir. Regresyon 
işlemleri çok çeşitli biçimlerde sınıflandırılabilmektedir. Çıktıya göre regresyon 
işlemleri istatistikte tipik olarak iki grupla sınıflandırılmaktadır:

1) (Lojistik Olmayan) Regresyon İşlemleri
2) Lojistik Regresyon İşlemleri

Aslında istatistikte "regresyon" denildiğinde çoğu kez default olarak zaten 
"lojistik olmayan regresyon" işlemleri anlaşılmaktadır. Bu tür regresyonlarda 
girdilerden hareketle kategorik değil sayısal bir çıktı elde edilmektedir.

Girdiler bir resmin pixel'leri olabilir. Çıktı da bu resmin elma mı, armut mu, 
kayısı mı olduğuna yönelik kategorik bir bilgi olabilir. Bu tür regresyonlara 
istatistikte "lojistik regresyonlar" ya da "logit regresyonları" denilmektedir.

Makine öğrenmesinde lojistik regresyon terimi yerine daha çok "sınıflandırma 
(classification)" terimi kullanılmaktadır. Buradaki "lojistik" sözcüğü mantıktaki 
"lojikten" gelmektedir. İngilizce "logistic" terimi "logic" ve "statistics" 
terimlerinin birleştirilmesiyle uydurulmuştur.

Lojistik olsun ya da olmasın aslında regresyon işlemlerinin hepsi girdiyi çıktıya 
dönüştüren bir f foonksiyonunun elde edilmesi sürecidir. Örneğin:

y = f(x1, x2, ..., xn)

İşte makine öğrenmesinde bu f fonksiyonunun elde edilmesinin çeşitli yöntemleri 
vardır. Yapay sinir ağlarında da aslında bu biçimde bir f fonksiyonu bulunmaya 
çalışılmaktadır.

---------------------------------------------------------------------------------
Sınıflandırma sürecindeki çıktının olabileceği değerlere "sınıf (class)" denilmektedir.  
Sınıflandırma (lojistik regresyon) problemlerinde eğer çıktı ancak iki değerden 
biri olabiliyorsa bu tür sınıflandırma problemlerine de "iki sınıflı sınıflandırma 
(binary classification)" problemleri denilmektedir. Örneğin bir film hakkında 
yazılan yorum yazısının "olumlu" ya da "olumsuz" biçiminde iki değerden oluştuğunu 
düşünelim. Bu sınıflandırma işlemi ikili sınıflandırmadır. Benzer biçimde bir 
biyomedikal görüntüdeki kitlenin "iyi huylu (benign)" mu "kötü huylu (malign)" mu 
olduğuna yönelik sınıflandırma da ikili sınıflandırmaya örnektir. 

Eğer sınıflandırmada çıktı sınıflarının sayısı ikiden fazla ise böyle sınıflandırma 
problemlerine "çok sınıflı (multiclass)" sınıflandırma problemleri denilmektedir. 
Örneğin bir resmin hangi meyveye ilişkin olduğunun tespit edilmesi için kullanılan 
sınıflandırma modeli "çok sınıflı" bir modeldir. 

İstatistikte "lojistik regresyon" denildiğinde aslında default olarak "iki sınıflı 
(binary)" lojistik regresyon anlaşılmaktadır. Çok sınıflı lojistik regresyonlara 
istatistikte genellikle İngilizce "multiclass logistic regression" ya da 
"multinomial logistic regression" denilmektedir.

---------------------------------------------------------------------------------
En basit yapay sinir ağı mimarisi tek bir nörondan oluşan mimaridir. Buna 
"perceptron" denilmektedir. Aşağıda bir nöronun bir sınıfla temsil edilmesine ilişkin 
bir örnek 

import numpy as np

class Neuron:
    def __init__(self, w, b):
        self.w= w
        self.b = b
        
    def output(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)
    
    @staticmethod
    def sigmoid(x):
        return np.e ** x / (1 + np.e ** x)

        
w = np.array([1, 2, 3])
b = 1.2

n = Neuron(w, b)

x = np.array([1, 3, 4])
result = n.output(x)
print(result)
---------------------------------------------------------------------------------
"""


#  Yapay Sinir Ağlarında Katmanlar (LAYER)

"""
---------------------------------------------------------------------------------
Bir yapay sinir ağı modelinde "katmanlar (layers)" vardır. Katman aynı düzeydeki 
nöron grubuna denilmektedir. Yapya sinir ağı katmanları tipik olarak üçe ayırmaktadır:

1) Girdi Katmanı (Input Layer)
2) Saklı Katmanlar (Hidden Layers)
3) Çıktı Katmanı (Output Layer)

Girdi katmanı veri kümesindeki satırları temsil eden yani ağa uygulanacak verileri 
belirten katmandır. Aslında girdi katmanı gerçek anlamda nöronlardan oluşmaz. Ancak 
anlatımları kolaylaştırmak için bu katmanın da nöronlardan oluştuğu varsayılmaktadır. 
Başka bir deyişle girdi katmanının tek bir nöron girişi ve tek bir çıktısı vardır.
Yani girdi katmanı bir şey yapmaz, girdiyi değiştirmeden çıktıya verir. Girdi 
katmanındaki nöron sayısı veri kümesindeki sütunların (yani özelliklerin) sayısı 
kadar olmalıdır. Örneğin 5 tane girdiye (özelliğe) sahip olan bir sinir ağının 
girdi katmanı şöyle gösterilebilir:

x1 ---> O --->
x2 ---> O --->
x3 ---> O --->
x4 ---> O --->
x5 ---> O --->

Buradaki O sembolleri girdi katmanındaki nöronları temsil etmektedir. Girdi katmanındaki 
nöronların 1 tane girdisinin 1 tane de çıktısınn olduğuna dikkat ediniz. Buradaki 
nöronlar girdiyi değiştirmediğine göre bunların w değerleri 1, b değerleri 0, 
aktivasyon fonksiyonu da f(x) = x biçiminde olmalıdır.


Hidden Layer -> Girdiler saklı katman denilen katmanlardaki nöronlara bağlanırlar. 
Modelde sıfır tane, bir tane ya da birden fazla saklı katman bulunabilir. Saklı 
katmanların sayısı ve saklı katmanlardaki nöronların sayısı ve bağlantı biçimleri 
problemin niteliğine göre değişebilmektedir. Yani saklı katmanlardaki nöronların 
girdi katmanıyla aynı sayıda olması gerekmez. Her saklı katmandaki nöron sayıları 
da aynı olmak zorunda değildir.


Çıktı katmanı bizim sonucu alacağımız katmandır. Çıktı katmanındaki nöron sayısı 
bizim kestirmeye çalıştığımız olgularla ilgilidir. Örneğin biz bir evin fiyatını 
kestirmeye çalışıyorsak çıktı katmanında tek bir nöron bulunur. Yine örneğin biz 
ikili sınıflandırma problemi üzerinde çalışıyorsak çıktı katmanı yine tek bir nörondan 
oluşabilir. Ancak biz evin fiyatının yanı sıra evin sağlamlığını da kestirmek 
istiyorsak bu durumda çıktı katmanında iki nöron olacaktır. Benzer biçimde çok 
sınıflı sınıflandırma problemlerinde çıktı katmanında sınıf sayısı kadar nöron bulunur.

---------------------------------------------------------------------------------
Bir yapay sinir ağı modelinde katman sayısının artırılması daha iyi bir sonucun 
elde edileceği anlamına gelmez. Benzer biçimde katmanlardaki nöron sayılarının 
artırılması da daha iyi bir sonucun elde edileceği anlamına gelmemektedir. Katmanların 
sayısından ziyade onların işlevleri daha önemli olmaktadır. Ağa gereksiz katman 
eklemek, katmanlardaki nöronları artırmak tam ters bir biçimde ağın başarısının 
düşmesine de yol açabilmektedir. Yani gerekmediği halde ağa saklı katman eklemek, 
katmanlardaki nöron sayısını artırmak bir fayda sağlamamakta tersine kestirim 
başarısını düşürebilmektedir. Ancak görüntü tanıma gibi özel ve zor problemlerde 
saklı katman sayılarının artırılması gerekebilmektedir. 

---------------------------------------------------------------------------------
Pekiyi bir sinir ağı modelinde kaç tane saklı katman olmalıdır? Pratik olarak 
şunları söyleyebiliriz:
    
- Sıfır tane saklı katmana sahip tek bir nörondan oluşan en basit modele "perceptron" 
dendiğini belirtmiştir. Bu perceptron "doğrusal olarak ayrıştırılabilen (linearly 
separable)" sınıflandırma problemlerini ve yalın doğrusal regresyon problemlerini 
çözebilmektedir. 

- Tek saklı katmanlı modeller aslında pek çok sınıflandırma problemini ve (doğrusal 
olmayan) regresyon problemlerini belli bir yeterlilikte çözebilmektedir. Ancak 
tek saklı katman yine de bu tarz bazı problemler için yetersiz kalabilmektedir. 

- İki saklı katman pek çok karmaşık olmayan sınıflandırma problemi için ve regresyon 
problemi için iyi bir modeldir. Bu nedenle karmaşık olmayan problemler için ilk 
akla gelecek model iki saklı katmanlı modeldir. 

- İkiden fazla saklı katmana sahip olan modeller karmaşık ve özel problemleri çözmek 
için kullanılmaktadır. İki saklı katmandan fazla katmana sahip olan modellere genel 
olarak "derin öğrenme ağları (deep learning networks)" denilmektedir. 

Yukarıda da belirttiğimiz gibi "derin öğrenme (deep learning)" farklı bir yöntemi 
belirtmemektedir. Derin öğrenme özel ve karmaşık problemleri çözebilmek için ikiden 
fazla saklı katman içeren sinir ağı modellerini belirtmek için kullanılan bir 
terimdir.    

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Bir veri kümesini CSV dosyasından okuduktan sonra onu Keras'ın kullanımına hazırlamak 
için bazı işlemlerin yapılması gerekir. Yapılması gereken ilk işlem veri kümesinin 
dataset_x ve dataset_y biçiminde iki parçaya ayrılmasıdır. Çünkü ağın eğitilmesi,
sırasında girdilerle çıktıların ayrıştırılması gerekmektedir. Burada dataset_x 
girdileri dataset_y ise kestirilecek çıktıları belirtmektedir. 

Eğitim bittikten sonra genellikle ağın kestirimine hangi ölçüde güvenileceğini 
belirleyebilmek için bir test işlemi yapılır. Ağın kestirim başarısı "test veri kümesi" 
denilen bir veri kümesi ile yapılmaktadır. Test veri kümesinin eğitimde kullanılmayan 
bir veri kümesi biçiminde olması gerekir.

Eğitim ve test veri kümesini manuel olarak ayırabiliriz. Ancak ayırma işleminden 
önce veri kümesini satırsal bakımdan karıştırmak uygun olur. Çünkü bazı veri kümeleri 
CSV dosyasına karışık bir biçimde değil yanlı bir biçimde kodlanmış olabilmektedir. 
Örneğin bazı veri kümeleri bazı alanlara göre sıraya dizilmiş bir biçimde bulunabilmektedir. 
Biz onun baştaki belli kısmını eğitim, sondaki belli kısmını test veri kümesi 
olarak kullanırsak eğitim ve test veri kümeleri yanlı hale gelebilmektedir.
-----> np.random.shuffle(dataset)

---------------------------------------------------------------------------------
diabetes.csv bazı sütunlar eksik veri içermektedir. Bu eksik verile NaN biçiminde 
değil 0 biçiminde kodlanmıştır. Biz bu eksik verileri ortalama değerle doldurabiliriz. 
Eksik veri içeren sütunlar şunlardır:

Glucose
BloodPressure
SkinThickness
Insulin
BMI

---------------------------------------------------------------------------------
TRAINING_RATIO = 0.80

import pandas as pd
df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\2- KerasIntroduction\diabetes.csv')


from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='mean', missing_values=0 )

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
print((df==0).sum())
print()

dataset = df.to_numpy()

import numpy as np

np.random.shuffle(dataset)

dataset_x = dataset[:, :-1] 
dataset_y = dataset[:, -1] 

training_len = int(np.round(len(dataset) * TRAINING_RATIO))

training_dataset_x = dataset_x[ :training_len]
test_dataset_x = dataset_x[training_len : ]

training_dataset_y = dataset_y[ :training_len]
test_dataset_y = dataset_y[training_len : ]

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Veri kümesini eğitim ve test olarak ayırma işlemi için sklearn.model_selection 
modülündeki train_test_split isimli fonksiyon sıkça kullanılmaktadır. Fonksiyon 
NumPy dizilerini ya da Pandas DataFrame ve Series nesnelerini ayırabilmektedir. 

Fonksiyon bizden dataset_x ve dataset_y değerlerini ayrı ayrı ister. test_size 
ya da train_size parametreleri 0 ile 1 arasında test ya da eğitim verilerinin oranını 
belirlemek için kullanılmaktadır. train_test_split fonksiyonu bize 4'lü bir liste 
vermektedir. Listenin elemanları sırasıyla şunlardır: training_dataset_x, 
test_dataset_x, training_dataset_y, test_dataset_y. Örneğin:

    
from sklearn.model_selection import train_test_split
training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

test_size = 0.2  ---> TRAINING_RATIO = 0.8 ---> training_size = 0.8

Burada fonksiyona dataset_x ve dataset_y girdi olarak verilmiştir. Fonksiyon 
bunları bölerek dörtlü bir listeye geri dönmüştür.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Keras'ta bir sinir ağı oluşturmanın çeşitli adımları vardır. Burada sırasıyla bu 
adımlardan ve adımlarla ilgili bazı olgulardan bahsedeceğiz.

1-) Öncelikle bir model nesnesi oluşturulmalıdır. tensorflow.keras modülü içerisinde 
çeşitli model sınıfları bulunmaktadır. En çok kullanılan model sınıfı Sequential 
isimli sınıftır. Tüm model sınıfları Model isimli sınıftan türetilmiştir. Sequential 
modelde ağa her eklenen katman sona eklenir. Böylece ağ katmanların sırasıyla 
eklenmesiyle oluşturulur. Sequential nesnesi yaratılırken name parametresiyle modele 
bir isim de verilebilir. Örneğin:

from tensorflow.keras import Sequential

model = Sequential(name='Sample')

Aslında Sequential nesnesi yaratılırken katmanlar belirlendiyse layers parametresiyle 
bu katmanlar da verilebilmektedir. Ancak sınıfın tipik kullanımında katmanlar daha 
sonra izleyen maddelerde ele alınacağı gibi sırasıyla eklenmektedir.

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
2-) Model nesnesinin yaratılmasından sonra katman nesnelerinin oluşturulup model nesnesine 
eklenmesi gerekir. Keras'ta farklı gereksinimler için farklı katman sınıfları 
bulundurulmuştur. En çok kullanılan katman sınıfı tensorflow.keras.layers modülündeki 
Dense sınıfıdır. Dense bir katman modele eklendiğinde önceki katmandaki tüm nöronların 
çıktıları eklenen katmandaki nöronların hepsine girdi yapılmaktadır. 

Bu durumda örneğin önceki katmanda k tane nöron varsa biz de modele n tane nörondan 
oluşan bir Dense katman ekliyorsak bu durumda modele k * n + n tane yeni parametre 
(yani tahmin edilmesi gereken parametre) eklemiş oluruz. Burada k * n tane ayarlanması 
gereken w (ağırlık) değerleri ve n tane de ayarlanması gereken bias değerleri söz 
konusudur. 

!!! Bir nörondaki w (ağırlık) değerlerinin o nörona giren nöron sayısı kadar 
olduğuna ve bias değerlerinin her nöron için bir tane olduğuna dikkat ediniz. !!!

Dense sınıfının __init__ metodunun ilk parametresi eklenecek katmandaki nöron sayısını 
belirtir. İkinci parametre olan activation parametresi o katmandaki tüm nöronların 
aktivasyon fonksiyonlarının ne olacağını belirtmektedir. Bu parametreye aktivasyon 
fonksiyonları birer yazı biçiminde isimsel olarak girilebilir. Ya da tensorflow.keras.activations 
modülündeki fonksiyonlar olarak girilebilir. Örneğin:

from tensorflow.keras.layers import Dense
layer = Dense(100, activation='relu')

ya da örneğin:
    
from tensorflow.keras.activations import relu
layer = Dense(100, activation=relu)


Dense fonksiyonun use_bias parametresi default durumda True biçimdedir. Bu parametre 
katmandaki nöronlarda "bias" değerinin kullanılıp kullanılmayacağını belirtmektedir. 

Metodun kernel_initializer parametresi katmandaki nöronlarda kullanılan w parametrelerinin 
ilk değerlerinin rastgele biçimde hangi algoritmayla oluşturulacağını belirtmektedir. 
Bu parametrenin default değeri "glorot_uniform" biçimindedir. 

Metodun bias_initializer parametresi ise katmandaki nöronların "bias" değerlerinin 
başlangıçta nasıl alınacağını belirtmektedir. Bu parametrenin default değeri de 
"zero" biçimdedir. Yani bias değerleri başlangıçta 0 durumundadır.

---------------------------------------------------------------------------------
Keras'ta Sequential modelde girdi katmanı programcı tarafından yaratılmaz. İlk 
saklı katman yaratılırken girdi katmanındaki nöron sayısı input_dim parametresiyle 
ya da input_shape parametresiyle belirtilmektedir. input_dim tek boyutlu girdiler için 
input_shape ise çok boyutlu girdiler için kullanılmaktadır. Örneğin:

layer = Dense(100, activation='relu', input_dim=8) # tek boyutlu 8 tane nörondan oluşuyor demek

input_shape= (10,10) # girdi katmanı 2 boyutlu 10'a 10'luk matris demek

Tabii input_dim ya da input_shape parametrelerini yalnızca ilk saklı katmanda kullanabiliriz. 
Genel olarak ağın girdi katmanında dataset_x'teki sütun sayısı kadar nöron olacağına 
göre ilk katmandaki input_dim parametresini aşağıdaki gibi de girebiliriz:

layer = Dense(100, activation='relu', input_dim= training_dataset_x.shape[1])

Aslında Keras'ta girdi katmanı için tensorflow.keras.layers modülünde Input isminde 
bir katman da kullanılmaktadır. Tenseoflow'un yeni versiyonlarında girdi katmanının 
Input katmanı ile oluşturulması istenmektedir. Aksi takdirde bu yeni versiyonlar uyarı 
vermektedir. Girdi katmanını Input isimli katman sınıfıyla oluştururken bu Input 
sınıfının __init__ metodunun birinci parametresi bir demet biçiminde (yani shape olarak) 
girilmelidir. Örneğin:

input = Input((8, ))

Burada 8 nöronluk bir girdi katmanı oluşturulmuştur. Yukarıda da belirttiğimiz gibi 
eskiden ilk saklı katmanda girdi katmanı belirtiliyordu. Ancak Tensorflow kütüphanesinin 
yeni verisyonlarında ilk saklı katmanda girdi katmanının belirtilmesi artık uyarıya 
(warning) yol açmaktadır.

Her katmana istersek name parametresi ile bir isim de verebiliriz. Bu isimler model 
özeti alınırken ya da katmanlara erişilirken kullanılabilmektedir. Örneğin:

layer = Dense(100, activation='relu',  name='Hidden-1')

Oluşturulan katman nesnesinin model nesnesine eklenmesi için Sequential sınıfının 
add metodu kullanılmaktadır. Örneğin:

input = Input((8, ))
model.add(input)
layer = Dense(100, activation='relu',  name='Hidden-1')
model.add(layer)

Programcılar genellikle katman nesnesinin yaratılması ve eklenmesini tek satırda 
aşağıdaki gibi yaparlar:

model.add(Dense(100, activation='relu', input_dim=9, name='Hidden-1'))

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
3) Modele katmanlar eklendikten sonra bir özet bilgi yazdırılabilir. Bu işlem 
Sequential sınıfının summary isimli metoduyla yapılmaktadır. 

Yukarıda da belirttiğimiz gibi bir katmandaki "eğitilebilir (trainable)" parametrelerin 
sayısı aşağıda olan örnekteki gibi hesaplanmaktadır. Aşağıdaki modeli inceleyiniz:
  
    
model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

Bu modelde bir girdi katmanı, iki saklı katman (biz bunlara ara katman da diyeceğiz) 
bir de çıktı katmanı vardır. summary metodundan elde edilen çıktı şöyledir.

    Model: "Diabetes"
    ┌─────────────────────────────────┬────────────────────────┬───────────────┐
    │ Layer (type)                    │ Output Shape           │       Param # │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ Hidden-1 (Dense)                │ (None, 16)             │           144 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ Hidden-2 (Dense)                │ (None, 16)             │           272 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ Output (Dense)                  │ (None, 1)              │            17 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    Total params: 433 (1.69 KB)
    Trainable params: 433 (1.69 KB)
    Non-trainable params: 0 (0.00 B)
    

Burada ağımızdaki girdi katmanında 8 nöron olduğuna göre ve ilk saklı katmanda da 
16 nöron olduğuna göre ilk saklı katmana (8 * 16) nöron girmektedir. Öte yandan 
her nöronun bir tane bias değeri de olduğuna göre ilk katmandaki tahmin ayarlanması 
gereken parametrelerin (trainable parameters) sayısı (8 * 16 + 16 = 144) tanedir. 
İkinci saklı katmana 16 nöron dense biçimde bağlanmıştır. O halde ikinci saklı 
katmandaki ayarlanması gereken parametreler toplamda (16 * 16 + 16 = 272) tanedir. 
Modelimizin çıktı katmanında 1 nöron vardır. Önceki katmanın 16 çıkışı olduğuna 
göre bu çıktı katmanında (16 * 1 + 1 = 17) tane ayarlanması gereken parametre vardır. 

Ağın saklı katmanlarında en çok kullanılan aktivasyon fonksiyonu "relu" isimli fonksiyondur. 
İkili sınıflandırma problemlerinde çıktı katmanı tek nörondan oluşur ve bu katmandaki 
aktivasyon fonksiyonu "sigmoid" fonksiyonu olur. Sigmoid fonksiyonu 0 ile 1 arasında 
bir değer vermektedir. Biz aktivasyon fonksiyonlarını izleyen paragraflarda ele alacağız.

Aşağıdaki örnekte "dibates" veri kümesi üzerinde ikili sınıflandırma problemi için 
bir sinir ağı oluşturulmuştur. 

---------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
4) Model oluşturulduktan sonra modelin derlenmesi (compile edilmesi) gerekir. Buradaki 
"derleme" makine diline dönüştürme anlamında bir terim değildir. Eğitim için bazı 
belirlemelerin yapılması anlamına gelmektedir. Bu işlem Sequential sınıfının 
compile isimli metoduyla yapılmaktadır. Modelin compile metoduyla derlenmesi sırasında 
en önemli iki parametre "loss fonksiyonu" ve "optimizasyon algoritması"dır.

Eğitim sırasında ağın ürettiği değerlerin gerçek değerlere yaklaştırılması için w 
ve bias değerlerinin nasıl güncelleneceğine ilişkin algoritmalara "optimizasyon 
algoritmaları" denilmektedir. Matematiksel optimizasyon işlemlerinde belli bir 
fonksiyonun minimize edilmesi istenir. İşte minimize edilecek bu fonksiyona da 
"loss fonksiyonu" denilmektedir. Başka bir deyişle optimizasyon algoritması loss 
fonksiyonun değerini minimize edecek biçimde işlem yapan algoritmadır. Yani optimizasyon 
algoritması loss fonksiyonunu minimize etmek için yapılan işlemleri temsil etmektedir. 
    
Loss fonksiyonları ağın ürettiği değerlerle gerçek değerler arasındaki farklılığı 
temsil eden fonksiyonlardır. Loss fonksiyonları genel olarak iki girdi alıp bir 
çıktı vermektedir. Loss fonksiyonunun girdileri gerçek değerler ile ağın ürettiği 
değerlerdir. Çıktı değeri ise aradaki farklığı belirten bir değerdir. Eğitim sırasında 
git gide loss fonksiyonun değerinin düşmesini bekleriz. Tabi loss değerinin düşmesi 
aslında ağın gerçek değerlere daha yakın değerler üretmesi anlamına gelmektedir.

Loss fonksiyonları çıktının biçimine yani problemin türüne bağlı olarak seçilmektedir. 
Örneğin ikili sınıflandırma problemleri için "binary cross-entropy", çoklu sınıflandırma 
problemleri için "categorical cross-entropy", lojistik olmayan regresyon problemleri 
için "mean squared error" isimli loss fonksiyonları tercih edilmektedir. 


ikili sınıflandırma problemleri --->   binary cross-entropy

çoklu sınıflandırma problemleri --->   categorical cross-entropy

lojistik olmayan regresyon problemleri --->  mean squared error

---------------------------------------------------------------------------------
Optimizasyon algoritmaları aslında genel yapı olarak birbirlerine benzemektedir. 
Pek çok problemde bu algoritmaların çoğu benzer performans göstermektedir. En çok 
kullanılan optimizasyon algoritmaları "rmsprop", "adam" ve "sgd" algoritmalarıdır. 
Bu algoritmalar "gradient descent" denilen genel optimizasyon yöntemini farklı 
biçimlerde uygulamaktadır.

compile metodunda optimizasyon algoritması bir yazı olarak ya da tensorflow.keras.optimizers 
modülündeki sınıflar türünden bir sınıf nesnesi olarak girilebilmektedir. Örneğin:

model.compile(optimizer='rmsprop', ...)

Örneğin:

from tensorflow.keras.optimizers import RMSprop

rmsprop = RMSprop()

model.compile(optimizer=rmsprop, ...)

Tabii optimizer parametresinin bir sınıf nesnesi olarak girilmesi daha detaylı 
belirlemelerin yapılmasına olanak sağlamaktadır. Optimizasyon işlemlerinde bazı 
parametrik değerler vardır. Bunlara makine öğrenmesinde "üst düzey parametreler 
(hyper parameters)" denilmektedir. İşte optimizasyon algoritması bir sınıf nesnesi 
biçiminde verilirse bu sınıfın __init__ metodunda biz bu üst düzey parametreleri 
istediğimiz gibi belirleyebiliriz. Eğer optimizasyon algoritması yazısal biçimde 
verilirse bu üst düzey parametreler default değerlerle kullanılmaktadır. optimzer 
parametresinin default değeri "rmsprop" biçimindedir. Yani biz bu parametre için 
değer girmezsek default optimazyon algoritması "rmsprop" olarak alınacaktır. 

---------------------------------------------------------------------------------
loss fonksiyonu compile metoduna yine isimsel olarak ya da tensorflow.keras.losses 
modülündeki sınıflar türünden sınıf nesleri biçiminde ya da doğrudan fonksiyon 
olarak girilebilmektedir. loss fonksiyonları kısa ya da uzun isim olarak yazısal 
biçimde kullanılabilmektedir. Tipik loss fonksiyon isimleri şunlardır:

'mean_squared_error' ya da 'mse'
'mean_absolute_error' ya da 'mae'
'mean_absolute_percentage_error' ya da 'mape'
'mean_squared_logarithmic_error' ya da 'msle'
'categorical_crossentropy'
'binary_crossentropy'

Bu durumda compile metodu örnek bir biçimde şöyle çağrılabilir:

model.compile(optimizer='rmsprop', loss='binary_crossentropy')

---------------------------------------------------------------------------------
compile metodunun üçüncü önemli parametresi "metrics" isimli parametredir. metrics 
parametresi bir liste ya da demet olarak girilir. metrics parametresi daha sonra 
açıklanacak olan "sınama (validation)" işlemi için kullanılacak fonksiyonları 
belirtmektedir. Sınamada her zaman zaten bizim loss fonksiyonu olarak belirttiğimiz 
fonksiyon kullanılmaktadır. metrics parametresinde ilave fonksiyonlar da girilebilmektedir. 

Örneğin ikili sınıflandırma problemleri için tipik olarak "binary_accuracy" denilen 
metrik fonksiyon, çoklu sınıflandırma için "categorical_accuracy" denilen metrik 
fonksiyon ve lojistik olmayan regresyon problemleri için de "mean_absolute_error" 
isimli metrik fonksiyon sıklıkla kullanılmaktadır. 

ikili sınıflandırma problemleri --->   binary_accuracy

çoklu sınıflandırma problemleri --->   categorical_accuracy

lojistik olmayan regresyon problemleri --->   mean_absolute_error


Örneğin ikili sınıflandırma problemi için biz eğitim sırasında "loss" değerinin 
yanı sıra "binary_accuracy" değerini de elde etmek isteyelim. Bu durumda compile 
metodunu şöyle çağırmalıyız:
    
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])    

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
5) Model derlenip çeşitli belirlemeler yapıldıktan sonra artık gerçekten eğitim 
aşamasına geçilir. Eğitim süreci Sequential sınıfının fit metoduyla yapılmaktadır. 
fit metodunun en önemli parametresi ilk iki parametre olan x ve y veri kümeleridir. 
Biz burada training_dataset_x ve training_dataset_y verilerini fit metodunun ilk 
iki parametresine geçirmeliyiz.

fit metodunun önemli bir parametresi batch_size isimli parametredir. Eğitim işlemi 
aslında satır satır değil batch batch yapılmaktadır. batch bir grup satıra denilmektedir. 
Yani ağa bir grup satır girdi olarak verilir. Ağdan bir grup çıktı elde edilir. Bu 
bir grup çıktı ile bu çıktıların gerçek değerleri loss fonksiyonuna sokulur ve 
optimizasyon algoritması çalıştırılarak w ve bias değerleri güncellenir. Yani 
optimizasyon algoritması her batch işlemden sonra devreye sokulmaktadır. Batch 
büyüklüğü fit metodunda batch_size parametresiyle belirtilmektedir. Bu değer girilmezse 
batch_size 32 olarak alınmaktadır. 32 değeri pek çok uygulama için uygun bir değerdir. 
Optimizasyon işleminin satır satır yapılması yerine batch batch yapılmasının iki 
önemli nedeni vardır: Birincisi işlem miktarının azaltılması, dolayısıyla eğitim 
süresinin kısaltılmasıdır. İkincisi ise "overfitting" denilen olumsuz durum için 
bir önlem oluşturmasıdır.

---------------------------------------------------------------------------------
fit metodunun diğer önemli parametresi de "epochs" isimli parametredir. Eğitim 
veri kümesinin eğitim sırasında yeniden eğitimde kullanılmasına "epoch" işlemi 
denilmektedir.

Örneğin elimizde 1000 satırlık bir eğitim veri kümesi olsun. batch_size parametresinin 
de 20 olduğunu varsayalım. Bu durumda bu eğitim veri kümesi 1000 / 20 = 50 batch 
işleminde bitecektir. Yani model parametreleri 50 kere ayarlanacaktır. Pek çok 
durumda eğitim veri kümesinin bir kez işleme sokulması model parametrelerinin
iyi bir biçimde konumlandırılması için yetersiz kalmaktadır. İşte eğitim veri kümesinin 
birden fazla kez yani fit metodundaki epochs sayısı kadar yeniden eğitimde kullanılması 
yoluna gidilmektedir. Pekiyi epochs değeri ne olmalıdır?

Aslında bunu uygulamacı belirler. Az sayıda epoch model parametrelerini yeterince 
iyi konumlandıramayabilir. Çok fazla sayıda epoch "overfitting" denilen olumsuz 
duruma zemin hazırlayabilir. Ayrıca çok fazla epoch eğitim zamanını da uzatmaktadır. 
Uygulamacı epoch'lar sırasında modelin davranışına bakabilir ve uygun epoch sayısında 
işlemi kesebilir. Eğitim sırasında Keras bizim belirlediğimiz fonksiyonları çağırabilmektedir. 
Buna Keras'ın "callback" mekanizması denilmektedir. Uygulamacı bu yolla model belli 
bir duruma geldiğinde eğitim işlemini kesebilir. Ya da uygulamacı eğer eğitim çok 
uzamayacaksa yüksek bir epoch ile eğitimini yapabilir. İşlemler bitince epoch'lardaki 
performansa bakabilir. Olması gereken epoch değerini kestirebilir. Sonra modeli 
yeniden bu sayıda epoch ile eğitir. 

fit metodunun shuffle parametresi her epoch'tan sonra eğitim veri kümesinin karıştırılıp 
karıştırılmayacağını belirtmektedir. Bu parametre default olarak True biçimdedir. 
Yani eğitim sırasında her epoch'ta eğitim veri kümesi karıştırılmaktadır. 

---------------------------------------------------------------------------------
Modelin fit metodu ile eğitilmesi sırasında "sınama (validation)" denilen önemli 
bir kavram daha devreye girmektedir. Sınama işlemi test işlemine benzemektedir. 
Ancak test işlemi tüm model eğitildikten sonra yapılırken sınama işlemi her epoch'tan 
sonra modelin eğitim sürecinde yapılmaktadır. Başka bir deyişle sınama işlemi model 
eğitilirken yapılan test işlemidir.

Epoch'lar sırasında modelin performansı hakkında bilgi edinebilmek için sınama 
işlemi yapılmaktadır. Sınamanın yapılması için fit metodunun validation_split parametresinin 
0 ile 1 arasında oransal bir değer olarak girilmesi gerekir. Bu oransal değer eğitim 
veri kümesinin yüzde kaçının sınama için kullanılacağını belirtmektedir. Örneğin 
validation_split=0.2 eğitim veri kümesinin %20'sinin sınama için kullanılacağını 
belirtmektedir.

fit metodu işin başında eğitim veri kümesini eğitimde kullanılacak kısım ile sınamada 
kullanılacak kısım biçiminde ikiye ayırmaktadır. Sonra her epoch'ta yalnızca eğitimde 
kullanılacak kümeyi karıştırmaktadır. Sınama işlemi aynı kümeyle her epoch sonrasında 
karıştırılmadan yapılmaktadır. fit metodunda ayrıca birde validation_data isimli 
bir parametre vardır. Bu parametre sınama verilerini girmek için kullanılmaktadır. 
Bazen programcı sınama verilerinin eğitim veri kümesinden çekilip alınmasını istemez. 
Onu ayrıca fit metoduna vermek isteyebilir. Tabii validation_data parametresi girildiyse 
artık validation_split parametresinin bir anlamı yoktur. Bu parametre girilse bile 
artık fonksiyon tarafından dikkate alınmaz. Örneğin:

model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)


validation_split parametresinin default değerinin 0 olduğuna dikkat ediniz. 
validation_split değerinin 0 olması epoch'lar sonrasında sınama işleminin 
yapılmayacağı anlamına gelmektedir. 

---------------------------------------------------------------------------------
Pekiyi modelin fit metodunda her epoch'tan sonra sınama işleminde hangi ölçümler 
ekrana yazdırılacaktır? İşte compile metodunda belirtilen ve ismine metrik fonksiyonlar 
denilen fonksiyonlar her epoch işlemi sonucunda ekrana yazdırılmaktadır. Her epoch 
sonucunda fit metodu şu değerleri yazdırmaktadır:

- Epoch sonrasında elde edilen loss değerlerinin ortalaması

- Epoch sonrasında eğitim veri kümesinin kendisi için elde edilen metrik değerlerin 
ortalaması

- Epoch sonrasında sınama için kullanılan sınama verilerinden elde edilen loss 
değerlerinin ortalaması

- Epoch sonrasında sınama için kullanılan sınama verilerinden elde edilen metrik 
değerlerin ortalaması


Bir epoch işleminin batch batch yapıldığını anımsayınız. Bu durumda epoch sonrasında 
fit tarafından ekrana yazdırılan değerler bu batch işlemlerden elde edilen ortalama 
değerlerdir. Yani örneğin her batch işleminden bir loss değeri elde edilir. Sonra 
bu loss değerlerinin ortalaması hesap edilerek yazdırılır. 

Eğitim veri kümesindeki değerler ile sınama veri kümesinden elde edilen değerler 
birbirine karışmasın diye fit metodu sınama verilerinden elde edilen değerlerin 
başına "val_" öneki getirmektedir. 

---------------------------------------------------------------------------------
Örneğin biz ikili sınıflandırma problemi üzerinde çalışıyor olalım ve metrik fonksion 
olarak "binary_accuracy" kullanmış olalım. fit metodu her epoch sonrasında şu 
değerleri ekrana yazdıracaktır:


loss (eğitim veri kümesinden elde edilen ortalama loss değeri)
binary_accuracy (eğitim veri kümesinden elde edilen ortalama metrik değer)
val_loss (sınama veri kümesinden elde edilen ortalama loss değeri)
val_binary_accuracy (sınama veri kümesinden elde edilen ortalama metrik değer)

Tabii compile metodunda birden fazla metirk değer de belirtilmiş olabilir. Bu durumda 
fit tüm bu metrik değerlerin ortalamasını ekrana yazdıracaktır. fit tarafından 
ekrana yazdırılan örnek bir çıktı şöyle olabilir:

....
Epoch 91/100
16/16 [==============================] - 0s 3ms/step - loss: 0.5536 - binary_accuracy: 0.7230 - val_loss: 0.5520 - 
val_binary_accuracy: 0.7480
Epoch 92/100
16/16 [==============================] - 0s 3ms/step - loss: 0.5392 - binary_accuracy: 0.7251 - val_loss: 0.5588 - 
val_binary_accuracy: 0.7805
Epoch 93/100
16/16 [==============================] - 0s 3ms/step - loss: 0.5539 - binary_accuracy: 0.7088 - val_loss: 0.5666 - 
val_binary_accuracy: 0.8049
...

Burada loss değeri epoch sonrasında eğitim verilerinden elde edilen ortalama loss 
değerini, val_loss değeri epoch sonrasında sınama verilerinden elde edilen ortalama 
loss değerini, binary_accuracy epoch sonrasında eğitim verilerinden elde edilen
ortalama isabet yüzdesini ve val_binary_accuracy ise epoch sonrasında sınama 
verilerinden elde edilen ortalama isabet yüzdesini belirtmektedir.


Eğitim sırasında eğitim veri kümesindeki başarının sınama veri kümesinde görülmemesi 
eğitimin kötü bir yöne gittiğine işaret etmektedir. 

Örneğin ikili sınıflandırma probleminde epoch sonucunda eğitim veri kümesindeki 
binary_accuracy değerinin %99 olduğunu ancak val_binary_accuracy değerinin %65 
olduğunu düşünelim. Bunun anlamı ne olabilir? Bu durum aslında epoch'lar sırasında 
modelin bir şeyler öğrendiği ama bizim istediğimiz şeyleri öğrenemediği anlamına 
gelmektedir. Çünkü eğitim veri kümesini epoch'larla sürekli bir biçimde gözden 
geçiren model artık onu ezberlemiştir. Ancak o başarıyı eğitimden bağımsız bir veri 
kümesinde gösterememektedir. İşte bu olguya "overfitting" denilmektedir. Yanlış 
bir şeyin öğrenilmesi bir şeyin öğrenilememesi kadar kötü bir durumdur. Overfitting 
oluşumunun çeşitli nedenleri vardır. Ancak overfitting epoch'lar dolayısıyla oluşuyorsa 
epoch'ları uygun bir noktada kesmek gerekir.      

---------------------------------------------------------------------------------  

---------------------------------------------------------------------------------   
6) fit işleminden sonra artık model eğitilmiştir. Onun test veri kümesiyle test 
edilmesi gerekir. Bu işlem Sequential sınıfının evaluate isimli metodu ile yapılmaktadır.
evaluate metodunun ilk iki parametresi test_dataset_x ve test_dataset_y değerlerini 
almaktadır. Diğer bir parametresi yine batch_size parametresidir. Buradaki bacth_size 
eğitim işlemi yapılırken fit metodunda kullanılan batch_size ile benzer anlamdadır, 
ancak işlevleri farklıdır. Model test edilirken test işlemi de birer birer değil batch 
batch yapılabilir. Ancak bu batch'ler arasında herhangi bir şey yapılmamaktadır. 
(Eğitim sırasındaki batch işlemleri sonrasında ağ parametrelerinin ayarlandığını 
anımsayınız. Test işlemi sırasında böyle bir ayarlama yapılmamaktadır.) Buradaki 
batch değeri işlemlerin hızlı yapılması üzerinde etkili olmaktadır. Yine batch_size 
parametresi girilmezse default 32 alınmaktadır. 

evaluate metodu bir liste geri döndürmektedir. Listenin ilk elemanı test veri kümesinden 
elde edilen loss fonksiyonunun değeridir. Diğer elemanları da sırasıyla metrik 
olarak verilen fonksiyonların değerleridir. Örneğin:

eval_result = model.evaluate(test_dataset_x, test_dataset_y)

Aslında eval_result listesinin elemanlarının hangi anlamlara geldiğini daha iyi 
ifade edebilmek için Sequential sınıfında metrics_names isimli bir örnek özniteliği 
(instance attribute) bulundurulmuştur. Bu metrics_names listesindeki isimler bire 
bir evalute metodunun geri döndürdüğü liste elemanları ile örtüşmektedir. Bu durumda 
evaluate metodunun geri döndürdüğü listeyi aşağıdaki gibi de yazdırabiliriz:

    
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

Aynı şeyi built-in zip fonksiyonuyla da şöyle yapabilirdik:

for name, value in zip(model.metrics_names, eval_result):
    print(f'{name}: {value}')
    
--------------------------------------------------------------------------------- 

--------------------------------------------------------------------------------- 
7) Artık model test de edilmiştir. Şimdi sıra "kestirim (prediction)" yapmaya gelmiştir. 
Kestirim işlemi için Sequential sınıfının predict metodu kullanılır. Biz bu metoda 
girdi katmanına uygulanacak sütun verilerini veriririz. predict metodu da bize 
çıktı katmanındaki nöronların değerlerini verir. predict metoduna biz her zaman 
İKİ BOYUTLU bir numpy dizisi vermeliyiz. Çünkü predict metodu tek hamlede birden 
çok satır için kestirim yapabilmektedir. Biz predict metoduna bir satır verecek 
olsak bile onu İKİ BOYUTLU  bir matris biçiminde vermeliyiz.

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

predict_result = model.predict(predict_data)
print(predict_result)

predict metodu bize tahmin edilen değerleri iki boyutlu bir NumPy dizisi biçiminde 
vermektedir. Bunun nedeni aslında ağın birden fazla çıktısının olabilmesidir. Örneğin 
ağın bir çıktısı varsa bu durumda predict metodu bize "n tane satırdan 1 tane 
sütundan" oluşan bir matris, ağın iki çıktısı varsa "n tane satırdan 2 iki tane 
sütundan oluşan bir matris verecektir. O halde örneğin çıktı olarak tek nöronun 
bulunduğu bir ağda ("diabetes" örneğindeki gibi) biz kestirim değerlerini şöyle 
yazdırabiliriz:

for i in range(len(predict_result)):
    print(predict_result[i, 0])

Ya da şöyle yazdırabiliriz:

for result in predict_result[:, 0]:
    print(result)

Tabii iki boyutlu diziyi Numpy'ın flatten metoduyla ya da ravel metoduyla tek 
boyutlu hale getirerek de yazırma işlemini yapabilirdik:

for val in predict_result.flatten():
    print(val)

--------------------------------------------------------------------------------- 
predict metodu bize ağın çıktı değerini vermektedir. Yukarıdaki "diabetes.csv" 
örneğimizde ağın çıktı katmanındaki aktivasyon fonksiyonunun "sigmoid" olduğunu 
anımsayınız. Sigmoid fonksiyonu 0 ile 1 arasında bir değer vermektedir. O halde 
biz ağın çıktısındaki değer 0.5'ten büyükse ilgili kişinin şeker hastası olduğu 
(çünkü 1'e daha yakındır), 0.5'ten küçükse o kişinin şeker hastası olmadığı 
(çünkü 0'a daha yakındır) sonucunu çıkartabiliriz. O halde sigmoid fonksiyonun 
çıktısının bir olasılık belirttiğini söyleyebiliriz. Bu durumda kişinin şeker 
hastası olup olmadığı ağın çıktı değerinin 0.5'ten büyük olup olmamasıyla 
kesitirilebilir:
    
for result in predict_result[:, 0]:
    print('Şeker hastası' if result > 0.5 else 'Şeker Hastası Değil')   

--------------------------------------------------------------------------------- 
"""

"""
--------------------------------------------------------------------------------- 
# Deep Learning (Derin Öğrenme)

Şimdi yukarıdaki adımların bazı detayları üzerinde duralım. Daha önceden de belirttiğimiz 
gibi pek çok problem iki saklı katmanla ve Dense bir bağlantı ile tatminkar biçimde 
çözülebilmektedir. Dolayısıyla bizim ilk aklımıza gelen model iki saklı katmanlı 
klasik modeldir. Ancak özel problemler (şekil tanıma, yazıdan anlam çıkartma, 
görüntü hakkında çıkarım yapma gibi) iki saklı katmanla tatminkar biçimde çözülememektedir. 
Bu durumda ikiden fazla saklı katman kullanılır. Bu modellere "derin öğrenme 
(deep learning)" modelleri de denilmektedir. 

Girdi katmanındaki nöron sayısı zaten problemdeki sütun sayısı (feature sayısı kadar) 
olmalıdır. Tabii kategorik sütunlar "one hot encoding" işlemine sokulmalıdır. 
Çıktı katmanındaki nöron sayıları ise yine probleme bağlı olmaktadır. İkili 
sınıflandırma (binary classification) problemlerinde çıktı katmanı tek nörondan, 
çoklu sınıflandırma problemlerinde (multiclass classification) çıktı katmanı sınıf 
sayısı kadar nörondan oluşur. Lojistik olmayan regresyon problemlerinde ise çıktı
katmanındaki nöron sayıları genellikle bir tane olmaktadır.

--------------------------------------------------------------------------------- 
Saklı katmanlardaki nöron sayıları için çok pratik şeyler söylemek zordur. Çünkü 
saklı katmanlardaki nöron sayıları bazı faktörlere de bağlı olabilmektedir. Örneğin 
eğitimde kullanılacak veri miktarı, problemin karmaşıklığı, hyper parametrelerin 
durumları saklı katmanlardaki nöron sayıları üzerinde etkili olabilmektedir. Saklı 
katmanlardaki nöron sayıları için şunlar söylenebilir:
    
- Problem karmaşıklaştıkça saklı katmanlardaki nöron sayılarını artırmak uygun 
olabilmektedir. 

- Saklı katmanlarda çok az nöron bulundurmak "underfitting" yani yetersiz öğrenmeye 
yol açabilmektedir. 

- Saklı katmanlarda gereksiz biçimde fazla sayıda nöron bulundurmak eğitim süresini 
uzatabileceği gibi hem de "overfitting" durumuna yol açabilir. Aynı zamanda modelin 
diskte saklanması için gereken disk alanını da artırabilmektedir.

- Eğitim veri kümesi azsa saklı katmanlardaki nöron sayıları düşürülebilir. 

- Pek çok problemde saklı katmanlardaki nöron sayıları çok fazla ya da çok az 
olmadıktan sonra önemli olmayabilmektedir. 

- Saklı katmanlardaki nöron sayısı girdi katmanındaki nöron sayısından az olmamlıdır. 

--------------------------------------------------------------------------------- 
Çeşitli kaynaklar saklı katmanlardaki nöronların sayıları için ise üstünkörü şu pratik 
tavsiyelerde bulunmaktadır:

- Saklı katmanlardaki nöronların sayıları girdi katmanındaki nöronların sayılarının 
2/3'ü ile çıktı katmanındaki nöronların sayısının toplamı kadar olabilir. Örneğin 
girdi katmanındaki nöron sayısı 5, çıktı katmanındaki 1 olsun. Bu durumda saklı 
katmandaki nöron sayısı 5 olabilir. 

- Saklı katmandaki nöronların sayısı girdi katmanındaki nöron sayısının iki 
katından fazla olmamalıdır.

--------------------------------------------------------------------------------- 
"""        


# reproducible traning

"""
--------------------------------------------------------------------------------- 
Yapay sinir ağı her farklı eğitimde farklı "w" ve "bias" değerlerini oluşturabilir. 
Yani ağın peformansı eğitimden eğitime değişebilir. Her eğitimde ağın farklı 
değerlere konumlanırılmasının nedenleri şunlardır:

1) train_test_split fonksiyonu her çalıştırıldığında aslında fonksiyon training_dataset 
ve test_dataset veri kümelerini karıştırarak elde etmektedir. 

2) Katmanlardaki "w" değerleri (ve istersek "bias" değerleri) programın her 
çalıştırılmasında rastgele başlangıç değerleriyle set edilmektedir.

3) fit işleminde her epoch sonrasında veri kümesi yeniden karıştırılmaktadır. 

Bir rastgele sayı üretiminde üretim aynı "tohum değerden (seed)" başlatılırsa hep 
aynı değerler elde edilir. Bu duruma rassal sayı üretiminin "reproducible olması" 
denmektedir. Eğer tohum değer belirtilmezse NumPy ve Tensorflow gibi kütüphanelerde 
bu tohum değer programın her çalıştırılmasında rastgele biçimde bilgisayarın 
saatinden alınmaktadır. 

--------------------------------------------------------------------------------- 
O halde eğitimden hep aynı sonucun elde edilmesi için (yani eğitimin "reproducible" 
hale getirilmesi için) yukarıdaki unsurların dikkate alınması gerekir. Tipik 
yapılması gereken birkaç şeyi şöyle belirtebiliriz:

1) scikit-learn makine öğrenmesi kütüphanelerinde aşağı seviyeli kütüphane olarak 
NumPy kullanıldığı için NumPy'ın rassal sayı üreticisinin tohum değeri belli bir 
değerle set edilebilir. Örneğin:

import numpy as np

np.random.seed(12345)


2) Tensorflow kütüphanesi bazı işlemlerde kendi rassal sayı üreticisini kullanmaktadır. 
Onun tohum değeri de belli bir değerle set edilebilir. Örneğin:

from tensorflow.keras.utils import set_random_seed

set_random_seed(78901)


Tabii yukarıdaki işlemler yapılsa bile rassal sayı üretimi "reproducible" hale 
getirilemeyebilir. Çünkü bu durum bu kütüphanelerin rassal sayı üretiminin hangi 
kütüphaneler kullanılarak yapıldığı ile ilgilidir. Yukarıdaki iki madde sezgisel 
bir çıkarımı ifade etmektedir.    

Pekiyi neden ağın her eğitilmesinde aynı sonuçların elde edilmesi (yani ağın 
"reproducible" sonuçlar vermesini) isteyebiliriz? İşte bazen modellerimizde ve 
algoritmalarımızda yaptığımız değişiklikleri birbirleriyle kıyaslamak isteyebiliriz. 
Bu durumda kıyaslamanın hep aynı biçimde yapılmasını sağlayabilmek için rassal 
bir biçimde alınan değerlerin her çalıştırmada aynı değerler olmasını sağlamamız 
gerekir. 

Tabii aslında algoritmaları karşılaştırmak için bu biçimde "reproducible" 
rassal sayı üretimi yapmak yerine algoritmaları çokça çalıştırıp bir ortalama 
değere de bakılabilir. 

O halde biz bu iki ayarlamayı yaparak yukarıdaki modelimizi çalıştırırsak bu durumda 
her eğitimde ve test işleminde aynı sonucu elde edebiliriz.

Aşağıda buna bir örnek verilmiştir. 

--------------------------------------------------------------------------------- 
import pandas as pd
import numpy as np
from tensorflow.keras.utils import set_random_seed

np.random.seed(1234567)
set_random_seed(678901)

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\2- KerasIntroduction\diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

predict_result = model.predict(predict_dataset)
print(predict_result)

for result in predict_result[:, 0]:
    print('Şeker hastası' if result > 0.5 else 'Şeker Hastası Değil')

--------------------------------------------------------------------------------- 
"""    


# Aktivasyon Fonksiyonları

"""
--------------------------------------------------------------------------------- 
Katmanlardaki aktivasyon fonksiyonları ne olmalıdır? Girdi katmanı gerçek bir katman 
olmadığına göre orada bir aktivasyon fonksiyonu yoktur. Saklı katmanlardaki aktivasyon 
fonksiyonları için çeşitli seçenekler bulunmaktadır.

Özellikle son yıllarda saklı katmanlarda en fazla tercih edilen aktivasyon fonksiyonu 
"relu (rectified linear unit)" denilen aktivasyon fonksiyonudur. Bu fonksiyona 
İngilizce "rectifier" da denilmektedir.  Relu fonksiyonu şöyledir:

x >= 0  ise y = x
x < 0   ise y = 0

Yani relu fonksiyonu x değeri 0'dan büyük ise (ya da eşit ise) aynı değeri veren, 
x değeri 0'dan küçük ise 0 değerini veren fonksiyondur. relu fonksiyonunu basit 
bir biçimde aşağıdaki gibi yazabiliriz:

def relu(x):
  return np.maximum(x, 0)  

NumPy kütüphanesinin maximum fonksiyonunun birinci parametresi bir NumPy dizisi 
ya da Python listesi biçiminde girilirse fonksiyon bu dizinin ya da listenin her 
elemanı ile maximum işlemi yapmaktadır. Örneğin:

>>> import numpy as np
>>> x = [10, -4, 5, 8, 1]
>>> y = np.maximum(x, 3)
>>> y
array([10,  3,  5,  8,  3])

Relu fonksiyonun grafiği aşağıdaki gibi çizilebilir. 

--------------------------------------------------------------------------------- 
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
      return np.maximum(x, 0)  

x = np.linspace(-10, 10, 1000)
y = relu(x)

plt.title('Relu Function', fontsize=14, fontweight='bold', pad=20)
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-10, 10)
plt.plot(x, y, color='red')
plt.show()

--------------------------------------------------------------------------------- 
Aktivasyon fonksiyonları katman nesnelerine isimsel girilebileceği gibi 
tensorflow.keras.activations modülündeki fonksiyonlar biçiminde de  girilebilir. 
Örneğin:

from tensorflow.keras.activations import relu
...
model.add(Dense(16, activation=relu, name='Hidden'))


Bu modüldeki fonksiyonlar keras Tensorflow kullanılarak yazıldığı için çıktı olarak 
Tensor nesneleri vermektedir. Biz relu grafik çizdirirken fonksiyonunu kendimiz 
yazmak yerine tensorflow içerisindeki fonksiyonu doğurdan kullanabiliriz. Tensor
nesnelerinin NumPy dizilerine dönüştürülmesi için Tensor sınıfının numpy metodu 
kullanılabilir. Örneğin:

    
from tensorflow.keras.activations import relu

x = np.linspace(-10, 10, 1000)
y = relu(x).numpy()

--------------------------------------------------------------------------------- 
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)

# y = relu(x)

from tensorflow.keras.activations import relu
y = relu(x)

plt.title('Relu Function', fontsize=14, fontweight='bold', pad=20)
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-10, 10)
plt.plot(x, y, color='red')
plt.show()

--------------------------------------------------------------------------------- 

--------------------------------------------------------------------------------- 
İkili sınıflandırma problemlerinde çıktı katmanında en fazla kullanılan aktivasyon 
fonksiyonu "sigmoid" denilen fonksiyondur. Yukarıdaki "diabetes" örneğinde biz çıktı 
katmanında sigmoid fonksiyonunu kullanmıştık. Gerçekten de ikili sınıflandırma 
problemlerinde ağın çıktı katmanında tek bir nöron bulunur ve bu nörounun da 
aktivasyon fonksiyonu "sigmoid" olur. 

Pekiyi sigmoid nasıl bir fonksiyondur? Bu fonksiyona "lojistik (logistic)" fonksiyonu 
da denilmektedir. Fonksiyonun matematiksel ifadesi şöyledir:

y = 1 / (1 + e ** -x)

Burada e değeri 2.71828... biçiminde irrasyonel bir değerdir. Yukarıdaki kesrin 
pay ve paydası e ** x ile çarpılırsa fonksiyon aşağıdaki gibi de ifade edilebilir:

y = e ** x / (1 + e ** x)

Fonksiyona "sigmoid" isminin verilmesinin nedeni S şekline benzemesinden dolayıdır. 
Sigmoid eğrisi x = 0 için 0.5 değerini veren x pozitif yönde arttıkça 1 değerine 
hızla yaklaşan, x negatif yönde arttıkça 0 değerine hızla yaklaşan S şeklinde bir 
eğridir. Sigmoid fonksiyonunun (0, 1) arasında bir değer verdiğine dikkat ediniz. 
x değeri artıkça eğri 1'e yaklaşır ancak hiçbir zaman 1 olmaz. Benzer biçimde x 
değeri azaldıkça eğri 0'a yaklaşır ancak hiçbir zaman 0 olmaz. 

Sigmoid fonksiyonu makine öğrenmesinde ve istatistikte belli bir gerçek değeri 0 
ile 1 arasına hapsetmek için sıkça kullanılmaktadır. Sigmoid çıktısı aslında bir 
bakımdan çıktının 1 olma olasılığını vermektedir. Tabii biz kestirimde bulunurken 
kesin bir yargı belirteceğimiz için eğrinin orta noktası olan 0.5 değerini referans 
alırız. Ağın ürettiği değer 0.5'ten büyükse bunu 1 gibi, 0.5'ten küçükse 0 gibi 
yorumlayabiliriz.  Sigmoid eğrisi aşağıdaki gibi çizilebilir.

--------------------------------------------------------------------------------- 
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = np.e ** x / (1 + np.e ** x)

plt.title('Sigmoid (Logistic) Function', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-1, 1)

plt.plot(x, y)
plt.show()


Yine benzer biçimde tensorflow.keras.activations modülü içerisinde sigmoid fonksiyonu 
zaten hazır biçimde bulunmaktadır. Tabii bu fonksiyon da bize Tensorflow'daki bir 
Tensor nesnesini vermektedir.

--------------------------------------------------------------------------------- 
Sigmoid fonksiyonu nasıl ortaya çıkartılmıştır. Aslında bu fonksiyonun elde edilmesinin 
bazı mantıksal gerekçeleri vardır. Ayrıca sigmoid fonksiyonunun birinci türevi 
Gauss eğrisine benzemektedir. Aşağıdaki örnekte Sigmoid fonksiyonunun birinci 
türevi alınıp eğrisi çizdirilmiştir. Ancak bu örnekte henüz görmediğimiz SymPy 
kütüphanesini kullandık. Sigmoid fonksiyonun birinci türevi şöyledir:

sigmoid'(x) = exp(x)/(exp(x) + 1) - exp(2*x)/(exp(x) + 1)**2

--------------------------------------------------------------------------------- 
import sympy
from sympy import init_printing

init_printing()

x = sympy.Symbol('x')
fx = sympy.E ** x / (1 + sympy.E ** x)
dx = sympy.diff(fx, x)

print(dx)

import numpy as np

np.linspace(-10, 10, 1000)
pdx = sympy.lambdify(x, dx)

x = np.linspace(-10, 10, 1000)
y = pdx(x)

import matplotlib.pyplot as plt

plt.title('First Derivative of Sigmoid Function', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-0.4, 0.4)

plt.plot(x, y)
plt.show()

--------------------------------------------------------------------------------- 

--------------------------------------------------------------------------------- 
Diğer çok kullanılan bir aktivasyon fonksiyonu da "hiperbolik tanjant" fonksiyonudur. 
Bu fonksiyona kısaca "tanh" fonksiyonu da denilmektedir. Fonksiyonun matematiksel 
ifadesi şöyledir:

f(x) = (e ** (2 * x) - 1) / (e ** (2 * x) + 1)

Fonksiyonun sigmoid fonksiyonuna benzediğine ancak üstel ifadenin x yerine 2 * x 
olduğuna dikkat ediniz. Tanh fonksiyonu adeta sigmoid fonksiyonunun (-1, +1) arası 
değer veren biçimi gibidir. Fonksiyon yine S şekli biçimindedir. Orta noktası 
x = 0'dadır.

Tanh fonksiyonu saklı katmanlarda da bazen çıktı katmanlarında da kullanılabilmektedir. 
Eskiden bu fonksiyon çok yoğun kullanılıyordu. Ancak artık saklı katmanlarda en 
çok relu fonksiyonu tercih edilmektedir. Fakat tanh fonksiyonunun daha iyi sonuç 
verdiği modeller de bulunmaktadır. 

tanh fonksiyonu Keras'ta tensorflow.keras.activations modülünde tanh ismiyle de 
bulunmaktadır.

--------------------------------------------------------------------------------- 
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = (np.e ** (2 * x) - 1) / (np.e ** (2 * x) + 1)

plt.title('Hiperbolik Tanjant (tanh) Fonksiyonunun Grafiği', fontsize=14, pad=20, fontweight='bold')
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_ylim(-1, 1)

plt.plot(x, y)
plt.show()

---------------------------------------------------------------------------------
 
--------------------------------------------------------------------------------- 
Diğer çok karşılaşılan bir aktivasyon fonksiyonu da "softmax" isimli fonksiyondur. 
Softmax fonksiyonu çok sınıflı sınıflandırma problemlerinde çıktı katmanlarında 
kullanılmaktadır. Bu aktivasyon fonksiyonu Keras'ta "softmax" ismiyle bulunmaktadır. 
Örneğin bir resmin "elma", "armut", "kayısı", "şeftali", "karpuz" resimlerinden 
hangisi olduğunu anlamak için kullanılan sınıflandırma modeli çok sınıflı bir 
sınıflandırma modelidir. Buna istatistikte "çok sınıflı lojistik regresyon (multinomial 
logistic regression)" da denilmektedir. Bu tür problemlerde sinir ağında sınıf 
sayısı kadar nöron bulundurulur. Örneğin yukarıdaki "elma", "armut", "kayısı", 
"şeftali", "karpuz" resim sınıflandırma probleminde ağın çıktısında 5 nöron bulunacaktır. 

Ağın çıktı katmanındaki tüm nöronların aktivasyon fonksiyonları "softmax" yapılırsa 
tüm çıktı katmanındaki nöronların çıktı değerlerinin toplamı her zaman 1 olur. 
Böylece çıktı katmanındaki nöronların çıktı değerleri ilgili sınıfın olasılığını 
belirtir hale gelir. Biz de toplamı 1 olan çıktıların en yüksek değerine bakarız 
ve sınıflandırmanın o sınıfı kestirdiğini kabul ederiz. Örneğin yukarıdaki "elma", 
"armut", "kayısı", "şeftali", "karpuz" sınıflandırma probleminde ağın çıktı 
katmanındaki nöronların çıktı değerlerinin şöyle olduğunu varsayalım: 

Elma Nöronunun Çıktısı ---> 0.2
Armut Nöronunun Çıktısı ---> 0.2
Kayısı Nöronunun Çıktısı ---> 0.3
Şeftali Nöronunun Çıktısı ---> 0.2
karpuz Nöronunun Çıktısı ---> 0.1

Burada en büyük çıktı 0.3 olan kayısı nöronuna ilişkindir. O halde biz bu kestirimin 
"kayısı" olduğuna karar veririz. Softmax fonksiyonu bir grup değer için o grup 
değerlere bağlı olarak şöyle hesaplanmaktadır: 

softmax(x) = np.e ** x / np.sum(np.e ** x)

Burada gruptaki değerler x vektörüyle temsil edilmektedir.  Fonksiyonda değerlerinin 
e tabanına göre kuvvetleri x değerlerinin e tabanına göre kuvvetlerinin toplamına 
bölünmüştür. Bu işlemden yine gruptaki eleman sayısı kadar değer elde edilecektir. 
Tabii bu değerlerin toplamı da 1 olacaktır. Örneğin elimizde aşağıdaki gibi x 
değerleri olsun:
    
x = np.array([3, 6, 4, 1, 7])

Şimdi bu x değerlerinin softmax değerlerini elde edelim:  

>>> import numpy as np
>>> x = np.array([3, 6, 4, 1, 7])
>>> x
array([3, 6, 4, 1, 7])
>> sm = np.e ** x / np.sum(np.e ** x)
>>> sm
array([0.0127328 , 0.25574518, 0.03461135, 0.0017232 , 0.69518747])
>>> np.sum(sm)
1.0    

--------------------------------------------------------------------------------- 

--------------------------------------------------------------------------------- 
Diğer çok kullanılan aktivasyon fonksiyonlarından biri de "linear" aktivasyon 
fonksiyonudur. Aslında bu fonksiyon y = x ya da f(x) = x fonksiyonudur. Yani "linear" 
fonksiyonu girdi ile aynı değeri üretmektedir. Başka bir deyişle bir şey yapmayan 
bir fonksiyondur. Pekiyi böyle bir aktivasyon fonksiyonunun ne anlamı olabilir? 

Bu aktivasyon fonksiyonu "regresyon problemlerinde (lojistik olmayan regresyon 
problemlerinde)" çıktı katmanında kullanılmaktadır. Lojistik olmayan regresyon 
problemleri çıktı olarak bir sınıf bilgisi değil gerçek bir değer bulmaya çalışan 
problemlerdir. Örneğin bir evin fiyatının kestirilmesi, bir otomobilin mil başına 
yaktığı yakıt miktarının kestirilemsi gibi problemler lojistik  olmayan regresyon 
problemleridir. 

Anımsanacağı gibi biz kursumuzda bir sayı kestirmek için kullanılan regresyon 
modellerine vurgulama amaçlı "lojistik olmayan regresyon modelleri)" diyoruz. 
Aslında "regresyon modeli" denildiğinde zaten default olarak lojistik olmayan 
regresyon modelleri anlaşılmaktadır.

linear aktivasyon fonksiyonu Keras'ta "linear" ismiyle kullanılmaktadır. Her ne 
kadar bir şey yapmıyorsa da bu aktivasyon fonksiyonu aynı zamanda 
tensorflow.keras.activations modülünde linear isimli bir fonksiyon biçiminde de 
bulunmaktadır. Örneğin:

>>> from tensorflow.keras.activations import linear
>>> x = [1, 2, 3, 4, 5]
>>> x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> result = linear(x)
>>> result
array([1., 2., 3., 4., 5.])

--------------------------------------------------------------------------------- 
import numpy as np
import matplotlib.pyplot as plt

def linear(x):
      return x

x = np.linspace(-10, 10, 1000)
y = linear(x)


# from tensorflow.keras.activations import linear

# y = linear(x).numpy()


plt.title('Linear Function', fontsize=14, fontweight='bold', pad=20)
axis = plt.gca()
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)
axis.set_ylim(-10, 10)
plt.plot(x, y, color='red')
plt.show()

--------------------------------------------------------------------------------- 
"""


# Loss Fonksiyonları

"""
--------------------------------------------------------------------------------- 
loss fonksiyonları gerçek değerlerle ağın tahmin ettiği değerleri girdi olarak 
alıp bu farklılığı bir sayısal sayısal değerle ifade eden fonksiyonlardır. Optimizasyon 
algoritmaları bu loss fonksiyonlarının değerini düşürmeye çalışmaktadır. Gerçek
değerlerle ağın ürettiği değerlerin arasındaki farkın minimize edilmesi aslında 
ağın gerçek değerlere yakın değerler üretmesi anlamına gelmektedir. 

Eğitim batch batch yapıldığı için loss fonksiyonları tek bir satırın çıktısından 
değil n tane satırın çıktısından hesaplanmaktadır. Yani bir batch işleminin gerçek 
sonucu ile ağdan o batch için elde edilecek kestirim sonuçlarına dayanılarak loss 
değerleri hesaplanmaktadır. Örneğin batch_size = 32 olduğu durumda aslında Keras 
ağa 32'lik bir giriş uygulayıp 32'lik bir çıktı elde eder. Bu 32 çıktı değeri 
gerçek 32 değerle loss fonksiyonuna sokulur.

--------------------------------------------------------------------------------- 
Lojistik olmayan regresyon problemleri için en yaygın kullanılan loss fonksiyonu 
"Mean Squared Error (MSE)" denilen fonksiyondur.  Bu fonksiyona Türkçe "Ortalama 
Karesel Hata" diyebiliriz.  MSE fonksiyonu çıktı olarak gerçek değerlerden kestirilen 
değerlerin farkının karelerinin ortalamasını vermektedr. Fonksiyonun sembolik 
gösterimi şöyledir:

mse = np.mean((y - y_hat) ** 2)

Burada y gerçek değerleri, y_hat ise kestirilen değerleri belirtmektedir. Örneğin:

>>> y = np.array([1, 2, 3, 4, 5])
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> np.mean((y - y_hat) ** 2)
0.020080000000000032

Aynı işlemi tensorflow.keras.losses modülündeki mse (ya da mean_squared_error) 
fonksiyonuyla da aşağıdaki gibi yapabilirdik:

>>> from tensorflow.keras.losses import mse
>>> mse(y, y_hat)
<tf.Tensor: shape=(), dtype=float64, numpy=0.020080000000000032>

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Yine lojistik olmayan regresyon problemleri için kullanılabilen diğer bir loss 
fonksiyonu da "Mean Absolute Error (MAE)" isimli fonksiyondur. Bu fonksiyona da 
Türkçe "Ortalama Mutlak Hata" diyebiliriz. Ortalama mutlak hata gerçek değerlerden 
kestirilen değerlerin farklarının mutlak değerlerinin ortalaması biçiminde 
hesaplanmaktadır. Sembolik gösterimi şöyledir:

mae = np.mean(np.abs(y - y_hat))

Burada y gerçek değerleri y_hat ise ağın kestirdiği değerleri belirtmektedir. 
Lojistik olmayan Regresyon problemleri için loss fonksiyonu olarak çoğu kez MSE 
tercih edilmektedir. Çünkü kare alma işlemi algoritmalar için daha uygun bir işlemdir. 
Aynı zamanda değerleri daha fazla farklılaştırmaktadır. MAE loss fonksiyonundan 
ziyade metrik değer olarak "insan algısına yakınlık" oluşturduğu için tercih 
edilmektedir. Örneğin:

>>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> mae = np.mean(np.abs(y - y_hat))
>>> mae
0.12400000000000003


Ortalama karesel hata bir metrik değer olarak bizim için iyi bir çağrışım yapmamaktadır. 
Halbuki ortalama mutlak hata bizim için anlamlı bir çağrışım yapmaktadır. Örneğin 
ağımızın ortalama mutlak hatası 0.124 ise gerçek değer ağımızın bulduğu değerden 
0.124 solda ya da sağda olabilir. 

Yine ortalama mutlak hata tensorflow.keras.losses modülü içerisindeki "mae" ya da 
"mean_absolute_error" isimli fonksiyonla da hesaplanabilmektedir. Örneğin:

>>> from tensorflow.keras.losses import mae
>>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> result = mae(y, y_hat)
>>> result
<tf.Tensor: shape=(), dtype=float64, numpy=0.12400000000000003>

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Lojistik olmayan regresyon problemleri için diğer bir loss fonksiyonu da "Mean 
Absolute Percentage Error (MAPE)" isimli fonksiyondur. Fonkisyonun sembolik 
ifadesi şöyledir:

mape = 100 * np.mean(np.abs(y - y_hat) / y)

Burada y gerçek değerleri y_hat ise ağın kestirdiği değerleri belirtmektedir. Örneğin:

>>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> mape = 100 * np.mean(np.abs(y - y_hat) / y)
>>> mape
5.413333333333335

Tabii aynı işlemi yine tensorflow.keras.losses modülündeki "mape" fonksiyonuyla da yapabiliriz:

>>> from tensorflow.keras.losses import mape
>>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> result = mape(y, y_hat)
>>> result
<tf.Tensor: shape=(), dtype=float64, numpy=5.413333333333335>

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Lojistik olmayan regresyon problemleri için diğer bir loss fonksiyonu da "Mean 
Squared Logarithmic Error (MSLE)" isimli fonksiyondur.  Bu fonksiyon gerçek değerlerle 
kestirilen değerlerin logaritmalarının farklarının karelerinin ortalaması 
biçiminde hesaplanır. Sembolik ifadesi şöyledir:

msle = np.mean((np.log(y) - np.log(y_hat)) ** 2)

Bazen bu fonksiyon gerçek ve kestirilen değerlere 1 toplanarak da oluşturulabilmektedir.
(Tensorflow "msle" fonksiyonunu bu biçimde kullanmaktadır):

msle = np.mean((np.log(y + 1) - np.log(y_hat + 1)) ** 2)

Örneğin:

>>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> msle = np.mean((np.log(y + 1) - np.log(y_hat + 1)) ** 2)
>>> msle
0.0015175569737783628

Aynı işlemi tensorflow.keras.losses modülündeki "msle" fonksiyonuyla da yapabiliriz:

>>>     
>>> y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
>>> y_hat = np.array([1.1, 1.9, 3.2, 3.8, 5.02])
>>> result = msle(y, y_hat)
>>> result
    <tf.Tensor: shape=(), dtype=float64, numpy=0.0015175569737783555>
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
İkili sınıflandırma problemleri için en yaygın kullanılan loss fonksiyonu 
"Binary Cross-Entropy (BCE)" denilen fonksiyondur. Bu fonksiyonun sembolik gösterimi 
şöyledir:

bce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

Burada bir toplam teriminin olduğunu görüyorsunuz. Gerçek değerler 0 ya da 1 olduğuna 
göre bu toplam teriminin ya sol tarafı ya da sağ tarafı 0 olacaktır. Burada yapılmak 
istenen şey aslında isabet olasılığının logaritmalarının ortalamasının alınmasıdır. 

Örneğin gerçek y değeri 0 olsun ve ağda sigmoid çıktısından 0.1 elde etmiş olsun. 
Bu durumda toplam ifadesinin sol tarafı 0, sağ tarafı ise log(0.9) olacaktır. Şimdi 
gerçek değerin 1 ancak ağın sigmoid çıktısından elde edilen değerim 0.9 olduğunu 
düşününelim. Bu kez toplamın sağ tarafı 0, sol tarafı ise log(0.9) olacaktır. İşte 
fonksiyonda bu biçimde isabet olasılıklarının logaritmalarının ortalaması bulunmaktadır. 
Örneğin:

>>> y = np.array([1, 0, 1, 1, 0])
>>> y_hat = np.array([0.9, 0.05, 0.095, 0.89, 0.111])
>>> -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
0.5489448114302314

Aynı işlemi tensorflow.keras.losses modülündeki binary_crossentropy isimli fonksiyonla 
da yapabiliriz:

>>> y_hat = np.array([0.9, 0.05, 0.095, 0.89, 0.111])
>>> result = binary_crossentropy(y, y_hat)
>>> result
<tf.Tensor: shape=(), dtype=float64, numpy=0.5489445126600796>

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Cok sınıflı sınıflandırma problemleri için en yaygın kullanılan loss fonksiyonu 
ise "Categorical Cross-Entropy (CCE)" isimli fonksiyondur. CCE fonksiyonu aslında 
BCE fonksiyonun çoklu biçimidir. Tabii CCE değerini hesaplayabilmek için ağın 
kategori sayısı kadar çıktıya sahip olması ve bu çıktıların toplamının da 1 olması 
gerekmektedir. Başka bir deyişle ağın çıktı katmanındaki nöronların aktivasyon 
fonksiyonları "softmax" olmalıdır. Ayrıca CCE çok sınıflı bir entropy hesabı 
yaptığına göre gerçek değerlerin one hot encoding biçiminde kodlanmış olması gerekir. 
(Yani örneğin ileride göreceğimiz gibi K sınıflı bir sınıflandırma problemi için 
biz ağa çıktı olarak "one hot encoding" kodlanmış K tane y değerini vermeliyiz.) 
K tane sınıf belirtem bir satırın CCE değeri şöyle hesaplanır (tabii burada K tane 
"one hot encoding" edilmiş gerçek değer ile M tane softmax çıktı değeri söz konusu 
olmalıdır):

cce = -np.sum(yk * log(yk_hat))

Burada yk one hot encoding yapılmış gerçek değerleri yk_hat ise softmax biçiminde 
elde edilmiş ağın çıktı katmanındaki değerleri temsil etmektedir. 

---------------------------------------------------------------------------------
"""


# metric fonksiyonlar
"""    
---------------------------------------------------------------------------------
Anımsanacağı gibi "metrik fonksiyonlar" her epoch'tan sonra sınama verlerine uygulanan 
ve eğitimin gidişatı hakkında bilgi almak için kullanılan performans fonksiyonlar 
idi. Problemin türüne göre çeşitli metrik fonksiyonlar hazır biçimde bulunmaktadır. 
Birden fazla metrik fonksiyon kullanılabileceği için metrik fonksiyonlar compile 
metodunda "metrics" parametresine bir liste biçiminde girilmektedir. Metrik fonksiyonlar 
yazısal biçimde girilebilceği gibi tensorflow.keras.metrics modülündeki fonksiyonlar 
ve sınıflar biçiminde de girilebilmektedir. 

Metrik fonksiyonlar da tıpkı loss fonksiyonları gibi gerçek çıktı değerleriyle 
ağın ürettiği çıktı değerlerini parametre olarak almaktadır. Aslında loss fonksiyonları 
da bir çeşit metrik fonksiyonlardır. Ancak loss fonksiyonları optimizasyon 
algoritmaları tarafından minimize edilmek için kullanılmaktadır. Halbuki metrik 
fonksiyonlar bizim durumu daha iyi anlamamız için bizim tarafımızdan kullanılmaktadır.

Aslında loss fonksiyonları da bir çeşit metrik fonksiyonlar olarak kullanılabilir. 
Ancak Keras zaten bize loss fonksiyonlarının değerlerini her epoch'ta eğitim ve 
sınama verileri için vermektedir. Dolayısıyla örneğin ikili sınıflandırma problemi 
için eğer biz loss fonksiyonu olarak "binary_crossentropy" girmişsek ayrıca bu 
fonksiyonu metrik olarak girmenin bir anlamı yoktur. Özetle her loss fonksiyonu 
bir metrik fonksiyon gibi de kullanılabilir. Ancak her metrik fonksiyon bir loss 
fonksiyonu olarak kullanılmaz. 

---------------------------------------------------------------------------------    
"binary_accuracy" isimli metrik fonksiyon ikili sınıflandırma problemleri için en 
yaygın kullanılan metrik fonksiyondur. Bu fonksiyon kabaca kaç gözlemin değerinin 
kestirilen değerle aynı olduğunun yüzdesini vermektedir. Örneğin "diabetes.csv" 
veri kümesinde "binary_accuracy" değerinin 0.70 olması demek her yüz ölçümün 70 
tanesinin doğru biçimde kestirilmesi demektir. "binary_accuracy" metrik değeri 
Keras'ta isimsel olarak girilebileceği gibi tensorflow.keras.metrics modülündeki 
fonksiyon ismi olarak da girilebilir. 

Aslında Keras'ta programcı kendi loss fonksiyonlarını ve metrik fonksiyonlarını da 
yazabilir. Ancak tensorflow bu konuda yeterli dokümantasyona sahip değildir. 
Tensorflow kütüphanesinin çeşitli versiyonlarında çeşitli farklılıklar bulunabilmektedir. 
Bu nedenle bu fonksiyonların programcı tarafından yazılması için bu konuya dikkat 
etmek gerekir. 

Örneğin biz tensorflow.keras.metrics modülündeki binary_accuracy fonksiyonunu 
aşağıdaki gibi kullanabiliriz.

---------------------------------------------------------------------------------
from tensorflow.keras.metrics import binary_accuracy
import numpy as np

y = np.array([1, 0, 1, 1, 0])
y_hat = np.array([0.90, 0.7, 0.6, 0.9, 0.6])

result = binary_accuracy(y, y_hat) # %60'ını doğru tahmin edicek
print(result)

---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------    
Çok sınıflı sınıflandırma problemlerinde tipik okullanılan metrik fonksiyon 
"categorical_accuracy" isimli fonksiyondur. Bu fonksiyon da yine gözlemlerin yüzde 
kaçının tam olarak isabet ettirildiğini belirtmektedir. Örneğin ikili sınıflandırmada 
0.50 değeri iyi bir değer değildir. Çünkü zaten rastgele seçim yapılsa bile ortalama 
0.50 başarı elde edilmektedir. Ancak 100 sınıflı bir problemde 0.50 başarı düşük 
bir başarı olmayabilir. Yine biz Keras'ta "categorical_accuracy" metrik fonksiyonunu
isimsel biçimde ya da tensorflow.keras.metrics modülündeki fonksiyon ismiyle 
kullanabiliriz. 

tensorflow.keras.metrics modülündeki categorical_accuracy fonksiyonu aslında toplam 
isabetlere ilişkin bir Tensor nesnesi vermektedir, ortalama vermemektedir. Aşağıda 
bu fonksiyonun kullanımına bir örnek verilmiştir.

---------------------------------------------------------------------------------    
from tensorflow.keras.metrics import categorical_accuracy
import numpy as np

# elma, armut, kayısı

y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
y_hat = np.array([[0.2, 0.7, 0.1], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.2, 0.2]])

result = categorical_accuracy(y, y_hat)  
result_ratio = np.sum(result) / len(result)

print(result_ratio) # %80

---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------    
Lojistik olmayan regresyon problemleri için pek çok loss fonksiyonu aslında metrik 
olarak da kullanılabilmektedir. Örneğin biz böyle bir problemde loss fonksiyonu 
olarak "mean_squared_error" seçmişsek metrik fonksiyon olarak "mean_absolute_error" 
seçebiliriz. mean_absolute_error fonksiyonu loss fonksiyonu olarak o kadar iyi 
olmasa da metrik anlamda kişilerin kolay anlayabileceği bir fonksiyondur. Benzer 
biçimde lojistik olmayan regresyon problemlerinde "mean_asbolute_percentage_error", 
"mean_squared_logarithmic_error" fonksiyonları da metrik olarak kullanılabilmektedir. 

Loss fonksiyonların metrik fonksiyonlar olarak kullanılabileceğini belirtmiştik. 
Aslında örneğin mean_absolute_error loss fonksiyonu ile mean_absolute_error metrik 
fonksiyonu aynı işlemi yapmaktadır. Ancak Keras'ta bu fonksiyonlar tensorflow.keras.losses 
ve tensorflow.keras.metrics modüllerinde ayrı ayrı bulundurulmuştur. 

Metrik fonksiyonlar yazısal biçimde girilecekse onlar için uzun ya da kısa isimler 
kullanılabilmektedir. Örneğin:

'binary_accuracy' (kısa ismi yok)
'categorical_accuracy' (kısa ismi yok)
'mean_absolute_error' (kısa ismi 'mae')
'mean_absolute_percentage_error' (kısa ismi 'mape')
'mean_squared_error' kısa ismi ('mse')

---------------------------------------------------------------------------------    
"""    
    


# Model Parametrelerinin Saklanması   
"""
---------------------------------------------------------------------------------    
Bir sinir ağı eğitildikten sonra onun diskte saklanması gerekebilir. Çünkü eğitim 
uzun sürebilir ve her bilgisayarı açtığımızda ağı yeniden eğitmemiz verimsiz bir 
çalışma biçimidir. Ayrıca eğitim tek seferde de yapılmayabilir. Yani örneğin bir 
kısım verilerle eğitim yapılıp sonuçlar saklanabilir. Sonra yeni veriler elde 
edildikçe eğitime devam edilebilir.

Keras'ta yapılan eğitimlerin saklanması demekle neyi kastediyoruz? Tabii öncelikle 
tüm nöronlardaki "w" ve "bias" değerleri kastedilmektedir. Ayrıca Keras bize tüm 
modeli saklama imkanı da vermektedir. Bu durumda modelin yeniden kurulmasına da 
gerek kalmaz. Tüm model katmanlarıyla "w" ve "bias" değerleriyle saklanıp geri 
yüklenebilmektedir. 

Sinir ağı modelini saklamak için hangi dosya formatı uygundur? Çok fazla veri söz 
konusu olduğu için buna uygun tasarımı olan dosya formatları tercih edilmelidir. 
Örneğin bu işlem için "CSV" dosyaları hiç uygun değildir. İşte bu tür amaçlar için 
ilk akla gelen format "HDF (Hieararchical Data Format)" denilen formattır. Bu 
formatın beşinci versiyonu HDF5 ya da H5 formatı olarak bilinmektedir. 

Modeli bir bütün olarak saklamak için Sequential sınıfının save isimli metodu kullanılır 
save metodunun birinci parametresi dosyanın yol ifadesini almaktadır. save_format 
parametresi saklanacak dosyanın formatını belirtir.??? Bu parametre girilmezse dosya 
TensorFlow kütüphanesinin kullandığı "tf" formatı ile saklanmaktadır.??? Biz HDF5 
formatı için bu parametreye 'h5' girmeliyiz. Örneğin:

model.save('diabetes.h5', save_format='h5')

---------------------------------------------------------------------------------   

---------------------------------------------------------------------------------   
Aslında modelin tamamını değil yalnızca "w" ve "bias" değerlerini save etmek de 
mümkündür. Bunun için Sequential sınıfının save_weights metodu kullanılmaktadır. 
Örneğin:

model.save_weights('diabetes-weights', save_format='h5')

Yalnızca modelin "w" ve "bias" değerleri saklanmıştır. 

---------------------------------------------------------------------------------   

---------------------------------------------------------------------------------   
HDF5 formatıyla sakladığımız model bir bütün olarak tensorflow.keras.models modülündeki 
load_model fonksiyonu ile geri yüklenebilir. load_model fonksiyonu bizden yüklenecek 
dosyanın yol ifadesini alır. Fonksiyon geri dönüş değeri olarak model nesnesini 
vermektedir. Örneğin:

from tensorflow.keras.models import load_model

model = load_model('diabetes.h5')


Artık biz modeli fit ettiğimiz biçimiyle tamamen geri almış durumdayız. Doğrudan 
predict metoduyla kestirim yapabiliriz. 

Aşağıdaki örnekte yukarıda save ettigimiz model yüklenebilir. 

---------------------------------------------------------------------------------  
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('diabetes.h5')

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

predict_result = model.predict(predict_dataset)
print(predict_result)

---------------------------------------------------------------------------------   

---------------------------------------------------------------------------------   
save_weights metodu yalnızca "w" ve "bias" değerlerini save etmektedir. Bizim bunu 
geri yükleyebilmemiz için modeli yeniden aynı biçimde oluşturmamız ve compile işlemini 
yapmamız gerekir. "w" ve "bias" değerlerinin geri yüklenmesi için Sequential 
sınıfının load_weights metodu kullanılmaktadır. Örneğin:

model.load_weights('diabetes-weights.h5')

save_weights model bilgilerini saklamadığı için modelin aynı biçimde yeniden 
oluşturulması gerekmektedir. Modelin "w" ve "bias" değerlerini load_weights metodu 
ile geri yüklerken veri kümesini oluşturmamız gerekmemektedir. save_weights"
metodu model yalnızca "w" ve "bias" değerlerini save ettiği için model programcı 
tarafından orijinal haliyle oluşturulmalıdır.

---------------------------------------------------------------------------------   
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((8,)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.load_weights('diabetes-weights.h5')

import numpy as np  

predict_dataset = np.array([[2 ,90, 68, 12, 120, 38.2, 0.503, 28],
                            [4, 111, 79, 47, 207, 37.1, 1.39, 56],
                            [3, 190, 65, 25, 130, 34, 0.271, 26],
                            [8, 176, 90, 34, 300, 50.7, 0.467, 58],
                            [7, 106, 92, 18, 200, 35, 0.300, 48]])

predict_result = model.predict(predict_dataset)
print(predict_result)

for result in predict_result[:, 0]:
    print('Şeker hastası' if result > 0.5 else 'Şeker Hastası Değil')
    
---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------    
Biz katman nesnelerini (Dense nesnelerini) model sınıfının (Sequential sınıfının) 
add metotlarıyla modelimize ekledik. Bu katman nesnelerini ileride kullanmak 
istediğimizde nasıl geri alabiliriz? Tabi ilk akla gelen yöntem katman nesnelerini 
yaratırken aynı zamanda saklamak olabilir. Örneğin:

model = Sequential(name='Diabetes')

model.add(Input((8,)))
layer1 = Dense(16, activation='relu', name='Hidden-1')
model.add(layer1)
layer2 = Dense(16, activation='relu', name='Hidden-2')
model.add(layer2)
layer3 = Dense(1, activation='sigmoid', name='Output')
model.add(layer3)
model.summary()

Aslında böyle saklama işlemine gerek yoktur. Zaten model nesnesinin (Sequential sınıfının) 
layers isimli özniteliği bir indeks eşliğinde bizim eklediğimiz katman nesnelerini 
bize vermektedir. layers örnek özniteliği bir Python listesidir. Örneğin:

layer3 = model.layers[2]


Tabii layers özniteliği bize Input katmanını gereksiz olduğundan dolayı vermemktedir. 
Buradaki 0'ıncı indeks ilk saklı katmanı belirtmektedir. 

---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------    
Biz yukarıda tüm modeli, modelin "w" ve "bias" değerlerini saklayıp geri yükledik. 
Peki yalnızca tek bir katmandaki ağırlık değerlerini alıp geri yükleyebilir miyiz? 
İşte bu işlem katman sınıfının (yani Dense sınıfının) get_weights  ve set_weights 
isimli metotları ile yapılmaktadır. Biz bu metotlar sayesinde bir katman nesnesindeki 
"w" ve "bias" değerlerini NumPy dizisi olarak elde edip geri yükleyebiliriz. 

Dense sınıfının get_weights metodu iki elemanlı bir listeye geri dönmektedir. Bu 
listenin her iki elemanı da NumPy dizisidir. Listenin ilk elemanı (0'ıncı indeksli 
elemanı) o katmandaki nöronların "w" değerlerini, ikinci elemanı ise "bias" değerlerini 
belirtmektedir. Katmandaki nöronların "w" değerleri iki boyutlu bir NumPy dizisi 
biçimindedir. 

Burada k'ıncı sütun önceki katmanın nöronlarının sonraki katmanın k'ıncı nöronuna 
bağlanmasındaki ağırlık değerlerini belirtmektedir. Benzer biçimde i'inci satır 
ise önceki katmanın i'inci nöronunun ilgili katmanın nöron bağlantısındaki ağırlık 
değerlerini belirtmektedir. Bu durumda örneğin [i, k] indeksindeki eleman önceki 
katmanın i'inci nörounun ilgili katmanın k'ıncı nöronu ile bağlantısındaki ağırlığı 
belirtmektedir. 

get_weights metodunun verdiği listenin ikinci elemanı (1'inci indeksli elemanı) 
nöronların "bias" değerlerini vermektedir. Bias değerlerinin ilgili katmandaki 
nöron sayısı kadar olması gerektiğine dikkat ediniz. Çünkü her nöron için bir 
tane bias değeri vardır. 

Örneğin yukarıdaki "diabetes" örneğinde ilk saklı katmandaki ağırlıkları şöyle 
elde edebiliriz:

layer = model.layers[0]
weights, bias = layer.get_weights()

Burada weights dizisinin shape'i yazdırıldığında (8, 16) görülecektir. bias 
dizisinin shape'i yazdırıldığında ise (16,) görülecektir.

---------------------------------------------------------------------------------    
import pandas as pd

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\2- KerasIntroduction\diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)
eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
    
    
    
hidden1 = model.layers[0]
weights, bias = hidden1.get_weights()


print(weights)

print(bias)

print("5'inci girdi nöronunun ilk katmanın 9'uncu nöronununa bağlantısındaki w değeri")
w = weights[5, 9]
print(w)

---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------    
Bir katmandaki "w" ve "bias" değerlerini Dense sınıfının set_weights metodu ile 
geri yükleyebiliriz. Örneğin:
    
hidden1 = model.layers[0]
weights, bias = hidden1.get_weights()

weights = weights + 0.1
hidden1.set_weights([weights, bias])

Burada birinci saklı katmandaki ağırlık değerlerine 0.1 eklenerek ağırlık değerleri 
geri yüklenmiştir. set_weights metodunun iki elemanı bir liste aldığına dikkat 
ediniz. Nasıl get_weights metodu hem "w" hem de "bias" değerlerini veriyorsa 
set_weights metodu da hem "w" hem de "bias" değerlerini istemektedir. 

Aşağıdaki örnekte ilk saklı katmandaki "w" değerlerine 0.1 toplanarak değerler 
geri yüknemiştir. 

hidden1 = model.layers[0]
weights, bias = hidden1.get_weights()

weights = weights + 0.1
hidden1.set_weights([weights, bias])

---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------    
Yapay sinir ağları ile kestirim yapabilmek için bazı aşamalardan geçmek gerekir. 
Bu aşamaları şöyle özetleyebiliriz:

1) Hedefin Belirlenmesi: Öncelikle uygulamacının ne yapmak istediğine karar vermesi 
   gerekir. Yani uygulamacının çözmek istediği problem nedir? Bir sınıflandırma 
   problemi midir? Lojistik olmayan regresyon problemi midir? Şekil tanıma problemi 
   midir? Doğal dili anlama problemi midir? gibi...

2) Kestirimle İlgili Olacak Özellerin (Sütunların) Belirlenmesi: Tespit edilen 
   problemin kestirimi için hangi bilgilere 
   gereksinim duyulmaktadır? Kestirim ile ilgili olabilecek özellikler nelerdir? 
   Örneğin bir eczanenin cirosunu tahmin etmek isteyelim. Burada ilk gelecek 
   özellikler şunlar olabilir:

    - Eczanenin konumu
    - Ecnanenin önünden geçen günlük insan sayısı
    - Eczanenin destek ürünleri satıp satmadığı
    - Eczanenin kozmetik ürünler satıp satmadığı
    - Eczanenin anlaşmalı olduğu kurumlar
    - Eczanenin büyüklüğü
    - Eczanenin yakınındaki, hastenelerin ve sağlık ocaklarının sayısı
    - Eczacının tanıdığı doktor sayısı

3) Eğitim İçin Verilerin Toplanması: Eğitim için verilerin toplanması en zor 
   süreçlerden biridir. Veri toplama için şu yöntemler söz konusu olabilir:

    - Anketler (Surveys)
    - Daha önce elde edilmiş veriler
    - Çeşitli kurumlar tarafından zaten elde edilmiş olan veriler
    - Sensörler yoluyla elde edilen veriler
    - Sosyal ağlardan elde edilen veriler
    - Birtakım doğal süreç içerisinde oluşan veriler (Örneğin her müşteri için 
      bir fiş kesildiğine göre bunlar kullanılabilir)

4) Verilerin Kullanıma Hazır Hale Getirilmesi: Veriler toplandıktan sonra bunların 
   üzerinde bazı ön işlemlerin yapılması gerekmektedir. Örneğin gereksiz sütunlar 
   atılmalıdır. Eksik veriler varsa bunlar bir biçimde ele alınmalıdır. Kategorik 
   veriler sayısallaştırılmalıdır. Text ve görüntü verileri kullanıma hazır hale 
   getirilmelidir. Özellik "ölçeklemeleri (feature scaling)" ve "özellik mühendisliği 
   (feature engineering)" işlemleri yapılmalıdır. 

5) Yapay Sinir Ağı Modelinin Oluşturulması: Probleme uygun bir yapay sinir ağı 
   modeli oluşturulmalıdır. 

6) Modelin Eğitilmesi ve Test Edilmesi: Oluşturulan model eldeki veri kümesiyle 
   eğitilmeli ve test edilmelidir. 

7) Kestirim İşleminin Yapılması: Nihayet eğtilmiş model artık kestirim amacıyla 
   kullanılmalıdır. 
                                                                   
---------------------------------------------------------------------------------    
"""


# Kerasta Callback Mekanizması

"""
---------------------------------------------------------------------------------    
Callback sözcüğü programlamada belli bir olay devam ederken programcının verdiği 
bir fonksiyonun (genel olarak callable bir nesnenin) çağrılmasına" ilişkin mekanizmayı 
anlatmak için kullanılmaktadır. Keras'ın da bir callback mekanizması vardır. Bu 
sayede biz çeşitli olaylar devam ederken bu olayları gerçekleştiren metotların bizim 
callable nesnelerimizi çağırmasını sağlayabiliriz. Böylece birtakım işlemler devam 
ederken arka planda o işlemleri programlama yoluyla izleyebilir duruma göre gerekli 
işlemleri yapabiliriz. 

Sequential sınıfının fit, evalaute ve predict metotları "callbacks" isimli bir 
parametre almaktadır. İşte biz bu parametreye callback fonksiyonlarımızı ve sınıf 
nesnelerimizi verebiliriz. Bu metotlar da ilgili olaylar sırasında bizim verdiğimiz 
bu callable nesneleri çağırır. 

Aslında Keras'ta hazır bazı callback sınıflar zaten vardır. Dolayısıyla her ne 
kadar programcı kendi callback sınıflarını yazabilirse de aslında buna fazlaca 
gereksinim duyulmamaktadır. Keras'ın sağladığı hazır callback sınıfları genellikle 
gereksinimi karşılamaktadır. Keras'ın hazır callback sınıfları tensorflow.keras.callbacks 
modülü içerisinde bulunmaktadır.

---------------------------------------------------------------------------------    

---------------------------------------------------------------------------------
# History

En sık kullanılan callback sınıfı History isimli callback sınıftır. Aslında programcı 
bu callback sınıfını genellikle kendisi kullanmaz. Sequential sınıfının fit metodu 
zaten bu sınıf türünden bir nesneye geri dönmektedir. Örneğin:

hist = model.fit(....)

History callback sınıfı aslında işlemler sırasında devreye girmek için değil 
(bu da sağlanabilir) epcoh'lar sırasındaki değerlerin kaydedilmesi için kullanılmaktadır. 
Yani programcı fit işlemi bittikten sonra bu callback nesnenin içerisinden fit 
işlemi sırasında elde edilen epoch değerlerini alabilmektedir. Dolayısıyla fit 
metodunun bize verdiği History nesnesi eğitim sırasında her epoch'tan sonra elde 
edilen loss ve metrik değerleri barındırmaktadır. Anımsanacağı gibi fit metodu 
zaten eğitim sırasında her epoch'tan sonra birtakım değerleri ekrana yazıyordu. 
İşte fit metodunun geri döndürdüğü bu history nesnesi aslında bu metodun ekrana 
yazdığı bilgileri barındırmaktadır. History nesnesinin epoch özniteliği uygulanan 
epoch numaralarını bize verir. 

Ancak nesnenin en önemli elemanı history isimli özniteliğidir. Nesnenin history 
isimli özniteliği bir sözlük türündendir. Sözlüğün anahtarları yazısal olarak loss 
ve metrik değer isimlerini barındırır. Bunlara karşı gelen değerler ise her epoch'taki 
ilgili değerleri belirten list türünden nesnelerdir. History nesnesinin history 
sözlüğü her zaman "loss" ve "val_loss" anahtarlarını barındırır. Bunun dışında 
bizim belirlediğimiz metriklere ilişkin eğitim ve sınama sonuçlarını da barındırmaktadır. 

Örneğin biz metrik olarak "binary_accuracy" girmiş olalım. history sözlüğü bu durumda 
"binary_accuracy" ve "val_binary_accuracy" isimli iki anahtara da sahip olacaktır. 
Burada "val_xxx" biçiminde "val" ile başlayan anahtarlar sınama verisinden elde 
edilen değerleri "val" ile başlamayan anahtarlar ise eğitim veri kümesinden elde 
edilen değerleri belirtir. "loss" değeri ve diğer metrik değerler epoch'un tamamı 
için elde edilen değerlerdir. Her epoch sonucunda bu değerler sıfırlanmaktadır. 
(Yani bu değerler kümülatif bir ortalama değil, her epoch'taki ortalamalara ilişkindir.)

---------------------------------------------------------------------------------    
Epoch'lar sonrasında History nesnesi yoluyla elde edilen bilgilerin grafiği çizdirilebilir. 
Uygulamacılar eğitimin gidişatı hakkında fikir edinebilmek için genellikle epoch 
grafiğini çizdirirler. Epoch sayıları arttıkça başarının artacağını söyleyemeyiz. 
Hatta tam tersine belli bir epoch'tan sonra "overfitting" denilen olgu kendini 
gösterebilmekte model git gide yanlış şeyleri öğrenir duruma gelebilmektedir. İşte 
uygulamacı gelellikle "loss", "val_loss", "binary_accuracy", "val_binary_accuracy" 
gibi grafikleri epoch sayılarına göre çizerek bunların uyumlu gidip gitmediğine 
bakabilir. 

Eğitimdeki verilerle sınama verilerinin birbirbirlerinden kopması genel olarak 
kötü bir gidişata işaret etmektedir. Uygulamacı bu grafiklere bakarak uygulaması 
gereken epoch sayısına karar verebilir. 


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Binary Accuracy'])
plt.show()

Aşağıda "diabetes" örneği için fit metodunun geri döndürdüğü History callback nesnesi 
kullanılarak epoch grafikleri çizdirilmiştir. 

---------------------------------------------------------------------------------    

import pandas as pd

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\2- KerasIntroduction\diabetes.csv')

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = si.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

dataset = df.to_numpy()

dataset_x = dataset[:, :-1]
dataset_y = dataset[:, -1]

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=300, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Binary Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

--------------------------------------------------------------------------------- 
Aslında History callback nesnesi fit metodunun callbacks parametresi yoluyla da 
elde edilebilir. Örneğin:

from tensorflow.keras.callbacks import History
hist = History()

model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=300, 
          validation_split=0.2, callbacks=[hist])

Tabii buna hiç gerek yoktur. Zaten bu History callback nesnesi fit metodu tarafından 
metodun içerisinde oluşturulup geri dönüş değeri yoluyla bize verilmektedir.

---------------------------------------------------------------------------------
CSVLogger isimli callback sınıfı epoch işlemlerinden elde edilen değerleri bir 
CSV dosyasının içerisine yazmaktadır. CSVLogger nesnesi yaratılırken __init__ 
metodunda CSV dosyasının yol ifadesi verilir. Eğitim bittiğinde bu dosaynın içi 
doldurulmuş olacaktır. Örneğin:

from tensorflow.keras.callbacks import CSVLogger

hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100, 
        validation_split=0.2, callbacks=[CSVLogger('diabtes-epoch.csv')])

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Daha önce de belirttiğimiz gibi tüm callback sınıfları tensorflow.keras.callbacks 
modülündeki Callback isimli sınıftan türetilmiştir. Biz de bu sınıftan türetme 
yaparak kendi callback sınıflarımızı yazabiliriz. fit, evaluate ve predict metotları
callbacks parametresine girilen callback nesnelerinin çeşitli metotlarını çeşitli 
olaylar sırasında çağırmaktadır. 

Programcı da kendi callback sınıflarını yazarken aslında bu taban Callback sınıfındaki 
metotları override eder. Örneğin her epoch bittiğinde Callback sınıfının on_epoch_end 
isimli metodu çağrılmaktadır. Callback sınıfının bu metodunun içi boştur. Ancak 
biz türemiş sınıfta bu metodu overide edersek (yani aynı isimle yeniden yazarsak) 
bizim override ettiğimiz metot devreye girecektir. on_epoch_end metodunun 
parametrik yapısı şöyle olmalıdır:

    
def on_epoch_end(epoch, logs):
    pass

Buradaki birinci parametre epoch numarasını (yani kaçıncı epoch olduğunu) ikinci 
parametre ise epoch sonrasındaki eğitim ve sınama işlemlerinden elde edilen "loss", 
metrik değerleri veren bir sözlük biçimindedir. Bu sözlüğün anahtarları ilgili 
değerleri belirten yazılardan değerleri o epoch'a ilişkin onların değerlerinden 
oluşmaktadır. 

---------------------------------------------------------------------------------
Örneğin her epoch sonrasında biz bir fonksiyonumuzun çağrılmasını isteyelim. Bu 
fonksiyon içerisinde de "loss" değerini ve "val_loss" ekrana yazdırmak isteyelim. 
Bu işlemi şöyle yapabiliriz:

class MyCallback(Callback):
    
    def on_epoch_end(self, epoch, logs):
        loss = logs['loss']
        val_loss = logs['val_loss']
        print(f'epoch: {epoch}, loss: {loss}, val_loss: {val_loss}')
 
mycallback = MyCallback()

hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=300, 
    validation_split=0.2, callbacks=[mycallback], verbose=0)


Burada fit metodunun verbose parametresine 0 değerini geçtik. fit metodu (evaluate 
ve predict metotlarında da aynı durum söz konusu) çalışırken zaten "loss" ve 
"metrik değerleri" ekrana yazdırmaktadır. verbose parametre için 0 girildiğinde 
artık fit metodu ekrana bir şey yazmamaktadır. Dolayısıyşa yalnızca bizim callback 
fonksiyonda ekrana yazdırdıklarımız ekranda görünecektir. 

---------------------------------------------------------------------------------
fit, evaluate ve predict tarafından çağrılan Callback sınıfının metotlarının en 
önemli olanları şunlardır:

on_epoch_begin
on_epoch_end
on_batch_begin
on_batch_end
on_train_begin
on_train_end

Tabii bir sınıf söz konusu olduğuna göre bu metotların birinci parametreleri self 
olacaktır. Bu metotların parametreleri aşağıdaki gibidir:

Metot                    Parametreler

on_epoch_begin	            self, epoch ve logs
on_epoch_end	            self, epoch ve logs
on_batch_begin	            self, batch ve logs
on_batch_end	            self, batch ve logs
on_train_begin 	            self, logs
on_train_end	            self, logs

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# LambdaCallback 

LambdaCallback isimli callback sınıfı bizden __init__ metodu yoluyla çeşitli fonksiyonlar 
alır ve bu fonksiyonları belli noktalarda çağırır. Metottaki parametreler belli 
olaylar gerçekleştiğinde çağrılacak fonksiyonları belirtmektedir. Parametrelerin 
anlamları şöyledir:

on_train_begin: Eğitim başladığında çağrılacak fonksiyonu belirtir. 
on_train_end: Eğitim bittiğinde çağrılacak fonksiyonu belirtir. 

on_epoch_begin: Her epoch başladığında çağrılacak fonksiyonu belirtir.
on_epoch_end: Her epoch bittiğinde çağrılacak fonksiyonu belirtir.

on_batch_begin: Her batch işleminin başında çağrılacak fonksiyonu belirtir.
on_batch_end: Her batch işlemi bittiğinde çağrılacak fonksiyonu belirtir. 


Bu fonksiyonların parametreleri şöyle olmalıdır:

  Fonksiyon                 Parametreler

on_epoch_begin	            epoch ve logs
on_epoch_end	            epoch ve logs

on_batch_begin	            batch ve logs
on_batch_end	            batch ve logs

on_train_begin 	            logs
on_train_end	            logs

Burada epoch parametresi epoch numarasını, batch parametresi batch numarasını belirtir. 
loss parametreleri ise birer sözlük belirtmektedir. Bu sözlüğün içerisinde loss 
değeri gibi metrik değerler gibi önemli bilgiler vardır. epoch'lar için logs 
parametresi History nesnesindeki anahtarları içermektedir. Ancak batch'ler için 
logs parametresi "val" önekli değerleri içermeyecektir.(Çünkü validation işlemi
epochlar bittiğinde uygulanıyor, her batch sonu değil.)

Örneğin biz her epoch bittiğinde, her batch başladığında ve bittiğinde bir 
fonksiyonumuzun çağrılmasını isteyelim. Bunu şöyle gerçekleştirebiliriz:

def on_epoch_end_proc(epoch, logs):
   pass

def on_batch_begin_proc(batch, logs):
   pass

def on_batch_end_proc(batch, logs):
    pass

from tensorflow.keras.callbacks import LambdaCallback


lambda_callback = LambdaCallback( on_epoch_end=on_epoch_end_proc, 
                on_batch_begin=on_batch_begin_proc, on_batch_end=on_batch_end_proc)


hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=300, 
            validation_split=0.2, callbacks=[lambda_callback], verbose=0)

---------------------------------------------------------------------------------
"""


# Özellik Ölçeklemesi (Feature Scaling)

"""
---------------------------------------------------------------------------------
Bir nörona giren değerlerin "w" değerleriyle çarpılıp toplandığını (dot-product) 
ve sonuca bias değerinin toplanarak aktivasyon fonksiyonuna sokulduğunu biliyoruz. 
Örneğin modelde girdi katmanında x1, x2 ve x3 olmak üzere üç "sütun (feature)" 
bulunuyor olsun. Bu girdiler ilk saklı katmana sokulduğunda w1x1 + w2x2 + w3x3 + bias 
biçiminde bir toplam elde edilecektir. Bu toplam da aktivasyon fonksiyonuna sokulacaktır. 
İşte bu "dot product" işleminde x1, x2, x3 sütunlarının mertebeleri bir birlerinden 
çok farklıysa mertebesi yüksek olan sütunun "dot product" etkisi yüksek olacaktır. 
Bu durum da sanki o sütunun daha önemli olarak değerlendirilmesine yol açacaktır. 

Bu biçimdeki bir sinir ağı "geç yakınsar" ve gücü kestirim bakımından zayıflar. 
İşte bu nedenden dolayı işin başında sütunların (yani özelliklerin) mertebelerininin 
birbirlerine yaklaştırılması gerekmektedir. Bu işlemlere "özellik ölçeklemesi 
(feature scaling)" denilmektedir. Yapay sinir ağlarında sütunlar arasında mertebe 
farklılıkları varsa mutlaka özellik ölçeklemesi yapılmalıdır. 

Özellik ölçeklemesi makine öğrenmesinde başka konularda da gerekebilmektedir. Ancak 
bazı konularda ise gerekmemektedir.

Çeşitli özellik ölçeklemesi yöntemleri vardır. Veri kümelerinin dağılımına ve 
kullanılan yöntemlere göre değişik özellik ölçeklendirmeleri diğerlerine göre 
avantaj sağlayabilmektedir. En çok kullanılan iki özellik ölçeklendirmesi yöntemi 
" standart ölçekleme (standard scaling)" ve "minmax ölçeklemesi (minmax scaling)" dir. 

---------------------------------------------------------------------------------
Özellik ölçeklemesi konusunda aşağıdaki sorular sıkça sorulmaktadır:

Soru:  Yapay sinir ağlarında özellik ölçeklemesi her zaman gerekir mi? 
Cavap: Eğer sütunlar mertebe olarak birbirlerine zaten yakınsa özellik ölçeklemesi 
       yapılmayabilir. 

Soru:  Gerektiği halde özellik ölçeklemesini yapmazsak ne olur?
Cevap: Modelin kestirim gücü azalır. Yani performans düşer.

Soru:  Özellik ölçeklemesi gerekmediği halde özellik ölçeklemesi yaparsak bunun 
       bir zararı dokunur mu?
Cevap: Hayır dokunmaz.

Soru:  Kategorik sütunlara (0 ve 1'lerden oluşan ya da one hot encoding yapılmış) 
       özellik ölçeklemesi uygulayabilir miyiz?
Cevap: Bu sütunlara özellik ölçeklemesi uygulanmayabilir. Ancak uygulamanın bir 
       sakıncası olmaz. Genellikle veri bilimcisi tüm sütunlara özellik ölçeklemesi 
       uyguladığı için bunlara da uygulamaktadır. 

Özellik ölçeklemesi yalnızca x verilerine uygulanmalıdır, y verilerine özellik 
ölçeklemesi uygulamanın genel olarak faydası ve anlamı yoktur. Ayrıca bir nokta 
önemlidir: Biz ağımızı nasıl eğitmişsek öyle test ve kestirim yapmalıyız. Yani ağı 
eğitmeden önce özellik ölçeklemesi yapmışsak test işleminden önce test verilerini 
de kestirim işleminden önce kestirim verilerini de aynı biçimde ölçeklendirip 
işleme sokmalıyız.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# standart ölçekleme 

En çok kullanılan özellik ölçeklendirmesi yöntemlerinden biri "standart ölçekleme 
(standard scaling)" yöntemidir. Bu yöntemde sütunlar diğerlerden bağımsız olarak 
kendi aralarında standart normal dağılıma uydurulmaktadır. Bu işlem şöyle yapılmaktadır:

result = (x - mean(x) ) / std(x)

Tabii burada biz formülü temsili kod (pseudo code) olarak verdik. Burdaki "mean" 
sütunun ortalamasını "std" ise standart sapmasını belirtmektedir. Sütunu bu biçimde 
ölçeklendirdiğimizde değerler büyük ölçüde 0'ın etrafında toplanır. Standart 
ölçeklemenin değerleri standart normal dağılma uydurmaya çalıştığına dikkat ediniz. 

Standart ölçeklemeye "standardizasyon (standardization)" da denilmektedir. Veri 
bilimcileri genellikle standart ölçeklemeyi default ölçekleme olarak kullanmaktadır. 
Bir NumPy dizisindeki sütunları aşağıdaki gibi bir fonksiyonla standart ölçeklemeye 
sokabiliriz.

Standart ölçekleme yapan bir fonksiyonu aşağıdaki gibi yazabiliriz:

def standard_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    
    for col in range(dataset.shape[1]):
        scaled_dataset[:, col] = dataset[:, col] - np.mean(dataset[:, col]) 
                                 / np.std(dataset[:, col])
    
    return scaled_dataset


Tabii aslında NumPy'ın eksensel işlem yapma özelliğinden faydalanarak yukarıdaki 
işlemi aşağıdaki gibi tek satırla da yapabiliriz:

    
def standard_scaler(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

---------------------------------------------------------------------------------
Aslında scikit-learn kütüphanesinde sklearn.preprocessing modülü içerisinde zaten 
standart ölçekleme yapan StandardScaler isimli bir sınıf vardır. Bu sınıf diğer 
scikit-learn sınıfları gibi kullanılmaktadır. Yani önce StandardScaler sınıfı 
türünden bir nesne yaratılır. Sonra bu nesne ile fit ve transform metotları çağrılır. 
Tabii fit ve transform metotlarında aynı veri kümesi kullanılacaksa bu işlem tek 
hamlede fit_transform metoduyla yapılabilir. Örneğin:

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)
scaled_dataset = ss.tranform(dataset)


fit işlemi sütunların ortalamasını ve standart sapmasını nesnenin içerisinde saklamaktadır. 
transform işleminde bu bilgiler kullanılmaktadır. fit işleminden sonra nesnenin 
özniteliklerinden sütunlara ilişkin bu bilgiler elde edilebilir. Örneğin fit işleminden 
sonra sınıfın mean_ örnek özniteliğinden sütun ortalamaları, scale_ örnek özniteliğinden 
sütun standart sapmaları ve var_ örnek özniteliğinden sütunların varyansları elde 
edilebilir.

import numpy as np
dataset = np.array([[1, 2, 3], [2, 1, 4], [7, 3, 8], [8, 9, 2], [20, 12, 3]])
print(dataset)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(dataset)

print(f'{ss.mean_}, {ss.scale_}')
print()

scaled_dataset = ss.transform(dataset)
print(scaled_dataset)

---------------------------------------------------------------------------------
Sinir ağlarında özellik ölçeklemesi yapılırken şu noktaya dikkat edilmelidir: 
Özellik ölçeklemesi önce eğitim veri kümesinde gerçekleştirilir. Sonra eğitim veri 
kümesindeki sütunlardaki bilgiler kullanılarak test veri kümesi ve kestirim veri 
kümesi ölçeklendirilir. (Yani test veri kümesi ve kestirim veri kümesi kendi arasında 
ölçeklendirilmez. Eğitim veri kümesi referans alınarak ölçeklendirilir. Çünkü modelin 
test edilmesi ve kestirimi eğitim şartlarında yapılmalıdır.) Bu durumu kodla şöyle 
ifade edebiliriz:

ss = StandardScaler()

ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)

scaled_test_dataset_x = ss.transform(test_dataset_x)

scaled_predict_dataset_x = ss.transform(predict_dataset_x)


Örneğin "diabetes.csv" veri kümesi üzerinde standart ölçekleme yapmak isteyelim. 
Bunun için önce veri kümesini dataset_x ve dataset_y biçiminde sonra da "eğitim" 
ve "test" olmak üzere ikiye ayırırız. Ondan sonra eğitim veri kümesi üzerinde 
özellik ölçeklemesi yapıp eğitimi uygularız. Yukarıda da belirttiğimiz gibi eğitim 
veri kümesinden elde edilen ölçekleme bilgisinin test işlemi işlemi öncesinde test 
veri kümesine de, kestirim işlemi öncesinde kestirim veri kümesine de uygulanması 
gerekmektedir.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# min-max ölçekleme

Diğer çok kullanılan bir özellik ölçeklemesi yöntemi de "min-max" ölçeklemesi denilen 
yöntemdir. Bu ölçeklemede sütun değerleri [0, 1] arasında noktalı sayılarla temsil 
edilir. Min-max ölçeklemesi aşağıdaki temsili kodda olduğu gibi yapılmaktadır:

( a - min(a) ) / ( max(a) - min(a) )

Örneğin sütun değerleri aşağıdaki gibi olsun:

2
5
9
4
12

Burada min-max ölçeklemesi şöyle yapılmaktadır:

2 => (2 - 2) / 10 (12 - 2)
5 => (5 - 2) / 10
9 => (9 - 2) / 10
4 => (4 - 2) / 10
12 => (12 - 2) / 10

Min-max ölçeklemesinde en küçük değerin ölçeklenmiş değerinin 0 olduğuna, en büyük 
değerin ölçeklenmiş değerinin 1 olduğuna, diğer değerlerin ise 0 ile 1 arasında 
ölçeklendirildiğine dikkat ediniz. 

Min-max ölçeklemesi yapan bir fonksiyon şöyle yazılabilir:

import numpy as np

def minmax_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    for col in range(dataset.shape[1]):
        min_val, max_val = np.min(dataset[:, col]), np.max(dataset[:, col])
        scaled_dataset[:, col] = 0 if max_val - min_val == 0 else (dataset[:, col] - min_val) / (max_val - min_val)
        
    return scaled_dataset


Bir sütundaki tüm değerlerin aynı olduğunu düşünelim. Böyle bir sütunun veri kümesinde 
bulunmasının bir faydası olabilir mi? Tabii ki olmaz. Min-max ölçeklemesi yaparken 
sütundaki tüm değerler aynı ise sıfıra bölme gibi bir anomali oluşabilmektedir.
Yukarıdak kodda bu durum da dikkate alınmıştır. (Tabii aslında böyle bir sütun 
ön işleme aşamasında veri kümesinden zaten atılması gerekir.) Yukarıdaki fonksiyonu 
yine NumPy'ın eksensel işlem yapma yenetiğini kullanarak tek satırda da yazabiliriz:

def minmax_scaler(dataset):
    return (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))

---------------------------------------------------------------------------------
sckit-learn kütüphanesinde sklearn.preprocessing modülünde min-max ölçeklemesi 
yapan MinMaxScaler isimli bir sınıf da bulunmaktadır. Sınıfın kullanımı tamamen 
benzerleri gibidir. Örneğin:

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(dataset)
scaled_dataset = mms.transform(dataset)

Yine sınıfın fit ve tarnsform işlemini birlikte yapan fit_transform isimli metodu 
bulunmaktadır. 

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# maxabs ölçekleme

Diğer bir özellik ölçeklemesi de "maxabs" ölçeklemesi denilen ölçeklemedir. Maxabs 
ölçeklemesinde sütundaki değerler sütundaki değerlerin mutlak değerlerinin en 
büyüğüne bölünmektedir.  Böylece sütun değerleri [-1, 1] arasına ölçeklenmektedir. 
maxabs ölçeklemesi şöyle yapılmaktadır:

x / max(abs(x))

Burada sütundaki tüm değerler en büyük mutlak değere bölündüğüne göre ölçeklenmiş 
değerler -1 ile +1 arasında olacaktır.


import numpy as np

def maxabs_scaler(dataset):
    scaled_dataset = np.zeros(dataset.shape)
    
    for col in range(dataset.shape[1]):
        maxabs_val = np.max(np.abs(dataset[:, col]))
        scaled_dataset[:, col] = 0 if maxabs_val == 0 else  dataset[:, col] / maxabs_val
        
    return scaled_dataset

---------------------------------------------------------------------------------
maxabs ölçeklemesi yapan bir fonksiyon şöyle yazılabilir:

def maxabs_scaler(dataset):
        return dataset /  np.max(np.abs(dataset), axis=0)

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Biz yukarıda üç ölçeklendirmeyi tanıttık. Bunlar "standart ölçekleme", "minmax 
ölçeklemesi" ve "maxabs ölçeklemesi" idi. Aslında bunların dışında başka ölçeklemeler 
de kullanılabilmektedir. Bu konuda başka kaynaklara başvurabilirsiniz. Pekiyi 
elimizdeki veri kümesi için hangi ölçeklemenin daha iyi olduğuna nasıl karar 
verebiliriz? Aslında bu kararı vermenin çok pratik yolları yoktur. En iyi yöntem 
yine de "deneme yanılma yoluyla" kıyaslama yapmaktır. Fakat yine de ölçekleme 
türünü seçerken aşağıdaki durumlara dikkat edilmelidir:

- Sütunlarda aşırı uç değerlerin (outliers) bulunduğu durumda minmax ölçeklemesi 
ölçeklenmiş değerlerin birbirinden uzaklaşmasına yol açabilmektedir. Bu durumda 
bu ölçeklemenin performansı düşürebileceğine dikkat etmek gerekir. 

- Sütunlardaki değerler normal dağılıma benziyorsa (örneğin doğal birtakım olgulardan 
geliyorsa) standart ölçekleme diğerlerine göre daha iyi performans gösterebilmektedir. 

- Sütunlardaki değerler düzgün dağılmışsa ve aşırı uç değerler yoksa minmax ölçeklemesi 
tercih edilebilir. 

Ancak yine uygulamacılar veri kümeleri hakkında özel bir bilgiye sahip değilse ve 
deneme yanılma yöntemini kullanmak istemiyorlarsa standart ölçeklemeyi default 
ölçekleme olarak tercih etmektedir.  

---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
Peki özellik ölçeklemesi yaptığımız bir modeli nasıl saklayıp geri yükleyebiliriz? 
Yukarıdaki örneklerde biz özellik ölçeklemesini scikit-learn kullanarak yaptık. 
Sequential sınıfının save metodu modeli save ederken bu ölçeklemeler modelin bir 
parçası olmadığı için onları saklayamamaktadır. Bu nedenle bizim bu bilgileri ayrıca 
save etmemiz gerekmektedir. Ancak Keras'ta özellik ölçeklendirmeleri bir katman 
nesnesi olarak da bulundurulmuştur. Eğer özellik ölçeklemeleri bir katman ile 
yapılırsa bu durumda modeli zaten save ettiğimizde bu bilgiler de save edilmiş 
olacaktır. Biz burada scikit-learn ölçekleme bilgisinin nasıl saklanacağı ve geri 
yükleneceği üzerinde duracağız. 

---------------------------------------------------------------------------------
Programalamada bir sınıf nesnesinin diskteki bir dosyaya yazılmasına ve oradan 
geri yüklenmesine "nesnelerin seri hale getirilmesi (object serialization)" 
denilmektedir. scikit-learn içerisinde "object serialiazation" işlemine yönelik 
özel sınıflar yoktur. Ancak seri hale getirme işlemi Python'un standart 
kütüphanesindeki pickle modülü ile yapılabilmektedir. 

object serialization --->> Bir sınıfın bütün bilgilerini bir dosyaya yazıp daha
                         sonra geri okumaya denir.

--> Nesne serileştirme bir nesnenin bellekteki durumunu veya verilerini, onu 
saklamak, aktarmak veya başka bir programda yeniden oluşturmak için bir veri 
akışına dönüştürme işlemidir.

---------------------------------------------------------------------------------    
Örneğin scikit-learn ile standard ölçekleme yapmış olalım ve bu ölçekleme bilgisini 
Python'un standart pickle modülü ile bir dosyada saklamak isteyelim. Bu işlemi 
şöyle yapabiliriz:

    
import pickle

with open('diabetes-scaling.dat', 'wb') as f:
    pickle.dump(ss, f)    


Nesneyi dosyadan geri yükleme işlemi de şöyle yapılmaktadır:

    
with open('diabetes-scaling.dat', 'rb') as f:
    ss = pickle.load(f)    

---------------------------------------------------------------------------------
Aşağıdaki örnekte model "diabetes.h5" dosyası içerisinde, MinMaxScaler nesnesi 
de "diabetes-scaling.dat" dosyası içerisinde saklanmıştır ve geri yüklenmiştir. 


model.save('diabetes.h5')

import pickle

with open('diabetes-scaling.dat', 'wb') as f:
    pickle.dump(ss, f)    


# diabetes-scaling-load.py

import numpy as np

from tensorflow.keras.models import load_model
import pickle 

model = load_model('diabetes.h5')

with open('diabetes-scaling.dat', 'rb') as f:
    ss = pickle.load(f)
    
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
# Normalization

Keras'a sonradan eklenen Normalization isimli katman özellik ölçeklemesi yapabilmektedir. 
Normalization katmanı default olarak "standart ölçekleme" için düşünülmüştür. Ancak 
"minmax ölçeklemesi" için de kullanılabilir. Bu katmanı kullanabilmek için önce 
Normalization türünden bir nesnenin yaratılması gerekir. Normalization sınıfının 
__init__ metodunun parametrik yapısı şöyledir:

tf.keras.layers.Normalization(axis=-1, mean=None, variance=None, invert=False, **kwargs)

Burada mean sütunların ortalama değerlerini, variance ise sütunların varyans 
değerlerini almaktadır. Ancak programıcnın bu değerleri girmesine gerek yoktur. 
Normalization sınıfının adapt isimli metodu bizden bir veri kümesi alıp bu 
değerleri o kümeden elde edebilmektedir. Bu durumda standart ölçekleme için 
Normalization katmanı aşağıdaki gibi oluşturulabilir. 

from tensorflow.keras.layers import Normalization

norm_layer = Normalization()
norm_layer.adapt(training_dataset_x)


Tabii nu katmanı input katmanından sonra modele eklememiz gerekir. Örneğin:

    
model = Sequential(name='Diabetes')

model.add(Input((training_dataset_x.shape[1],)))

model.add(norm_layer)

model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

---------------------------------------------------------------------------------
import numpy as np

dataset = np.array([[1, 2, 3], [4, 5, 6], [3, 2, 7], [5, 9, 5]])
print(dataset)
print('--------')

from tensorflow.keras.layers import Normalization

norm_layer = Normalization()
norm_layer.adapt(dataset)

print(norm_layer.mean)
print('--------')
print(norm_layer.variance)    

---------------------------------------------------------------------------------    
!!!!

Tabii biz ölçeklemeyi bir katman biçiminde modele eklediğimizde artık test ve 
predict işlemlerinde ayrıca ölçekleme yapmamıza gerek kalmamaktadır. Bu katman 
zaten modele dahil olduğuna göre işlemler ölçeklendirilerek yapılacaktır.Yukarıda 
da belirttiğimiz gibi özellik ölçeklemesini Keras'ın bir katmanına yaptırdığımızda 
ayrıca ölçekleme bilgilerinin saklanmasına gerek olmadığına dikkat ediniz. Çünkü 
ölçekleme bilgileri artık modelin bir parçası durumundadır. 

!!!!
---------------------------------------------------------------------------------
Keras'ta minmax ölçeklemesi için hazır bir katman bulunmamaktadır. Ancak böyle 
bir katman nesnesi programcı tarafından da oluşturulabilir. 

Tabi biz henüz Tensorflow kütüphanesini incelemediğimiz için şu anda böyle bir 
örnek vermeyeceğiz.

Aslında hazır Normalization sınıfını biz minmax ölçeklemesi için de kullanabiliriz. 
Standart ölçeklemenin aşağıdaki gibi yapıldığını anımsayınız:

(X - mu) / sigma

Burada mu ve sigma ilgili sütunun ortalamasını ve standart sapmasını belirtmektedir. 
Minmax ölçeklemesinin ise şöyle yapıldığını anımsayınız:

(X - min ) / (max - min)

O halde biz aslında standart ölçeklemeyi minmax ölçeklemesi haline de getirebiliriz. 
Burada mu değerinin min, sigma değerinin ise max - min olması gerektiğine dikkat 
ediniz. Örneğin.

mins = np.min(training_dataset_x, axis=0)
maxmin_diffs = np.max(training_dataset_x, axis=0) - np.min(training_dataset_x, axis=0)
norm_layer = Normalization(mean=mins, variance=maxmin_diffs ** 2)

---------------------------------------------------------------------------------
Anımsanacağı gibi çıktının sınıfsal olmadığı modellere "regresyon modelleri" deniyordu. 
Biz kursumuzda bu durumu vurgulamak için "lojistik olmayan regresyon modelleri" 
terimini de kullanıyorduk. Regreson modellerinde sinir ağının çıktı katmanındaki 
aktivasyon fonksiyonu "linear" olmalıdır. Linear aktivasyon fonksiyonu bir şey 
yapmayan fonksiyondur. Başka bir deyişle f(x) değerinin x ile aynı olduğu fonksiyondur. 
(Zaten anımsanacağı gibi Dense katmanında activation parametresi girilmezse default 
durumda aktivasyon fonksiyonu "linear" alınmaktaydı.)

---------------------------------------------------------------------------------
"""        


# Auto-MPG

"""
---------------------------------------------------------------------------------
Auto-MPG otomobillerin bir galon benzinle kaç mil gidebildiklerininin (başka bir 
deyişle yakıt tüketiminin) tahmin edilmesi amacıyla oluşturulmuş bir veri kümesidir. 
Regresyon problemleri için sık kullanılan veri kümelerinden biridir. Veriler 80'li 
yılların başlarında toplanmıştır. O zamanki otomobil teknolojisi dikkate alınmalıdır. 
Veri kümesi aşağıdaki adresten indirilebilir:

https://archive.ics.uci.edu/dataset/9/auto+mpg

Veri kümesi bir zip dosyası olarak indirilmektedir. Buradaki "auto-mpg.data" 
dosyasını kullanabilirsiniz. Zip dosyasındaki "auto-mpg.names" dosyasında veri 
kümesi hakkında açıklamalar ve sütunların isimleri ve anlamları bulunmaktadır. 

Veri kümsindeki sütunlar SPACE karakterleriyle ve son sütun da TAB karakteriyle 
birbirlerinden ayrılmıştır. Sütunların anlamları şöyledir:

1. mpg:           continuous
2. cylinders:     multi-valued discrete
3. displacement:  continuous
4. horsepower:    continuous
5. weight:        continuous
6. acceleration:  continuous
7. model year:    multi-valued discrete
8. origin:        multi-valued discrete
9. car name:      string (unique for each instance)

Burada orijin kategorik bir sütundur. Buradaki değer 1 ise araba Amerika orijinli, 
2 ise Avrupa orijinli ve 3 ise Japon orijinlidir.

Veri kümesinin text dosyadaki görünümü aşağıdaki gibidir:

18.0   8   307.0      130.0      3504.      12.0   70  1	"chevrolet chevelle malibu"
15.0   8   350.0      165.0      3693.      11.5   70  1	"buick skylark 320"
18.0   8   318.0      150.0      3436.      11.0   70  1	"plymouth satellite"
16.0   8   304.0      150.0      3433.      12.0   70  1	"amc rebel sst"
17.0   8   302.0      140.0      3449.      10.5   70  1	"ford torino"
25.0   4   98.00      ?          2046.      19.0   71  1	"ford pinto"
19.0   6   232.0      100.0      2634.      13.0   71  1	"amc gremlin"
16.0   6   225.0      105.0      3439.      15.5   71  1	"plymouth satellite custom"
17.0   6   250.0      100.0      3329.      15.5   71  1	"chevrolet chevelle malib
....   


Veri kümesi incelendiğinde dördüncü sütunda (horsepower) '?' karakteri ile belirtilen 
eksik verilerin bulunduğu görülmektedir. Veri kümesindeki veriler az olmadığı için 
bu eksik verilerin bulunduğuğu satırlar tamamen atılabilir. Ya da daha önceden 
de yaptığımız gibi ortalama değerle imputation uygulanabilir. Ayrıca arabanın yakıt 
tüketimi arabanın markası ile ilişkili olsa da arabaların pek çok alt modelleri 
vardır. Bu nedenle son sütundaki araba isimlerinin kestirimde faydası dokunmayacaktır. 
Bu sütun da veri kümesinden atılabilir.

Veri kümesinin bir başlık kısmı içermediğine dikkat ediniz. read_csv default durumda 
veri kümesinde başlık kısmının olup olmadığına kendi algoritması ile karar vermektedir. 
Ancak başlık kısmının bu biçimde read_csv tarafından otomatik belirlenmesi sağlam 
bir yöntem değildir. Bu nedenle bu veri kümesi okunurken read_csv fonksiyonunda 
header parametresi None geçilmelidir. read_csv default olarak sütunlardaki ayıraçların 
',' karakteri olduğunu varsaymaktadır. Halbuki bu veri kümesinde sütun ayıraçları 
ASCII SPACE ve TAB karakterleridir. Bu nedenle dosyanın read_csv tarafından düzgün 
parse edilebilmesi için delimeter parametresine r'\s+' biçiminde "düzenli ifade 
(regular expression)" kalıbı girilmelidir. (Düzenli ifadeler "Python Uygulamaları" 
kursunda ele alınmaktadır.) read_csv fonksiyonu eğer dosyada başlık kısmı yoksa 
sütun isimlerini 0, 1, ... biçiminde nümerik almaktadır. 

Bu durumda yukarıdaki veri kümsinin okunması şöyle yapılabilir:

df = pd.read_csv('auto-mpg.data', delimiter=r'\s+', header=None)

Şimdi bizim araba markalarının bulunduğu son sütundan kurtulmamız gerekir. Bu 
işlemi şöyle yapabiliriz:

df = df.iloc[:, :-1]

Tabii bu işlemi DataFrame sınıfının drop metodu ile de yapabiliriz:

df.drop(8, axis=1, inplace=True)

Aslında bu sütun read_csv ile okuma sırasında usecols parametresi yardımıyla da 
atılabilir. 

---------------------------------------------------------------------------------
Şimdi de 3'üncü indeksli sütundaki eksik verileri temsil eden '?' bulunan satırlar 
üzerinde çalışalım. Yukarıda da belirttiğimiz gibi bu sütundaki eksik verilerin 
sayıları çok az olduğu için veri kümesinden atılabilirler. Bu işlem şöyle yapılabilir:

df = df[df.iloc[:, 3] != '?']

Ancak biz geçmiş konuları da kullanabilmek için bu eksik verileri sütun ortalamaları 
ile doldurmaya (imputation) çalışalım. Burada dikkat edilmesi gereken nokta DataFrame 
nesnesinin 3'üncü indeksli sütununun türünün nümerik olmamasıdır. Bu nedenle
öncelikle bu sütunun türünü nümerik hale getirmek gerekir. Tabii sütunda '?' 
karakterleri olduğuna göre önce bu karakterler yerine 0 gibi nümerik değerleri 
yerleştirmeliyiz:

df.iloc[df.loc[:, 3] == '?', 3] = 0

Tabii eğer ilgili sütunda zaten 0 değerleri varsa bu durumda 0 ile doldurmak yerine 
np.nan değeri ile dolduma yolunu tercih edebilirsiniz. Örneğin:

df.iloc[df.loc[:, 3] == '?', 3] = np.nan

Artık sütunun türünü nümerik hala getirebiliriz:

df[3] = df[3].astype('float64')

Aslında read_csv ile okuma sırasında da fonksiyonun na_values parametresi yardımıyla 
işin başında '?' karakterleri yerine fonksiyonun np.nan değerlerini yerleştirmesini 
de sağlayabiliriz.

Burada doğrudan indekslemede sütun isimlerinin kullanılması gerektiğine, sütun 
isimlerinin de sütun başlığı olmadığı için sayısal biçimde verildiğine dikkat ediniz. 
Artık 3'üncü indeksli sütun üzerinde imputation uygulayabiliriz:
    
---------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean', missing_values=0)
df[3] = si.fit_transform(df[[3]])

Henüz NumPy'a dönmeden önce 7'inci sütundaki kategorik verileri Pandas'ın get_dummies 
fonksiyonu ile "one hot encoding" biçimine dönüştürebiliriz:

df = pd.get_dummies(df, columns=[7], dtype='uint8')

Artık NumPy'a dönebiliriz:

dataset = df.to_numpy()

Şimdi de veri kümesini x ve y olarak ayrıştıracağız. Ancak y verilerinin son 
sütunda değil ilk sütunda olduğuna dikkat ediniz:

dataset_x = dataset[:, 1:]
dataset_y = dataset[:, 0]   

Bundan sonra veri kümesi eğitim ve test amasıyla train_test_slit fonksiyonu ile 
ayrıştrılabiliriz

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = 

train_test_split(dataset_x, dataset_y, test_size=0.2)      


Artık özellik ölçeklemesi yapabiliriz. Özellik ölçeklemesini scikit-learn kullanarak 
ya da yukarıda da bahsettiğimiz gibi Normalization isimli Keras katmanı kullanarak 
da yapabiliriz. Sütun dağılımlarına bakıldığında standart ölçekleme yerine minmax 
ölçeklemesinin daha iyi performans verebileceği izlenimi edinilmektedir. Ancak 
bu konuda deneme yanılma yöntemi uygulamak gerekir. Biz default standart ölçekleme 
uygulayalım:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)

scaled_training_dataset_x = ss.transform(training_dataset_x)

scaled_test_dataset_x = ss.transform(test_dataset_x)


Artık modelimizi oluşturabiliriz. Bunun için yine iki saklı katman kullanacağız. 
Saklı katmanlardaki aktivasyon fonksiyonlarını yine "relu" olarak alacağız. Ancak 
çıktı katmanındaki aktivasyonun "linear" olması gerektiğini anımsayınız:


model = Sequential(name='Auto-MPG')

model.add(Input((training_dataset_x.shape[1],)))

model.add(Dense(32, activation='relu', name='Hidden-1'))
model.add(Dense(32, activation='relu', name='Hidden-2'))

model.add(Dense(1, activation='linear', name='Output'))
model.summary()

Şimdi de modelimizi compile edip fit işlemi uygulayalım. Modelimiz için optimizasyon 
algoritması yine "rmsprop" seçilebilir. Regresyon problemleri için loss fonksiyonunun 
genellikle "mean_squared_error" biçiminde alınabileceğini belirtmiştik. Yine 
regresyon problemleri için "mean_absolute_error" metrik değeri kullanılabilir:

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, 
                 epochs=200, validation_split=0.2)


Modelimizi test veri kümesiyle test edebiliriz:

eval_result = model.evaluate(scaled_test_dataset_x, test_dataset_y, batch_size=32)    

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')


Şimdi kestirim yapmaya çalışalım. Kesitirilecek veriler üzerinde de one-hot encoding 
dönüştürmesinin ve özellik ölçeklemesinin yapılması gerektiğini anımsayınız. 
Kestirilecek verileri bir "predict.csv" isimli bir dosyada aşağıdaki gibi oluşturmuş 
olalım:

8,307.0,130.0,3504,12.0,70,1	
4,350.0,165.0,3693,11.5,77,2	
8,318.0,150.0,3436,11.0,74,3

Bu dosyayı okuduktan predict işlemi yapmadan önce sonra sırasıyla "one hot encoding" 
ve standart ölçeklemenin uygulanması gerekir:

predict_df = pd.read_csv('predict.csv', header=None)
predict_df = pd.get_dummies(predict_df, columns=[6])

predict_dataset_x = predict_df.to_numpy() 
scaled_predict_dataset_x = ss.transform(predict_dataset_x)

predict_result = model.predict(scaled_predict_dataset_x)


for val in predict_result[:, 0]:
    print(val)

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
One hot encoding işleminde Pandas'ın get_dummies fonksiyonunu kullanırken dikkat 
ediniz. Bu fonksiyon one hot encoding yapılacak sütundaki kategorileri kendisi 
belirlemektedir. (Yani bu fonksiyon one hot encoding yapılacak sütunda unique olan
değerlerden hareketle kategorileri belirlemektedir.) Eğer predict yapacağınız CSV 
dosyasındaki satırlar tüm kategorileri içermezse bu durum bir sorun yaratır. 
scikit-learn içerisindeki OneHotEncoder sınıfı bu tür durumlarda "categories" 
isimli parametreyle bizlere yardımcı olmaktadır. Maalesef get_dummies fonksiyonun 
böyle bir parametresi yoktur. 

OneHotEncoder sınfının __init__ metodunda categories isimli parametre ilgili 
sütundaki kategorileri belirtmek için düşünülmüştür. Ancak biz mevcut kategorilerden 
daha fazla kategori oluşturmak istiyorsak bu parametredeki listeye eklemeler 
yapabiliriz. categories parametresi iki boyutlu bir liste olarak girilmelidir. 
Çünkü birden fazla sütun one hot encoding işlemine sokulabilmektedir. Bu durumda 
bu iki boyutlu listenin her elemanı sırasıyla sütunlardaki kategorileri belirtir. 
Örneğin:

ohe = OneHotEncoder(sparse=False, categories=[[0, 1, 2], ['elma', 'armut', 'kayısı']])

Burada biz iki sütunlu bir veri tablosunun iki sütununu da one hot encoding yapmak 
istemekteyiz. Bu veri tablosunun ilk sütunundaki kategoriler 0, 1, 2 biçiminde, 
ikinci sütunundaki kategoriler 'elma', 'armut', 'kayısı' biçimindedir. Eğer bu 
sütunlarda daha az kategori varsa burada belirtilen sayıda ve sırada sütun oluşturulur. 

predict işlemi yapılırken uygulanan "one hot encoding" işlemindeki sütunların 
eğitimdeki "one hot encoding" sütunlarıyla uyuşması gerekir. Aslında kütüphanelerde 
one hot encoding işlemini yapan fonksiyonlar ve metotlar önce sütunlardaki unique 
elemanları belirleyip sonra onları sıraya dizip sütunları bu sırada oluşturmaktadır. 
Örneğin Pandas'daki get_dummies fonklsiyonu ve scikit-learn'deki OneHotEncoder 
sınıfı böyle davranmaktadır. Fakat yine de tam uyuşma için one hot encoding 
işlemlerini farklı sınıflarla yapmamaya çalışınız. Predict işlemindeki one hot 
encoding sütunların eğitimde kullanılan one hot encoding sütunlarıyla uyuşmasını 
sağlamak için eğitimde kullanılan sütunlardaki değerleri saklabilirsiniz. OneHotEncoder
sınıfının categories_ örnek özniteliği zaten bu kategorileri bize vermektedir. 
Tabii daha önce yaptığımız gibi pickle modülü ile bu OneHotEncoder nesnesini 
bütünsel olarak saklayıp predict aşamasında kullanabiliriz.

Bir veri kümesinde "one hot encoding" yapılacak birden fazla sütun varsa veri kümesini
önce "one hot encoding" yapılacak kısım ile yapılmayacak kısmı iki parçaya ayırıp 
one hot encoding işlemini tek hamlede birden fazla sütun için uygulayabilirsiniz. 
Anımsanacağı gibi scikit-learn içerisindeki OneHotEncoder sınıfı zaten tek 
hamlede birden fazla sütunu one hot encoding yapabiliyordu. 

---------------------------------------------------------------------------------
"""   


# Boston Housing Prices (BHP) --- lojistik olmayan regresyon problemi
"""
---------------------------------------------------------------------------------
Regresyon problemlerinde çok kullanılan veri kümelerinden biri de "Boston Housing Prices (BHP)" 
isimli veri kümesidir. Bu veri kümesi daha önce görümüş olduğumuz "Melbourne 
Housing Snapshot (MHS)" veri kümesine benzemektedir. Bu veri kümesinde evlerin 
çeşitli bilgileri sütunlar halinde kodlanmıştır. Amaç yine evin fiyatını tahmin 
etmektir. Veriler 1070 yılında toplanmıştır. Veri kümesi aşağıdaki bağlantıdan 
indirilebilir:

https://www.kaggle.com/datasets/vikrishnan/boston-house-prices

Buradan veri kümesi bir zip dosyası biçiminde indirilmektedir. Zip dosyası açıldığında 
"housing.csv" isimli dosya elde edilmektedir. Veri kümesi aşağıdaki görünümdedir:

    
0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00
0.02731   0.00   7.070  0  0.4690  6.4210  78.90  4.9671   2  242.0  17.80 396.90   9.14  21.60
0.02729   0.00   7.070  0  0.4690  7.1850  61.10  4.9671   2  242.0  17.80 392.83   4.03  34.70
0.03237   0.00   2.180  0  0.4580  6.9980  45.80  6.0622   3  222.0  18.70 394.63   2.94  33.40
0.06905   0.00   2.180  0  0.4580  7.1470  54.20  6.0622   3  222.0  18.70 396.90   5.33  36.20
0.02985   0.00   2.180  0  0.4580  6.4300  58.70  6.0622   3  222.0  18.70 394.12   5.21  28.70
0.08829  12.50   7.870  0  0.5240  6.0120  66.60  5.5605   5  311.0  15.20 395.60  12.43  22.90
0.14455  12.50   7.870  0  0.5240  6.1720  96.10  5.9505   5  311.0  15.20 396.90  19.15  27.10
............


Burada da görüldüğü gibi her ne kadar dosyasının uzantısı "csv" ise de aslında 
sütunlar virgüllerle değil, SPACE karakterleriyle ayrıştırılmıştır. Tüm sütunlarda 
zaten sayısal bilgiler olduğu için aslında dosya en kolay bir biçimde NumPy'ın 
loadtxt fonksiyonuyla okunabilir. Örneğin:

dataset = np.loadtxt('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\6- BostonHousingPrices\\housing.csv')

Ancak biz kursumuzda ilk aşamaları Pandas ile yaptığımızdan aynı süreçleri izlemek 
için okumayı da yine Pandas'ın read_csv fonksiyonuyla yapacağız. Tabii read_csv 
fonksiyonunda yine delimiter parametresi boşlukları belirten "düzenli ifade (regular
expression)" biçiminde olmalıdır. Dosyada bir başlık kısmının olmadığına da 
dikkat ediniz. Örneğin:

df = pd.read_csv('housing.csv', header=None, delimiter=r'\s+')

Veri kümesindeki sütunlar için İngilizce aşağıdaki açıklamalar yapılmıştır:

1. CRIM: per capita crime rate by town
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS: proportion of non-retail business acres per town
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. NOX: nitric oxides concentration (parts per 10 million)
6. RM: average number of rooms per dwelling
7. AGE: proportion of owner-occupied units built prior to 1940
8. DIS: weighted distances to ﬁve Boston employment centers
9. RAD: index of accessibility to radial highways
10. TAX: full-value property-tax rate per $10,000
11. PTRATIO: pupil-teacher ratio by town 
12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population
14. MEDV: Median value of owner-occupied homes in $1000s

Buradaki son sütun evin fiyatını 1000 dolar cinsinden belirtmektedir. Veri kümesinde 
eksik veri yoktur. Veri kümesinin 4'üncü sütununda kategorik bir bilgi bulunmaktadır. 
Ancak bu alanda yalnızca 0 ve 1 biçiminde iki değer vardır. İki değerli sütunlar 
için "one hot encoding" işlemine gerek olmadığını anımsayınız. Ancak 9'uncu sütunda 
ikiden daha fazla sınıf içeren kategorik bir bilgi bulunmaktadır. Dolayısıyla yine 
onu "one hot encoding" yapabiliriz. Sütunlar arasında önemli basamaksal farklılıklar 
göze çarpmaktadır. Yani veri kümesi üzerinde özellik ölçeklemesinin yapılması 
gerekmektedir. Veri kümesinin sütunlarında aşırı uç değerler (outliars) de 
bulunmamaktadır. Özellik ölçeklemesi için standart ölçekleme ya da min-max 
ölçeklemesi kullanılabilir. 

---------------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv('housing.csv', delimiter=r'\s+', header=None)

highway_class = df.iloc[:, 8].to_numpy()



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
ohe_highway = ohe.fit_transform(highway_class.reshape(-1, 1))

dataset_y = df.iloc[:, -1].to_numpy()

df.drop([8, 13], axis=1, inplace=True) # 8 ve dataset_y  kısmı drop

dataset_x = pd.concat([df, pd.DataFrame(ohe_highway)], axis=1).to_numpy()



from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = 
                    train_test_split(dataset_x, dataset_y, test_size=0.1)



from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential(name='Boston-Housing-Prices')
model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))
model.summary()

model.compile('rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, 
                 epochs=200, validation_split=0.2)



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Mean Absolute Error - Validation Mean Absolute Error Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['mae'])
plt.plot(hist.epoch, hist.history['val_mae'])
plt.legend(['Mean Absolute Error', 'Validation Mean Absolute Error'])
plt.show()


scaled_test_dataset_x = ss.transform(test_dataset_x)
eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')



predict_df = pd.read_csv('predict-boston-housing-prices.csv', delimiter=r'\s+', header=None)

highway_class = predict_df.iloc[:, 8].to_numpy()
ohe_highway = ohe.transform(highway_class.reshape(-1, 1))

predict_df.drop(8, axis=1, inplace=True)
predict_dataset_x = pd.concat([predict_df, pd.DataFrame(ohe_highway)], axis=1).to_numpy()


scaled_predict_dataset_x = ss.transform(predict_dataset_x )
predict_result = model.predict(scaled_predict_dataset_x)

for val in predict_result[:, 0]:
    print(val)
    
---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
Biz regresyon terimini daha çok "çıktının bir sınıf değil sayısal bir değer olduğu" 
modeller için kullanıyorduk. Bu paragrafta daha genel bir terim olarak kullanacağız. 
Regresyon çeşitli biçimlerde yani çeşitli yöntemlerle gerçekleştirilebilmektedir. 
Aslında yapay sinir ağları da regresyon için bir yöntem grubunu oluşturmaktadır. 

Regresyon en genel anlamda girdi ile çıktı arasında bir ilişki kurma sürecini 
belirtmektedir. Matematiksel olarak regresyon y = f(x) biçiminde bir f fonksiyonunun 
elde edilme süreci olarak da tanımlanabilir. Eğer biz böyle bir f fonksiyonu bulursak 
x değerlerini fonksiyonda yerine koyarak y değerini elde edebiliriz. Tabi y = f(x) 
fonksiyonunda x değişkeni aslında (x0, x1, x2, ..., xn) biçiminde birden fazla 
değişkeni de temsil ediyor olabilir. Bu durumda f fonksiyonu f((x0, x1, x2, ..., xn)) 
biçiminde çok değişkenli bir fonksiyon olacaktır. Benzer biçimde y = f(x) eşitliğinde 
f fonksiyonu birden fazla değer de veriyor olabilir. Yani buradaki y değeri 
(y0, y1, y2, ..., ym) biçiminde de olabilir. 

---------------------------------------------------------------------------------
 İstatistikte regresyon işlemleri tipik olarak aşağıdaki gibi sınıflandırılmaktadır:

- Eğer bağımsız değişken (x değişkeni) bir tane ise buna genellikle "basit regresyon 
(simple regression)" denilmektedir. 


- Eğer bağımsız değişken (x değişkeni) birden fazla ise buna da genellikle 
"çoklu regresyon (mulptiple regression)" denilmektedir. 


- Eğer girdiyle çıktı arasında doğrusal bir ilişki kurulmak isteniyorsa (yani regresyon 
işleminden doğrusal bir fonksiyon elde edilmek isteniyorsa) bu tür regresyonlara 
"doğrusal regresyon (linear regression)" denilmektedir. Doğrusal regresyon da 
bağımsız değişken bir tane ise "basit doğrusal regresyon (simple linear regression)", 
bağımsız değişken birden fazla ise "çoklu doğrusal regresyon (multiple linear 
regression)" biçiminde ikiye ayrılabilmektedir. 
   

- Bağımsız değişken ile bağımlı değişken arasında polinomsal ilişki kurulmaya 
çalışılabilir. (Yani regresyon sonucunda bir polinom elde edilmeye çalışılabilir). 
Buna da "polinomsal regresyon (polynomial regression)" denilmektedir. Bu da yine 
basit ya da çoklu olabilir. Aslında işin matematiksel tarafında polinomsal regresyon 
bir transformasyonla doğrusal regresyon haline dönüştürülebilmektedir. Dolayısıyla 
doğrusal regresyonla polinomsal regresyon arasında aslında işlem bakımından önemli 
bir fark yoktur.


- Girdiyle çıktı arasında doğrusal olmayan bir ilişki de kurulmak istenebilir. 
(Yani doğrusal olmayan bir fonksiyon da oluşturulmak istenebilir). Bu tür regresyonlara 
"doğrusal olmayan regresyon (nonlinear regressions)" denilmektedir. Yukarıda da 
belirttiğimiz gibi her ne kadar polinomlar doğrusal fonksiyonlar olmasa da bunlar 
transformasyonla doğrusal hale getirilebildikleri için doğrusal olmayan regresyon 
denildiğinde genel olarak polinomsal regresyonlar kastedilmemektedir. Örneğin 
logatirmik, üstel regresyonlar doğrusal olmayan regresyonlara örnektir. 


- Bir regresyonda çıktı da birden fazla olabilir. Genellikle (her zaman değil) 
bu tür regresyonlara "çok değişkenli (multivariate)" regresyonlar denilmektedir. 
Örneğin:

(y1, y2) = f((x1, x2, x3, x4, x5))

Regresyon terminolojisinde "çok değişkenli" sözcüğü bağımsız değişkenin birden 
fazla olmasını değil (buna "çoklu" denilmektedir) bağımlı değişkenin birden fazla 
olmasını anlatan bir terimdir. İngilizce bu bağlamda "çok değişkenli" terimi 
"multivariate" biçiminde ifade edilmektedir. 


- Eğer regresyonun çıktısı kategorik değerler ise yani f fonksiyonu kategorik bir 
değer üretiyorsa buna "lojistik regresyon (logictic regression)" ya da "logit 
regresyonu" denilmektedir. Lojistk regresyonda çıktı iki sınıftan oluşuyorsa 
(hasta-sağlıklı gibi, olumlu-olumsuz gibi, doğru-yanlış gibi) böyle lojistik 
regresyonlara "iki sınıflı lojistik regresyon (binary logistic regression)" denilmektedir. 
Eğer çıktı ikiden fazla sınıftan oluşuyorsa böyle lojistik regresyonlara da 
"çok sınıflı lojistik regresyon (multi-class/multinomial logistic regression)" 
denilmektedir. Tabii aslında makine öğrenmesinde ve sinir sinir ağlarında 
"lojistik regresyon" terimi yerine "sınıflandırma (classification)" terimi tercih 
edilmektedir. Bizim de genellikle (ancak her zaman değil) kategorik kestirim 
modellerine "lojistik regresyon modelleri" yerine "sınıflandırma problemleri" 
dediğimizi anımsayınız.
   

- Sınıflandırma problemlerinde bir de "etiket (label)" kavramı sıklıkla karşımıza 
çıkmaktadır. Etiket genellikle çok değişkenli (multivariate) sınıflandırma problemlerinde 
(yani çıktının birden fazla olduğu ve kategorik olduğu problemlerde) her çıktı 
için kullanılan bir terimdir. 

Örneğin biz bir sinir ağından üç bilgi elde etmek isteyebiliriz: "kişinin hasta olup 
olmadığı", "kişinin obez olup olmadığı", "kişinin mutlu olup olmadığı". Burada 
üç tane etiket vardır. Sınıf kavramının belli bir etiketteki kategorileri belirtmek 
için kullanıldığına dikkat ediniz. Etiketlerin sayısına göre lojistik regresyon 
modelleri (yani "multivariate lojistik regresyon" modelleri) genellikle aşağıdaki 
gibi sınıflandırılmaktadır:

* Tek Etiketli İki Sınıflı Sınıflandırma (Single Label Binary Classification) Modelleri: 

Bu modellerde çıktı yani etiket bir tanedir. Etiket de iki sınıftan oluşmaktadır. 
Örneğin bir tümörün iyi huylu mu kötü huylu mu olduğunu kestirimeye çalışan model 
tek etiketli iki sınıflı modeldir.


* Tek Etiketli Çok Sınıflı Sınıflandırma (Single Label Multiclass) Modelleri: 
    
Burada bir tane çıktı vardır. Ancak çıktı ikiden fazla sınıftan oluşmaktadır. 
Örneğin bir resmin "elma mı, armut mu, kayısı mı" olduğunu anlamaya çalışan 
sınıflandırma problemi tek etiketli çok sınıflı bir modeldir. 


* Çok Etiketli İki Sınıflı Sınıflandırma (Multilabel Binary Classification) Modelleri: 
    
Çok etiketli modeller denildiğinde zaten default olarak iki sınıflı çok etiketli 
modeller anlaşılmaktadır. Örneğin bir yazının içeriğine göre yazıyı tag'lamak 
istediğimizde her tag ayrı bir etikettir. O tag'ın olması ya da olmaması da iki 
sınıflı bir çıktı belirtmektedir. 


* Çok Etiketli Çok Sınıflı Sınıflandırma (Multilabel Multiclass / Multidimentional Classification) Modelleri: 

Bu tür modellere genellikle "çok boyutlu (multidimentional)" modeller denilmektedir. 
Yani çıktı birden fazladır. Her çıktıda ikiden fazla sınıfa ilişkin olabilmektedir. 
Bu modelleri çok etiketli sınıflandırma modellerinin genel biçimi olarak düşünebilirsiniz.
  
---------------------------------------------------------------------------------
"""



# iris (zambak) --- çok sınıflı sınıflandırma(lojistik regresyon) problemi

"""
---------------------------------------------------------------------------------
Şimdi de tek etiketli çok sınıflı bir sınıflandırma problemine örnek verelim. 
Örneğimizde "iris (zambak)" isimli bir veri kümesini kullanacağız. Bu veri kümesi 
bu tür uygulamalarda örnek veri kümesi olarak çok sık kullanılmaktadır. Veri 
kümesi aşağıdaki bağlantıdan indirilebilir:

https://www.kaggle.com/datasets/uciml/iris?resource=download

Yukarıdaki bağlantıdan Iris veri kümesi ibir zip dosyası biçiminde indirilmektedir. 
Bu zip dosyası açıldığında "Iris.csv" dosyası elde edilecektir.

Veri kümesi aşağıdaki görünümdedir:

Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
1,5.1,3.5,1.4,0.2,Iris-setosa
2,4.9,3.0,1.4,0.2,Iris-setosa
3,4.7,3.2,1.3,0.2,Iris-setosa
4,4.6,3.1,1.5,0.2,Iris-setosa
5,5.0,3.6,1.4,0.2,Iris-setosa
6,5.4,3.9,1.7,0.4,Iris-setosa
7,4.6,3.4,1.4,0.3,Iris-setosa
8,5.0,3.4,1.5,0.2,Iris-setosa
9,4.4,2.9,1.4,0.2,Iris-setosa
......

Veri kümesinde üç grup zambak vardır: "Iris-setosa", "Iris-versicolor" ve 
"Iris-virginica". x verileri ise çanak (sepal) yaprakların ve taç (petal) yaprakların 
genişlik ve yüksekliklerine ilişkin dört değerden oluşmaktadır. Veri kümesi 
içerisinde "Id" isimli ilk sütun sıra numarası belirtir. Dolayısıyla kestirim 
sürecinde bu sütunun bir faydası yoktur.
 
!!!
Çok sınıflı sınıflandırma (multiclass lojistik regresyon) problemlerinde çıktıların 
(yani y verilerinin) "one-hot encoding" işlemine sokulması gerekir. 
!!!

Çıktı sütunu one-hot encoding yapıldığında uygulamacının hangi sütunların hangi 
sınıfları belirttiğini biliyor olması gerekir. (Anımsanacağı gibi Pandas'ın 
get_dummies fonksiyonu aslında unique fonksiyonunu ile elde ettiği unique değerleri 
sort ettikten sonra "one-hot encoding" işlemi yapmaktadır. (Dolayısıyla aslında 
get_dummies fonksiyonu sütunları kategorik değerleri küçükten büyüğe sıraya dizerek 
oluşturmaktadır. Scikit-learn içerisindeki OneHotEncoder sınıfı zaten kendi 
içerisinde categories_ özniteliği le bu sütunların neler olduğunu bize vermektedir. 
Tabii aslında OneHotEncoder sınıfı da kendi içerisinde unique işlemini uygulamaktadır. 
NumPy'ın unique fonksiyonunun aynı zamanda sıraya dizmeyi de yaptığını anımsayınız. 
Yani aslında categories_ özniteliğindeki kategoriler de leksikografik olarak sıraya 
dizilmiş biçimdedir.)

Veri kümesini aşağıdaki gibi okuyabiliriz:

df = pd.read_csv('Iris.csv')

x verilerini aşağıdaki gibi ayrıştırabiliriz:

dataset_x = df.iloc[:, 1:-1].to_numpy(dtype='float32')

y verilerini aşağıdaki gibi onet hot encoding yaparak ayrıştırabiliriz:

ohe = OneHotEncoder(sparse= False)
dataset_y = ohe.fit_transform(df.iloc[:, -1].to_numpy().reshape(-1, 1))


Anımsanacağı gibi çok sınıflı sınıflandırma problemlerindeki loss fonksiyonu 
"categorical_crossentropy", çıktı katmanındaki aktivasyon fonksiyonu "softmax" 
olmalıdır. Metrik değer olarak "binary_accuracy" yerine "categorical_accuracy" 
kullanılmalıdır.(Keras metrik değer olarak "accuracy" girildiğinde zaten problemin 
türüne göre onu "binary_accuracy" ya da "categorical_accuracy" biçiminde ele 
alabilmektedir.) Veri kümesi yine özellik ölçeklemesine sokulmalıdır. Bunun için 
standart ölçekleme kullanılabilir. Sinir ağı modelini şöyle oluşturulabiliriz:

    
model = Sequential(name='Iris')

model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))

model.add(Dense(dataset_y.shape[1], activation='softmax', name='Output'))
model.summary()


Çok sınıflı modellerin çıktı katmanında sınıf sayısı kadar nöron olması gerektiğini 
belirtmiştik. Çıktı katmanında aktivasyon fonksiyonu olarak softmax alındığı için 
çıktı değerlerinin toplamı 1 olmak zorundadır. Bu durumda biz kestirim işlemi 
yaparken çıktıdaki en büyük değerli nöronu tespit etmemiz gerekir. 

Tabii aslında bizim en büyük çıktıya sahip olan nöronun çıktı değerinden ziyade 
onun çıktıdaki kaçıncı nöron olduğunu tespit etmemiz gerekmektedir. Bu işlem tipik 
olarak NumPy kütüphanesindeki argmax fonksiyonu ile yapılabilir. Pekiyi varsayalım 
ki ilki 0 olmak üzere 2 numaralı nöronun değeri en yüksek olmuş olsun. Bu 2 
numaralı nöron hangi sınıfı temsil etmektedir? 

İşte bu 2 numaralı nöron aslında eğitimdeki dataset_y sütununun one hot encoding 
sonucundaki 2 numaralı sütununu temsil eder. O halde bizim dataset_y değerlerini 
one-hot encoding yaparken hangi sütunun hangi sınıfa karşı geldiğini biliyor 
olmamız gerekir. Zaten OneHotEncoder sınıfının bu bilgiyi categories_ örnek 
özniteliğinde sakladığını anımsayınız.

---------------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\7- Iris\\Iris.csv')

dataset_x = df.iloc[:, 1:-1].to_numpy(dtype='float32')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse= False)
dataset_y = ohe.fit_transform(df.iloc[:, -1].to_numpy().reshape(-1, 1))

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential(name='Iris')

model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))

model.add(Dense(dataset_y.shape[1], activation='softmax', name='Output'))
model.summary()

model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Categorcal Accuracy - Validation Categorical Accuracy', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

scaled_test_dataset_x = ss.transform(test_dataset_x)
eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_dataset_x = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\7- Iris\\predict-iris.csv').to_numpy(dtype='float32')
scaled_predict_dataset_x = ss.transform(predict_dataset_x)

import numpy as np

predict_result = model.predict(scaled_predict_dataset_x)
predict_indexes = np.argmax(predict_result, axis=1)

for pi in predict_indexes:
    print(ohe.categories_[0][pi])


#predict_categories = ohe.categories_[0][predict_indexes]
#print(predict_categories)

---------------------------------------------------------------------------------
"""



"""
---------------------------------------------------------------------------------
Yapay zeka ve makine öğrenmesinin en önemli uygulama alanlarından biri de "doğal 
dil işleme (natuaral lanuage processing)" alanıdır. Doğal dil işleme denildiğinde 
Türkçe, İngilizce gibi konuşma dilleri üzerindeki her türlü işlemler kastedilmektedir. 
Bugün doğal dil işlemede artık ağırlıklı olarak makine öğrenmesi teknikleri 
kullanılmaktadır. Örneğin makine çevirisi (machine translation) süreci aslında 
ana konu olarak doğal dil işlemenin bir konusudur. Ancak bugün artık makine çevirileri 
artık neredeyse  tamamen makine öğrenmesi teknikleriyle  yapılmaktadır. Doğal dil 
işleme alanı ile ilişkili olan diğer bir alanda "metin madenciliği (text mining)" 
denilen alandır. Metin madenciliği metinler içerisinden faydalı bilgilerin çekilip 
alınması ve onlardan faydalanılması ile ilgili süreçleri belirtmektedir. Bugün 
veri madenciliğinde de yine veri bilimi ve makine öğrenmesi teknikleri yoğun 
olarak kullanılmaktadır. 

Metinler üzerinde makine öğrenmesi teknikleri uygulanırken metinlerin ön işlemlere 
sokularak sayısal biçime dönüştürülmesi gerekir. Çünkü makine öğrenmesi tekniklerinde 
yazılar üzerinde değil sayılar üzerinde işlemler yapılmaktadır. Bu nedenle makine 
öğrenmesinde yazılar üzerinde çalışılırken doğal dil işleme ve metin madenciliği 
alanlarında daha önceden elde edilmiş bilgiler ve deneyimler kullanılmaktadır. 
Örneğin bir film hakkında aşağıdaki gibi bir yazı olsun:

"Filmi pek beğenmedim. Oyuncular iyi oynayamamışlar. Filmde pek çok abartılı 
sahneler de vardı. Neticede filmin iyi mi kötü mü olduğu konusunda kafam karışık. 
Size tavsiyem filme gidip boşuna para harcamayın!"

Bu yazıyı sayısal hale dönüştürmeden önce yazı üzerinde bazı ön işlemlerin 
yapılması gerekebilmektedir. Tipik ön işlemler şunlardır:

- Yazıyı sözcüklere ayırma ve noktalama işaretlerini atma (tokenizing)
- Sözükleri küçük harfe ya da büyük harfe dönüştürmek (transformation)
- Kendi başına anlamı olmayan, edatlar gibi soru ekleri gibi sözcüklerin atılması 
(bunlara İngilizce "stop words") denilmektedir. 
- Sözcüklerin köklerini elde edilmesi ve köklerinin kullanılması (stemming)
- Bağlam içerisinde farklı sözcüklerin aynı sözcükle yer değiştirmesi (lemmatization)

Yukarıdaki işlemleri yapabilen çeşitli kütüphaneler de bulunmaktadır. Bunlardan 
Python'da en çok kullanılanlardan biri NLTK isimli kütüphanedir. 

Sözcükleri birbirinden bağımsız sayılar biçiminde ele alarak denetimli ya da 
denetimsiz modeller oluşturulabilmektedir. Ancak son 20 yılda yazılardaki 
sözcüklerin bir bağlam içerisinde ele alınabilmesine yönelik sinir ağları 
geliştirilmiştir. Bunun için sözcüklerin sırası dikkate alınır ve sinir ağına 
bir hafıza kazandırılır.

---------------------------------------------------------------------------------
"""


# Sentiment Analysis

"""
---------------------------------------------------------------------------------
# IMDB

Sınıflandırma problemlerinde üzerinde çalışılan problem gruplarından biri de 
"sentiment analysis" denilen gruptur. Bu grup problemlerde kişiler bir olgu hakkında 
kanılarını belirten yazılar yazarlar. Buradaki kanılar tipik olarak "olumlu", 
"olumsuz" biçiminde iki sınıflıdır. Ancak çok sınıflı kanılar da söz konusu 
olabilmektedir. Sentiment analysis için oluşturulmuş çeşitli örnek veri kümeleri 
vardır. Bunlardan en çok kullanılanlarından biri "IMDB (Internet Movie Database)" 
veri kümesidir. Bu veri kümesinde kişiler bir film hakkında yorum yazısı yazmışlardır. 
Bu yorum yazısı "positive" ya da "negative" olarak sınıflandırılmaktadır. Böylece 
birisinin yazdığı yazının olumlu ya da olumsuz yargı içerdiği otomatik olarak 
tespit edilebilmektedir. 

Bu problemde girdiler (yani dataset_x) tipik olarak yazılardan oluşmaktadır. Çıktı 
ise tipik olarak ikili bir çıktıdır. IMDB veri kümesini aşağıdaki bağlantıdan 
indirebilirsiniz:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Buradan veri kümesi zip dosyası olarak indirilir. Açıldığında "IMDB Dataset.csv" 
isimli CSV dosyası elde edilmektedir. Bu CSV dosaysında "review" ve "sentiment" 
isimli iki sütun vardır. "review" sütunu film hakkındaki yorum yazısını "sentiment" 
sütunu ise "positive" ya da "negative" yazısını içermektedir. Buradaki model 
"ikili sınflandırma" problemi biçimindedir. 

Bu tarz problemlerde girdiler birer yazı olduğu için işlemlere doğrudan sokulamazlar. 
Önce onların bir biçimde sayısal hale dönüştürülmeleri gerekir. Yazıların sayısal 
hale dönüştürülmesi için tipik olarak iki yöntem kullanılmaktadır:

1) Vektörizasyon (vectorization) yöntemi
2) Sözcük Gömme (Word Embedding) yöntemi


Her iki yöntemde de önce yazılar sözcüklere ayrılır ve gerekli görülen ön işlemlerden 
geçirilir. Böylece bir yazı bir sözcük grubu haline getirilir. Biz burada en basit 
yöntem olan "vektörizasyon" yöntemi üzerinde duracağız. Sözcük gömme yöntemi 
sonraki paragraflarda ele alınacaktır. 

---------------------------------------------------------------------------------
Vektörizasyon şöyle bir yöntemdir:

- Tüm yorumlardaki tüm sözcüklerin kümesine "kelime haznesi (vocabulary)" denilmektedir. 
Örneğin IMDB veri kümesinde tek olan tüm sözcüklerin sayısı 50000 ise kelime 
haznesi bu 50000 sözcükten oluşmaktadır. 

- Veri kümesindeki x verileri yorum sayısı kadar satırdan, sözcük haznesindeki 
sözcük sayısı kadar sütundan oluşan iki boyutlu bir matris biçiminde oluşturulur. 
Örneğin sözcük haznesindeki sözcük sayısı 50000 ise ve toplamda veri kümesinde 
10000 yorum varsa x veri kümesi 10000x50000 büyüklüğünde bir matris biçimindedir. 
Bir yorum bu matriste bir satır ile temsil edilmektedir. Yoruma ilişkin satırda 
eğer sözcük haznesindeki bir sözcük kullanılmışsa o sözcüğe ilişkin sütun 1, 
kullanılmamışsa 0 yapılmaktadır. Böylece yorum yazıları 0'lardan ve 1'lerden 
oluşmuş olan eşit uzunluklu sayı dizilerine dönüştürülmüş olur.                                                                    


Bu vektörizasyon yöntemi fazlaca bellek kullanma eğilimindedir. Veri yapıları 
dünyasında çok büyük kısmı 0 olan, az kısmı farklı değerde bulunan matrislere 
"sparse (seyrek)" matris denilmektedir. Buradaki vektörler seyrek durumda olacaktır. 
Eğer sözcük haznesi çok büyükse gerçekten de tüm girdilerin yukarıda belirtildiği 
gibi bir matris içerisinde toplanması zor ve hatta imkansız olabilmektedir. 
Çünkü fit metodu bizden training_dataset_x ve training_dataset_y yi bir bütün 
olarak istemektedir. 

Yukarıdaki gibi vektörizasyon işleminde sözcükler arasında sırasal bir ilişkinin 
ortadan kaldırıldığına dikkat ediniz. Böyle bir vektörizasyon sözcükleri bağlamı 
içerisinde değerlendirmede fayda sağlamayacaktır. Ayrıca yukarıdaki vektörizasyon
ikili (binary) biçimdedir. Ancak istenirse vektör ikili olmaktan çıkartılıp 
sözcüklerin frekanslarıyla da oluşturulabilir. Örneğin "film" sözcüğü yazı içerisinde 
10 kere geçmişse vektörde ona karşılık gelen eleman 1 yapılmak yerine 10 yapılabilir.

---------------------------------------------------------------------------------
Şimdi de IMDB örneğinde yukarıda açıkladığımız ikili vektörüzasyon işlemini 
programlama yoluyla yapalım. Önce veri kümesini okuyalım:

df = pd.read_csv('IMDB Dataset.csv')

Şimdi tüm yorumlardaki farklı olan tüm sözcüklerden bir sözcük haznesi (vocabulary) 
oluşturalım:


import re

vocab =  set()
for text in df['review']:
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    vocab.update(words)


Burada Python'daki "düzenli ifade (regular expression)" kütüphanesinden faydalanılmıştır. 

Şimdi de sözcük haznesindeki her bir sözcüğüe bir numara verelim. Sözcüğe göre 
arama yapılacağı için bir sözlük nesnesinin kullanılması uygun olacaktır. Bu 
işlem sözlük içlemi ile tek hamlede gereçekleştirilebilir:

vocab_dict = {word: index for index, word in enumerate(vocab)}

Aslında burada yapılan aşağıdakiile aynı şeydir:

vocab_dict = {} 

for index, word in enumerate(vocab):
    vocab_dict[word] = index


Şimdi artık x verileri oluşturalım. Bunun için önce içi sıfırlarla dolu bir matris 
oluşturalım. Bu matrisin satır sayısı len(df) kadar (yani yorum sayısı kadar) sütun 
sayısı ise sözcük haznesi kadar (yani len(vocab) kadar) olmalıdır:

dataset_x = np.zeros((len(df), len(vocab)), dtype='uint8')  

Şimdi yeniden tüm yorumları tek tek sözcüklere ayırıp onları sayısallaştırıp 
dataset_x matrisinin ilgili satırının ilgili sütunlarını 1 yapalım:

for row, text in enumerate(df['review']):
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    word_numbers = [vocab_dict[word] for word in words]
    dataset_x[row, word_numbers] = 1


y değerlerini de "positive" için 1, "negatif" için 0 biçiminde oluşturabiliriz:

dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8') 


Artık dataset_x ve dataset_y hazırlanmıştır. Bundan sonra ikili sınıflandırma 
problemi için daha önce yaptığımız sinir ağı işlemleri yapılabilir. Veri kümsini 
eğitim ve test biçiminde ikiye ayırabiliriz:

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

Yapay sinir ağımızı oluşturabiliriz. Girdi katmanında çok fazla nöron olduğu için 
katmanlardaki nöron sayılarını yükseltebiliriz.


model = Sequential(name='IMDB')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)

---------------------------------------------------------------------------------
Pekiyi biz binary vektör haline geitirilmiş yazıyı bu vektörden hareketle yeniden 
orjinal haline getirebilir miyiz? Hayır getiremeyiz. Çünkü biz burada binary vector 
oluşturduğumuz için sözcük sıklıklarını kaybetmiş durumdayız. Dahası bu vektörde 
sözcüklerin sırası da kaybedilmiştir. Ancak yine de bu vektördü anlamsız olsa da 
aşağıdaki gibi bir yazı haline getirebiliriz:

rev_vocab_dict = {index: word for word, index in vocab_dict.items()}

word_indices = np.argwhere(dataset_x[0] == 1).flatten()
words = [rev_vocab_dict[index] for index in word_indices]
text = ' '.join(words)
print(text)

NumPy'ın where ya da argwhere fonksiyonları belli koşulu sağlayan elemanların 
indekslerini bize verebilmektedir. Buradaki argwhere fonksiyonu bize iki boyutlu 
bir dizi vermektedir. Biz de onu flatten (ya da reshape ile) tek boyutlu dizi 
haline getirdik. Sonra liste içlemiyle bu indekslere karşı gelen sözcükleri bir 
liste biçiminde elde ettik. Sonra da bunların aralarına SPACE karakterleri koyarak 
join metodu ile bunları tek bir yazı biçiminde oluşturduk.

---------------------------------------------------------------------------------
Yukarıdaki gibi yazıların vektörizasyon işlemiyle binary bir vektöre dönüştürülmesi 
işleminin görünen dezavantajları şunlardır:

- Aynı sözcükten birden fazla kez yazı içerisinde kullanılmışsa bunun eğitimde bir 
anlamı kalmamaktadır. Oysa gerçek hayattaki yazılarda örneğin "mükemmel" gibi bir 
sözcük çokça tekrarlanıyorsa bu durum onun olumlu yorumlanma olasılığını artırmaktadır.

- Vektörizasyon işlemi bir bağlam oluşturamamaktadır. Şöyle ki: Biz bir yazıda 
"çok kötü" dersek buradaki "çok" aslında "kötüyü" nitelemektedir. Ancak bunu bizim 
ağımız anlayamaz. Başka bir deyişle biz yorumdaki sözcüklerin sırasını değiştirsek 
de elde ettiğimiz vektör değişmeyecektir. 

- Vektörizasyon işlemi çok yer kaplama potansiyelinde olan bir işlemdir. Bu durumda 
ağı parçalı olarak eğitmek zorunda kalabiliriz. Parçalı eğitimler ileride ele 
alınacaktır.

- Biz işlemden önce tüm sözcükleri küçük harfe dönüştürdük. Bazı özel karakterleri 
sözcüğün parçası olmaktan çıkardık. Halbuki bu gibi bazı küçük ayrıntılar yazının 
daha iyi anlamlandırılmasına katkı sağlayabilir. Tabi bu durumda vocabulary büyür 
bu da eğitimin zorlaşması anlamına gelir. 

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------

# CountVectorizer

1) Önce CountVectorizer sınıfı türünden bir nesne yaratılır. Nesne yaratılırken 
sınıfın __init__ metodunda bazı önemli belirlemeler yapılabilmektedir. Örneğin 
dtype parametresi elde edilecek vektörün elemanlarının türünü belirtmektedir. Bu 
parametreyi elde edilecek matrisin kaplayacğı yeri azaltmak için 'uint8' gibi 
küçük bir tür olarak geçmek istebilirsiniz. Default durumda bu dtype parametresi 
'float64' biçimindedir. 

Sınıf yine default durumda tüm sözcükleri küçük harfe dönüştürmektedir. Ancak 
metodun lowercase parametresi False geçilirse bu dönüştürme yapılmamaktadır. Metodun 
diğer önemli parametreleri de vardır. 

Örneğin metodun stop_words parametresi "stop word" denilen anlamsız sözcükleri 
atmak için kullanılabilir. Bu parametreye stop words'lerden oluşan bir liste ya 
da NumPy dizisi girilirse bu sözcükler sözcük haznesinden atılmaktadır. Başka bir 
deyişle yokmuş gibi ele alınmaktadır. 

Metodun binary parametresi default olarak False biçimdedir. Bu durumda bir yazı 
içerisinde aynı sözcükten birden fazla kez geçerse vektörün ilgili elemanı 1 değil 
o sözcüğün sayısı olacak biçimde set edilmektedir. Biz eğer yukarıdaki örneğimizde 
olduğu gibi binary bir vektör oluşturmak istiyorsak bu parametreyi True yapabiliriz. 
Metodun diğer parametreleri için scikit-learn dokümanlarına başvurabilirsiniz.

Örneğin:

cv = CountVectorizer(dtype='uint8', stop_words=['de', 'bir', 've', 'mu'], binary=True)


2) Bundan sonra scikit-learn kütüphanesinin diğer sınıflarında olduğu gibi fit 
ve trasform işlemleri yapılır. fit işleminde biz fit metoduna yazılardan oluşan 
dolaşılabilir bir nesne veriririz. fit metoudu tüm yazılardan bir "sözlük haznesi 
(vocabulary)" oluşturur. Biz de bu sözlük haznesini bir sözlük nesnesi biçiminde 
nesnenin vocabulary_ özniteliğinden elde edebiliriz. Bu vocabulary_ tıpkı bizim 
yukarıdaki örnekte yaptığımız gibi anahtarları sözcükler değerleri de sözcüklerin 
indeksinden oluşan bir sözlük biçimindedir. Örneğin:


texts = ["film güzeldi ve senaryo iyidi", "film berbattı, tam anlamıyla berbattı", 
            "seyretmeye değmez", "oyuncular güzel oynamışlar", 
            "senaryo berbattı, böyle senaryo olur mu?", "filme gidin de bir de siz görün"]

cv = CountVectorizer(dtype='uint8', stop_words=['de', 'bir', 've', 'mu'], binary=True)
cv.fit(texts)

fit işlemi sonrasında elde edilen vocabulary_ sözlüğü şöyledir:

{'film': 4, 'güzeldi': 9, 'senaryo': 14, 'iyidi': 10, 'berbattı': 1, 'tam': 17, 
 'anlamıyla': 0, 'seyretmeye': 15, 'değmez': 3, 'oyuncular': 13, 'güzel': 8, 
 'oynamışlar': 12, 'böyle': 2, 'olur': 11, 'filme': 5, 'gidin': 6, 'siz': 16, 
 'görün': 7}
    
fit medodunun yalnızca sözük haznesi oluşturduğuna dikkat ediniz. Asıl dönüştürmeyi 
transform metodu yapmaktadır. Ancak tranform bize vektörel hale getirilmiş olan 
yazıları "seyrek matris (sparse matrix)" biçiminde csr_matrix isimli bir sınıf 
nesnesi olarak vermektedir. Bu sınıfın todense metodu ile biz bu seyrek matrisi 
normal matrise dönüştürebiliriz. Örneğin:

dataset_x = cv.transform(dataset).todense()


Aslında fit metodu herhangi bir dolaşılabilir nesneyi parametre olarak kabul etmektedir. 
Örneğin yazılar satır satır bulunuyorsa biz doğrudan dosya nesnesini bu fit metoduna 
verebiliriz. Bu durumda !!!! tüm yazıları belleğe okumak zorunda kalmayız. !!!! 
Örneğin:

from sklearn.feature_extraction.text import CountVectorizer

f = open('text.csv')
cv = CountVectorizer()
cv.fit(f)

Artık bu CountVectorizer nesnesi predict işleminde de aynı biçimde kullanılabilir. 

---------------------------------------------------------------------------------
Yukarıda da belirttiğimiz gibi CountVectorizer sınıfının __init__ metodunun binary 
parametresi default olarak False durumdadır. Bu parametrenin False olması yazı 
içerisinde belli bir sözcük n defa geçtiğinde o sözcüğe ilişkin sütun elemanın n 
olacağı anlamına gelmektedir. Eğer bu parametre True yapılırsa bu durumda binary 
bir vector elde edilir. Pekiyi biz vektörizasyon yaparken "binary" mi yoksa 
frekanslı mı vektörizasyon yapmalıyız? Aslında frekanslı vektörizasyon yapmak 
toplamda daha iyidir. Ancak binaryx" bilgilerin tutulma biçiminden özel olarak bir 
kazanç sağlanmaya çalışılabilir.  

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
 Keras içerisinde tensorflow.keras.datasets modülünde IMDB veri kümesi de hazır 
biçimde bulunmaktadır. Diğer hazır veri kümelerinde olduğu gibi bu IMDB veri 
kümesi de modüldeki load_data fonksiyonu ile yüklenmektedir. Örneğin:

from tensorflow.keras.datasets import imdb

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = imdb.load_data()

Burada load_data fonksiyonunun num_words isimli parametresine eğer bir değer 
girilirse bu değer toplam sözcük haznesinin sayısını belirtmektedir.  Örneğin:

(training_dataset_x, training_dataset_y), (test_dataset_x, test_dataset_y) = 
                                                imdb.load_data(num_words=1000)

Burada artık yorumlar en sık kullanılan 1000 sözcükten hareketle oluşturulmaktadır. 
Yani bu durumda vektörizasyon sonrasında vektörlerin sütun uzunlukları 1000 olacaktır. 
Bu parametre girilmezse IMDB yorumlarındaki tüm sözükler dikkate alınmaktadır.

Bize load_data fonksiyonu x verileri olarak vektörizasyon sonucundaki vektörleri 
vermemektedir. Sözcüklerin indekslerine ilişkin vektörleri bir liste olarak 
vermektedir. (Bunun nedeni uygulamacının vektörizasyon yerine başka işlemler 
yapabilmesine olanak sağlamaktır.) load_data fonksiyonun verdiği index listeleri 
için şöyle bir ayrıntı da vardır: Bu fonksiyon bize listelerdeki sözcük indekslerini 
üç fazla vermektedir. Bu sözcük indekslerindeki 0, 1 ve 2 indeksleri özel anlam 
ifade etmektedir. Dolayısıyla aslında örneğin bize verilen 1234 numaralı indeks 
1231 numaralı indekstir. Bizim bu indekslerden 3 çıkartmamız gerekmektedir. 

imdb modülündeki get_word_index fonksiyonu bize sözcük haznesini bir sözlük olarak 
vermektedir. num_words ne olursa olsun bu sözlük her zaman tüm kelime haznesini 
içermektedir. Başka bir deyişle buradaki get_word_index fonksiyonu bizim 
kodlarımızdaki vocab_dict sözlüğünü vermektedir. Örneğin:

vocab_dict = imdb.get_word_index()


Bu durumda biz training_dataset_x ve test_dataset_x listelerini aşağıdaki gibi 
binary vector haline getiren bir fonksiyon yazabiliriz:

def vectorize(sequence, colsize):
    dataset_x = np.zeros((len(sequence), colsize), dtype='uint8')
    for index, vals in enumerate(sequence):
        dataset_x[index, vals] = 1
        
    return dataset_x

Burada vectorize fonksiyonu indekslerin bulunduğu liste listesini ve oluşturulacak 
matrisin sütun uzunluğunu parametre olarak almıştır. Fonksiyon vektörize edilmiş 
NumPy dizisi ile geri dönmektedir. Ancak biz bu fonksiyonu kullanırken colsize 
parametresine get_word_index ile verilen sözlüğün eleman sayısından 3 fazla olan 
değeri geçirmeliyiz. Çünkü bu indeks listelerinde 0, 1 ve 2 değerleri özel bazı 
amaçlarla kullanılmıştır. Dolayısıyla buradaki sözcük indeksleri hep 3 fazladır. 
Yapay sinir ağımızda bu indekslerin 3 fazla olmasının bir önemi yoktur. Ancak ters 
dönüşüm uygulanacaksa tüm indeks değerleriden 3 çıkartılmalıdır. O halde 
vektörizasonu şöyle yapabiliriz:

training_dataset_x = vectorize(training_dataset_x, len(vocab_dict) + 3)
test_dataset_x = vectorize(test_dataset_x, len(vocab_dict) + 3)

Artık her şey tamadır. Yukarıda yaptığımız işlemleri yapabiliriz. 

Kestirim işleminde de aynı duruma dikkat edilmesi gerekir. Biz eğitimi sözcük 
indekslerinin 3 fazla olduğu duruma göre yaptık. O halde kestirim işleminde de 
aynı şeyi yapmamız gerekir. Yani kestirim yapılacak yazıyı get_word_index sözlüğüne
sokup onun numarasını elde ettikten sonra ona 3 toplamalıyız. Bu biçimde liste 
listesi oluşturursak bunu yine yukarıda yazmış olduğumuz vectorize fonksiyonuna 
sokabiliriz. 

predict_df = pd.read_csv('predict-imdb.csv')

predict_list = []
for text in predict_df['review']:
    index_list = []
    words = re.findall('[A-Za-z0-9]+', text.lower())
    for word in words:
        index_list.append(vocab_dict[word] + 3)
    predict_list.append(index_list)
    
predict_dataset_x = vectorize(predict_list, len(vocab_dict) + 3)

predict_result = model.predict(predict_dataset_x)

---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
# Reuters veri kümesi (çok sınıflı sınıflandırma problemi)

Yazıların sınıflandırılması için çok kullanılan diğer bir veri kümesi de "Reuters" 
isimli veri kümesidir. Bu veri kümesi 80'lerin başlarında Reuters haber ajansının 
haber yazılarından oluşmaktadır. Bu haber yazıları birtakım konulara ilişkindir. 
Dolayısıyla bu veri kümesi "çok sınıflı sınıflandırma" problemleri için örnek amacıyla 
kullanılmaktadır. 

Haberler toplam 46 farklı konuya ilişkindir. Veri kümesinin orijinali "çok etiketli 
(multilabel)" biçimdedir. Yani veri kümesindeki bazı yazılara birden fazla etiket 
iliştirilmiştir. Ancak biz burada bir yazıya birden fazla etiket iliştirilmişse 
onun yalnızca ilk etiketini alacağız. Böylece veri kümesini "çok etiketli (multilabel)" 
olmaktan çıkartıp "çok sınıflı (multiclass)" biçimde kullanacağız

Reuters veri kümesinin orijinali ".SGM" uzantılı dosyalar biçiminde "SGML" formatındadır. 
Dolayısıyla bu verilerin kullanıma hazır hale getirilmesi biraz yorucudur. Ancak 
aşağıdaki bağlantıda Reuters veri kümesindeki her yazı bir dosya biçiminde 
kaydedilmiş biçimde sunulmaktadır:

https://www.kaggle.com/datasets/nltkdata/reuters

Buradaki dosya indirilip açıldığında aşağıdaki gibi bir dizin yapısı oluşacaktır:

training    <DIR>
test        <DIR>
cats.txt
stopwords

Ancak veri kümesini açtığınızda iç içe bazı dizinlerin olduğunu göreceksiniz. Bu 
dizinlerden yalnızca bir tanesini alıp diğerlerini atabilirsiniz. 

Buradaki "cats.txt" dosyası tüm yazıların kategorilerinin belirtildiği bir dosyadır. 
training dizininde ve test dizininide her bir yazı bir text dosya biçiminde 
oluşturulmuştur. Buradaki text dosyaları okumak için "latin-1" encoding'ini 
kullanmalısınız. Biz yukarıdaki dizin yapısını çalışma dizininde "ReutersData" 
isimli bir dizine çektik. Yani veri kümesinin dizin yapısı şu hale getirilmiştir:

ReutersData
    training    <DIR>
    test        <DIR>
    cats.txt
    stopwords

Burada veri kümesi "eğitim" ve "test" biçiminde zaten ikiye ayrılmış durumdadır. 
Dolayısıyla bizim veri kümesini ayrıca "eğitim" ve "test" biçiminde ayırmamıza 
gerek yoktur. Buradaki "cats.txt" dosyasının içeriği aşağıdaki gibidir:

test/14826 trade
test/14828 grain
test/14829 nat-gas crude
test/14832 rubber tin sugar corn rice grain trade
test/14833 palm-oil veg-oil
test/14839 ship
test/14840 rubber coffee lumber palm-oil veg-oil
...
training/5793 nat-gas
training/5796 crude
training/5797 money-supply
training/5798 money-supply
training/5800 grain
training/5803 gnp
training/5804 gnp
training/5805 gnp
training/5807 gnp
training/5808 acq
training/5810 trade
training/5811 money-fx
training/5812 carcass livestock
...

Reuters veri kümesinde ayrıca "stopwords" isimli bir dosya içersinde stop word'lerin 
listesi de verilmiştir. Bu sözcüklerin sözcük haznesinden çıkartılması (yani stop 
word'lerin atılması) daha iyi bir sonucun elde edilmesine yol açabilecektir. 

Biz burada vektörizasyon işlemini önce manuel bir biçimde yapıp sonra CountVectorizer 
sınıfını kullanacağız. 

---------------------------------------------------------------------------------
Önce "cats.txt" dosyasını açıp buradaki bilgilerden training_dict ve test_dict 
isimli iki sözlük nesnesi oluşturalım. Bu sözlük nesnelerinin anahtarları dosya 
isimleri değerleri ise o dosyadaki yazının sınıfını belirtiyor olsun:

training_dict = {}
test_dict = {}
cats = set()

with open('ReutersData/cats.txt') as f:
    for line in f:
        toklist = line.split()
        ttype, fname = toklist[0].split('/')
        if ttype == 'training':
            training_dict[fname] = toklist[1]
        else:
            if ttype == 'test':
                test_dict[fname] = toklist[1]
        cats.add(toklist[1])
    
vocab =  set()
training_texts = []
training_y = []
                
for fname in os.listdir('ReutersData/training'):
    with open('ReutersData/training/' + fname, encoding='latin-1') as f:
        text = f.read()
        training_texts.append(text)
        words = re.findall('[a-zA-Z0-9]+', text.lower())
        vocab.update(words)
        training_y.append(training_dict[fname])
        
test_texts = []
test_y = []
for fname in os.listdir('ReutersData/test'):
    with open('ReutersData/test/' + fname, encoding='latin-1') as f:
        text = f.read()
        test_texts.append(text)
        words = re.findall('[a-zA-Z0-9]+', text.lower())
        vocab.update(words)
        test_y.append(test_dict[fname])
        
Burada tüm sözcük haznesinin vocab isimli bir kümede tüm kategorilerin de cats 
isimli bir kümede toplandığına dikkat ediniz. Hazır dosyaları açmışken dosyalar 
içerisindeki yazıları da training_texts ve test_texts isimli listelerde topladık. 
Ayrıca her yazının kategorilerini de training_y ve test_y listelerinde topladığımıza 
dikkat ediniz. Artık sözcüklere numaralar verebiliriz:


vocab_dict = {word: index for index, word in enumerate(vocab)}


Şimdi manuel olarak binary vektörizasyon uygulayalım:

training_dataset_x = np.zeros((len(training_texts), len(vocab)), dtype='uint8')  
test_dataset_x = np.zeros((len(test_texts), len(vocab)), dtype='uint8')  

for row, text in enumerate(training_texts):
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    word_numbers = [vocab_dict[word] for word in words]
    training_dataset_x[row, word_numbers] = 1
    
for row, text in enumerate(test_texts):
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    word_numbers = [vocab_dict[word] for word in words]
    test_dataset_x[row, word_numbers] = 1


Problem çok sınıflı bir sınıflandırma problemidir. Bunun için y değerlerini
değerleri üzerinde one-hot encoding dönüştürmesi uygulayabiliriz:


ohe = OneHotEncoder(sparse_output=False, dtype='uint8')
ohe.fit(np.array(list(cats)).reshape(-1, 1))

training_dataset_y = ohe.transform(np.array(training_y).reshape(-1, 1))
test_dataset_y = ohe.transform(np.array(test_y).reshape(-1, 1))


Artık modelimizi kurup eğitebiliriz:

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(len(cats), activation='softmax', name='Output'))
model.summary()
            
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=10, 
                 validation_split=0.2)


Kestirim işlemi için eğitimdeki veri kümesine benzer bir veri kümesi oluşturulabilir. 
Biz örneğimizde kestirim için "PredictData" isimli bir dizin oluşturup o dizine 
yazılardan oluşan dosyalar yerleştirdik. O dosyaların da olması gereken etiketlerini 
dosya ismine ek yaptık. PredictData dizinindeki dosya isimleri şöyledir:

14829-nat-gas
14841-wheat
14849-interest
14854-ipi
14860-earn
14862-bop
14876-earn
21394-acq

Kestirim kodu şöyle oluşturulabilir:

word_numbers_list = []
fnames = []
for fname in os.listdir('PredictData'):
    with open('PredictData/' + fname, encoding='latin-1') as f:
        text = f.read()
        words = re.findall('[a-zA-Z0-9]+', text.lower())
        word_numbers = [vocab_dict[word] for word in words]
        word_numbers_list.append(word_numbers)
        fnames.append(fname)
    
predict_dataset_x = np.zeros((len(word_numbers_list), len(vocab)), dtype='uint8')
for row, word_numbers in enumerate(word_numbers_list):
    predict_dataset_x[row, word_numbers] = 1
    
predict_result = model.predict(predict_dataset_x)
predict_indexes = np.argmax(predict_result, axis=1)

for index, pi in enumerate(predict_indexes):
    print(f'{fnames[index]} => {ohe.categories_[0][pi]}')

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
34
Şimdi de Reuters örneğini CountVectorizer sınıfını kullanılarak gerçekleştirelim. 
Anımsanacağı gibi CountVectorizer sınıfı zaten vektörizasyon işlemini kendisi 
yapmaktaydı. O halde biz Reuters yazılarını bir listede topladıktan sonra CountVectorizer
sınıfı ile fit işlemini yapabiliriz. Orijinal Reuters veri kümesinde ayrıca 
"stopwords" dosyası içerisinde "stop word'ler" satır satır sözcükler biçiminde 
verilmiştir. Anımsanacağı gibi CountVectorizer sınıfında biz stop word'leri de 
ayrıca belirtebiliyorduk. Reuters veri kümesinde verilen stop word'ler aşağıdaki 
gibi bir Python listesi biçiminde elde edilebilir:

import pandas as pd

df_sw = pd.read_csv('ReutersData/stopwords', header=None)
sw = df_sw.iloc[:, 0].to_list()

CountVectorizer sınıfı önce yazıları sözcüklere ayırıp (tokenizing) sonra stop 
word'leri atmaktadır. Ancak sınıfın sözcüklere ayırmada default kullandığı düzenli 
ifade kalıbı tek tırnaklı sözcüklerdeki tırnaklardan da ayrıştırma yapmaktadır. 
(Ancak tırnaktan sonraki kısmı da atmaktadır.) Orjinal veri kümesinde verilen 
stop word'ler tek tırnaklı yazı içerdiği için burada bir uyumsuzluk durumu 
ortaya çıkmaktadır.

O halde biz ya sınıfın kullandığı sözcüklere ayırma düzenli ifadesini 
(token_pattern parametresi) tırnakları kapsayacak biçimde değiştirmeliyiz ya da 
bu tırnaklı stop word'lerdeki tırnakları silmeliyiz. Dosyanın orijinalini bozmamak 
için uyarıda sözü edilen sözcükleri de listeye ekleyerek problemi pratik bir 
biçimde çözebiliriz:

sw +=  ['ain', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 
        'isn', 'll', 'mon', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']

Ayrıca CountVectorizer sınıfının stop_words parametresine 'english' girilirse 
scikit-learn içerisindeki İngilizce için oluşturulmuş default stop word listesi 
kullanılmaktadır. Tabii veri kümesindeki orijinal listenin kullanılması daha 
uygun olacaktır.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------

# TextVectorization

Aslında vektörizasyon işlemi daha sonraları Keras'a eklenmiş olan TextVectorization 
isimli katman sınıfı yoluyla da yapılabilmektedir. Uygulamacı Input katmanından 
sonra bu katman nesnesini modele ekler daha sonra da diğer katmanları modele ekler. 
Böylece alınan girdiler önce TextVectorization katmanı yoluyla vektörel hale 
getirilip diğer katmanlara iletilir. Tabii bu durumda bizim ağa girdi olarak 
vektörleri değil yazıları vermemiz gerekir. Çünkü bu katmanın kendisi zaten 
yazıları vektörel hale getirmektedir. Örneğin:

tv = TextVectorization(...)
...
model = Sequential(name='IMDB')

model.add(Input((1, )))
model.add(tv)
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

Burada ağın girdi katmanında tek sütunlu bir veri kümesi olduğuna dikkat ediniz. 
Çünkü biz ağa artık yazıları girdi olarak vereceğiz. Bu yazılar TextVectorization 
katmanına sokulacak ancak bu katmandan sözcük hanzesi kadar çıktı elde edilecektir. 
Bu çıktılarda sonraki Dense katmana verilmiştir.

TextVectorization sınıfı diğer katman nesnelerinde olduğu gibi tensorflow.keras.layers 
modülü içerisinde bulunmaktadır. Sınıfın __init__ metodunun parametrik yapısı 
şöyledir:

tf.keras.layers.TextVectorization(
    max_tokens=None,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int',
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding='utf-8',
    name=None,
    **kwargs
)

Görüldüğü gibi bu parametrelerin hepsi default değer almıştır. 

max_tokens parametresi --> En fazla yinelenen belli sayıda sözcüğün vektörel hale 
                        getirilmesi için kullanılmaktadır. Yani adeta sözcük haznesi 
                        burada belirtilen miktarda sözcük içeriyor gibi olmaktadır. 
                        Örneğin; 1000 tane farklı sözcük var en çok tekrarlanan 300
                        tanesini al denilebilir.

standardize parametresi --> Yazılardaki sözcüklerin elde edildikten sonra nasıl ön 
                        işleme sokulacağını belirtmektedir. Bu parametrenin default 
                        değerinin 'lower_and_strip_punctuation' biçiminde olduğuna 
                        dikkat ediniz. Bu durumda yazılardaki sözcükler küçük 
                        harflere dönüştürülecek ve sözcüklerin içerisindeki noktalama 
                        işaretleri atılacaktır. (Yani örneğin yazıdaki "Dikkat!" 
                        sözcüğü "dikkat" olarak ele alınacaktır.) Bu parametre 
                        için "çağrılabilir (callable)" bir nesne de girilebilmektedir. 
                        Bu fonksiyon eğitim sırasında çağrılıp buradan elde 
                        edilen yazılar vektörizasyon işlemine sokulmaktadır.

split parametresi --> Sözcüklerin nasıl birbirinden ayrılacağını belirtmektedir. 
                    Default durumda sözcükler boşluk karakterleriyle birbirinden 
                    ayrılmaktadır. 
                    
output_mode  --> Default değeri int biçimindedir. Bu durumda yazıdaki sözcükler 
                sözcük haznesindeki numaralar biçiminde verilecektir. Bu parametrenin 
                "count" biçiminde girilmesi uygundur. Eğer bu parametre "count" 
                biçiminde girilirse bu durumda yazı bizim istediğimiz gibi 
                frekanslardan oluşan vektör biçimine dönüştürülecektir. vocabulary 
                parametresi doğrudan sözcük haznesinin programcı tarafından metoda 
                verilmesini sağlamak için buludurulmuştur. Bu durumda adapt işleminde 
                sözcük haznesi adapt tarafından oluşturulmaz, burada verilen 
                sözcük haznesi kullamılır.


TextVectorization sınıfının get_vocabulary metodu adapt işleminin sonucunda 
oluşturulmuş olan sözcük haznesini bize vermektedir. 

set_vocabulary metodu ise sözcük haznesini set etmek için kullanılmaktadır.

TextVectorization nesnesi yaratıldıktan sonra sözcük haznesinin ve dönüştürmede 
kullanılacak sözcük nesnesinin oluşturulması için sınıfın adapt metodu çağrılmalıdır. 
Örneğin:

tv = TextVectorization(output_mode='count')
tv.adapt(texts)
 

---------------------------------------------------------------------------------
"""



# Keras'ta Parçalı Eğitim

"""
---------------------------------------------------------------------------------
Çok büyük verilerle eğitim, test hatta predict işlemi sorunlu bir konudur. Çünkü 
örneğin fit işleminde büyük miktarda  veri kümeleriyle eğitim ve test yapılırken 
bu veri kümeleri bir bütün olarak metotlara verilmektedir. Ancak büyük veri kümeleri 
eldeki belleğe sığmayabilir. (Her ne kadar 64 bit Windows ve Linux sistemlerinde 
prosesin sanal bellek alanı çok büyükse de bu teorik sanal bellek alanını 
kullanabilmek için swap alanlarının büyütülmesi gerekmektedir.) Örneğin IMDB ya 
da Reuters örneklerinde vektörizasyon işlemi sonucunda çok büyük matrisler 
oluşmaktadır. Gerçi bu matrislerin çok büyük kısmı 0'lardan oluştuğu için "seyrek 
(sparse)" durumdadır. Ancak fit, evaluate ve predict metotları seyrek matrislerle 
çalışmamaktadır. İşte bu nedenden dolayı Keras'ta modeller parçalı verilerle 
eğitilip test ve predict edilebilmektedir. Parçalı eğitim, test ve predict 
işlemlerinde eğitim, test ve predict sırasında her batch işleminde fit, evaluate 
ve predict metotları o anda bizden bir batch'lik verileri istemekte ve eğitim 
batch-batch verilere tedarik edilerek yapılabilmektedir. 

Parçalı eğitim ve test işlemi için fit, evaluate ve predict metotlarının birinci 
parametrelerinde x verileri yerine bir "üretici fonksiyon (generator)" ya da 
"Sequence sınıfından türetilmiş olan bir sınıf nesnesi" girilir. Biz burada önce 
üretici fonksiyon yoluyla sonra da Sequence sınıfından türetilmiş olan sınıf yoluyla 
parçalı işlemlerin nasıl yapılacağını göreceğiz. Eskiden Keras'ta normal fit, 
evaluate ve predict metotlarının ayrı fit_generator, evalute_generator ve 
predict_generator biçiminde parçalı eğitim için kullanılan biçimleri vardı. Ancak 
bu metotlar daha sonra kaldırıldı. Artık fit, evaluate ve predcit metotları hem 
bütünsel hem de parçalı işlemler yapabilmektedir.

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Üretici fonksiyon yoluyla parçalı eğitim yaparken fit metodunun birinci parametresine 
bir üretici fonksiyon nesnesi girilir. Artık fit metodunun batch_size ve 
validation_split parametrelerinin bir önemi kalmaz. Çünkü batch miktarı zaten üretici 
fonksiyonden elde  edilen verilerle yapılmaktadır. Kaldı ki bu yöntemde aslında 
her batch işleminde aynı miktarda verinin kullanılması da zorunlu değildir. Yine 
bu biçimde eğitim yapılırken fit metodunun validation_split parametresinin de bir 
anlamı yoktur. Çünkü sınama işlemi de yine parçalı verilerle yapılmaktadır. 
Dolayısıyla sınama verilerinin parçalı olarak verilmesi de yine uygulamacının 
sorumluluğundadır. Ancak bu yöntemde programcının iki parametreyi yine açıkça 
belirlemesi gerekir. Birincisi steps_per_epoch parametresidir. Bu parametre bir 
epoch işleminin kaç batch işleminden oluşacağını belirtir. İkinci parametre ise 
epochs parametresidir. Bu da yine toplam epoch sayısını belirtir. (epoch 
parametresinin default değerinin 1 olduğunu anımsayınız.) Bu durumda programcının 
üretici fonksiyon içerisinde epochs * steps_per_epoch kadar yield uygulaması gerekir. 
Çünkü toplam batch sayısı bu kadardır. fit metodu aslında epochs * steps_per_epoch 
kadar işlemden sonra son kez bir daha next işlemi yaparak üretici fonksiyonun 
bitmesine yol açmaktadır.

Aşağıdaki toplam 100 epoch'tan oluşan her epoch'ta 20 tane batch işlemi yapılan 
ikili sınıflandırma örneği verilmiştir. Ancak bu örnekte x ve y verileri üretici 
fonksiyonlardan elde edilmiştir. Üretici fonksiyon içerisinde toplam epochs * steps_per_epoch  
kadar yield işlemi yapılmıştır. Ancak bu örnekte bir sınama işlemi yapılmamıştır. 
Biz bu örneği rastgele verilerle yalnızca mekanizmayı açıklamak için veriyoruz. 
(Rastgele verilerde rastgele sayı üreticisi uygun bir biçimde oluşturulmuşsa bir 
kalıp söz konusu olmayacağı için "binary_accuracy" metrik değerinin 0.50 civarında 
olması beklenir.)

---------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

EPOCHS = 100
NFEATURES = 10
STEPS_PER_EPOCH = 20
BATCH_SIZE = 32

def data_generator():
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            x = np.random.random((BATCH_SIZE, NFEATURES))
            y = np.random.randint(0, 2, BATCH_SIZE)
            yield x, y

model = Sequential(name='Diabetes')

model.add(Input(shape=(NFEATURES,)))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(data_generator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

---------------------------------------------------------------------------------
Yukarıdaki örnekte bir sınama işlemi yapılmamamıştır. İşte sınama işlemleri için 
veriler de tek hamlede değil parça parça verilebilmektedir. 

Bunun için yine fit metodunun validation_data parametresine bir üretici fonksiyon 
nesnesi girilir. fit metodu da her epoch sonrasında validation_steps parametresinde 
belirtilen miktarda bu üretici fonksiyon üzerinde iterasyon yaparak bizden sınama 
verilerini almaktadır. Böylece biz her epoch sonrasında kullanılacak sınama 
verilerini fit metoduna üretici fonksiyon nesnesi yoluyla parça parça vermiş oluruz. 
Sınama verilerinin parçalı oluşturulması sırasında üretici fonksiyonlarda her 
epoch için validation_steps parametresi kadar değil bundan 1 fazla yield işlemi 
yapılmalıdır. Bu fit metodunun içsel tasarımıyla ilgilidir. 

Aşağıda sınama verilerinin parçalı bir biçimde nasıl verildiğine ilişkin bir 
örnek verilmiştir.

---------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

EPOCHS = 100
NFEATURES = 10
STEPS_PER_EPOCH = 20
BATCH_SIZE = 32
VALIDATION_STEPS = 10

def data_generator():
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            x = np.random.random((BATCH_SIZE, NFEATURES))
            y = np.random.randint(0, 2, BATCH_SIZE)
            yield x, y   

def validation_generator():
    x = np.random.random((BATCH_SIZE, NFEATURES))
    y = np.random.randint(0, 2, BATCH_SIZE)
    
    for _ in range(EPOCHS):
        for _ in range(VALIDATION_STEPS + 1):
            yield x, y

model = Sequential(name='Diabetes')

model.add(Input(shape= (NFEATURES, )))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
              metrics=['binary_accuracy'])

model.fit(data_generator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, 
        validation_data=validation_generator(), validation_steps=VALIDATION_STEPS)

---------------------------------------------------------------------------------
Benzer biçimde predcit metodunda da kesitirimi yapılacak veriler bizden parça parça 
istenebilmektedir. Bunun için yine predict metodunun x parametresine bir üretici 
fonksiyon nesnesi girilir. predcit metonunun da kestirim verilerininin kaç parça
olarak isteneceğine yönelik steps parametresi vardır. Tabii sınama işlemi sırasında 
üretici fonksiyon artık yalnızca x değerlerini vermelidir. Aşağıda buna ilişkin 
bir örnek verilmektedir. 

---------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

EPOCHS = 100
NFEATURES = 10
STEPS_PER_EPOCH = 20
BATCH_SIZE = 32
VALIDATION_STEPS = 10
EVALUATION_STEPS = 15
PREDICTION_STEPS = 5

def data_generator():
    for _ in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            x = np.random.random((BATCH_SIZE, NFEATURES))
            y = np.random.randint(0, 2, BATCH_SIZE)
            yield x, y   

def validation_generator():
    x = np.random.random((BATCH_SIZE, NFEATURES))
    y = np.random.randint(0, 2, BATCH_SIZE)
    for _ in range(EPOCHS):
        for _ in range(VALIDATION_STEPS + 1):
            yield x, y

def evaluation_generator():
    for _ in range(EVALUATION_STEPS):
        x = np.random.random((BATCH_SIZE, NFEATURES))
        y = np.random.randint(0, 2, BATCH_SIZE)
        yield x, y
        
def prediction_generator():
    for _ in range(PREDICTION_STEPS):
        x = np.random.random((BATCH_SIZE, NFEATURES))
        yield x

model = Sequential(name='Diabetes')

model.add(Input(shape= (NFEATURES, )))
model.add(Dense(16, activation='relu', name='Hidden-1'))
model.add(Dense(16, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
              metrics=['binary_accuracy'])

model.fit(data_generator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, 
          validation_data=validation_generator(), validation_steps=VALIDATION_STEPS)

eval_result = model.evaluate(evaluation_generator(), steps=EVALUATION_STEPS)
predict_result = model.predict(prediction_generator(), steps=PREDICTION_STEPS)

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Şimdi de gerçekten bellekte büyük bir yer kaplayan vektörizasyon işlemi gerektiren 
bir örneği parçalı bir biçimde eğitelim ve test edelim. Bu örnek için IMDB veri 
kümesini kullanalım. Burada karşılaşacağımız önemli bir sorun "karıştırma (shuffling)"
işleminin nasıl yapılacağına ilişkindir. 

Bilindiği gibi fit işlemi sırasında her epoch işleminden sonra eğitim veri kümesi
karıştırılmaktadır. (Aslında epoch sonrasında veri kümesinin karıştırılıp 
karıştırılmayacağı fit metodundaki shuffle parametresi ile belirlenebilmektedir. 
Bu paramerenin default değeri True biçimindedir.) Parçalı eğitim yapılırken fit 
metodunun shuffle parametresinin bir etkisi yoktur. Dolayısıyla veri kümesinin 
epoch sonrasında karıştırılması programcı tarafından üretici fonksiyon içeisinde 
yapılmalıdır. Pekiyi programcı bu karıştırmayı nasıl yapabilir? CSV dosyasının 
dosya üzerinde karıştırılması uygun bir yöntem değildir. Bu durumda birkaç yöntem 
izlenebilir:

1) Veri kümesi yine read_csv fonksiyonu ile tek hamlede belleğe okutulabilir. Her 
batch işleminde bellekteki ilgili batch'lik kısım verilebilir. Karıştırma da bellek 
üzerinde yapılabilir. Ancak veri kümesini belleğe okumak yapılmak istenen şeye 
bir tezat oluşturmaktadır. Ancak yine de text işlemlerinde asıl bellekte yer 
kaplayan unsur vektörizasyon işleminden elde edilen matris olduğu için bu yöntem 
kullanılabilir. 

2) Veri kümesi bir kez okunup dosyadaki kaydın offset numaraları bir dizide saklanabilir. 
Sonra bu dizi karıştırılıp dizinin elemanlarına ilişkin kayıtlar okunabilir. Eğer 
veritabanı üzerinde doğrudan çalışılıyorsa da işlemler benzer biçimde yürütülebilir. 


Pekiyi biz vektörizasyon işlemini TextVectorization sınıfı türünden katman nesnesiyle 
yapmaya çalışırken tüm verileri yine belleğe çekmek zorunda değil miyiz? Aslında 
TextVectorization sınıfının adapt metodu henüz görmediğimiz Tensorflow kütüphanesindeki 
Dataset nesnesini de parametre olarak alabilmektedir. Bu sayede biz bir Dataset 
nesnesi oluşturarak adapt metodunun da verileri parçalı bir biçimde almasını 
sağlayabiliriz. 

Scikit-learn kütüphanesindeki CountVectorizer sınıfının fit metodu da aslında 
dolaşılabilir nesneyi parametre olarak alabilmektedir. Dolayısıyla CountVectorizer 
kullanılırken yine üretici fonksiyonlar yoluyla biz verileri fit metoduna parça 
parça verebilmekteyiz. Dosya nesneslerinin de dolaşılabilir nesneler olduğunu 
anımsayınız.

---------------------------------------------------------------------------------
Peki parçalı verilerle eğitim yapılırken sınama, test ve kestirim işlemlerini de 
parçalı bir biçimde yapabilir miyiz? Evet bu işlemlerin hepsi parçalı bir biçimde 
yapılabilmektedir. fit metodunda validation_data parametresi bir üretici nesne 
olarak (ya da bir PyDataset nesnesi olarak) girilirse bu durumda her epoch'tan 
sonra sınama verileri bu üretici fonksiyondan (ya da PyDataset nesnesinden) elde 
edilecektir. Ancak bu durumda fit metodunun validation_steps parametresinde kaç 
dolaşımla (yield işlemi ile) sınama verilerinin elde edileceği de girilmelidir.
Örneğin:

model.fit(..., validation_data=validation_generator(), validation_steps=32)

Burada sınama verilerinin elde edilmesi için toplma 32 kez dolaşım (yield işlemi) 
yapılacaktır. Her epoch sonrasındaki sınamada sınama veri kümesinin karıştırılmasına 
gerek yoktur. 

Test işlemi de benzer biçimde parçalı olarak yapılabilir. Bunun için Sequential 
sınıfının x parametresine bir üretici nesne (ya da bir Sequential nesnesi) girilir. 
Test işlemi yapılırken kaç kere dolaşım uygulanacağı da steps parametresiyle
belirtilmektedir. Örneğin:

eval_result = model.evalute(test_generator(), steps=32)

Kestirim işleminde parçalı veri kullanılmasına genellikle gereksinim duyulmuyor 
olsa da kestirim işlemi yine parçalı verilerle yapılabilir. Bunun için Sequential 
sınıfının predict metdounda x parametresine bir üretici nesne (ya da PyDataset nesnesi) 
nesne gerilir. Yine metodun steps parametresi sınama verileri için kaç kez dolaşım 
uygulanacağını (yani yield yapılacağını) beelirtir. Örneğin:

predict_result = model.predict(predict_generator(), steps=32)

---------------------------------------------------------------------------------
"""
"""
---------------------------------------------------------------------------------
Parçalı biçimde eğitim, test ve kestirim işlemi üretici fonksiyon yerine sınıfsal 
bir biçimde de yapılabilmektedir. Bunun için tensorflow.keras.utils modülü içerisindeki 
PyDataset sınıfından türetilmiş sınıflar kullanılmaktadır. (Aslında bir süre öncesine 
kadar bu amaçla Sequence isimli bir sınıftan türetme yapılıyordu. Ancak Keras 
ekibi bunun yerine TenserFlow kütüphanesi içerisindeki Dataset sınıfını Keras'tan 
kullanılabilir hale getirdi. Dokümantasyondan da Sequence sınıfını kaldırıldı.) 

Programcı türetmiş sınıf içerisinde __len__ ve __getiitem__ metotlarını yazar. 
fit metodu bir epoch'un kaç tane batch içerdiğini tespit etmek için sınıfın __len__ 
metodunu çağırmaktadır. Daha sonra fit metodu eğitim sırasında her batch bilgiyi 
elde etmek için sınıfın __getitem__ metodunu çağırır. Bu metodu çağırırken batch 
numarasını metoda parametre olarak geçirir. Bu metottan programcı batch büyüklüğü 
kadar x ve y verisinden oluşan bir demetle geri dönmelidir. Tabii aslında metodun 
geri döndürdüğü x ve Y değerleri her defasında aynı uzunlukta olmak zorunda da 
değildir. Sonra programcı fit metodunun training_dataset_x parametresine bu sınıf 
türünden bir nesne yaratarak o nesneyi girer. Örneğin:

class DataGenerator(PyDataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, batch_no):
        pass
...
model.fit(DataGenrator(...), epochs=100)

Tabii artık bu yöntemde fit metodunun steps_per_epoch parametresi kullanılmamaktadır. 
Metodun bath_size parametresinin de bu yöntemde bir anlamı yoktur. Ancak epochs 
parametresi kaç epoch uygulanacağını belirtmek içn kullanılmaktadır.

Sınama işlemi yine benzer bir biçimde yapılmaktadır. Yani programcı yine PyDataset 
sınıfından sınıf türetip __len__ ve __getitem__ metotlarını yazar. Tabii bu durumda 
her epoch sonrasında bu sınama verileri __getitem__ metodu yoluyla elde edilecektir. 
Programcı yine bunun için validation_data parametresine PyDataset sınıfından 
türettiği sınıf türünden bir nesne girer. Örneğin:

model.fit(DataGenrator(...), epochs=100, validation_data=DataGenerator(....))

Ayrıca PyDataset sınıfından türetilmiş olan sınıfta on_epoch_end isimli bir metot 
da yazılabilmektedir. Eğitim sırasında her epoch bittiğinde fit tarafından bu metot 
çağrılmaktadır. Programcı da tipik olarak bu metotta eğitim veri kümesini karıştırır.  
Bu biçimdeki parçalı eğitimde artık fit metodunun steps_per_epoch gibi bir 
parametresi kullanılmamaktadır. Zaten bu bilgi metot tarafından __len__ metodu 
çağrılarak elde edilmektedir. Benzer biçimde evaluate işleminde de yine Sequence 
sınıfından sınıf türetilerek test edilecek veriler parçalı bir biçimde evalaute 
metoduna verilebilmektedir. 

Uygulamada parçalı verilerle eğitim işleminde aslında üretici fonksiyonlardan 
ziyade bu sınıfsal yöntem daha çok tercih edilmektedir. Bu yöntemi uygulamak daha 
kolaydır. Ayrıca toplamda bu yöntem daha hızlı olma eğilimindedir. 

---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
Pekiyi biz parçalı eğitimi fit metodunu birden fazla kez çağırarak yapamaz mıyız? 
Keras'ın orijinal dokümanlarında bu konuda çok açık örnekler verilmemiştir. Ancak 
kaynak kodlar incelendiğinde fit işleminin artırımlı bir biçimde yapıldığı 
görülmektedir. Yani birden fazla kez fit metodu çağrıldığında eğitim kalınan yerden 
devam ettirilmektedir. Bu nedenle biz eğitimi farklı zamanlarda fit işlemlerini 
birden fazla kez yaparak devam ettirebiliriz. Ancak fit metodunun bu biçimde birden 
fazla kez çağrılması işleminde dikkat edilmesi gereken bazı noktalar da olabilmektedir. 
Keras ekibi bu tür  parçalı eğitimler için daha aşağı seviyeli xxx_on_bath isimli 
metotlar bulundurmuştur. Programcının birden fazla kez fit metodu çağırmak yerine 
bu metotları kullanması daha uygundur. Parçalı işlem için Sequential sınıfının 
şu metotları bulundurulmuştur:

train_on_batch
test_on_batch
predict_on_batch


Ancak bu yöntemde sınama işlemleri otomatik olarak train_on_batch içerisinde 
yapılmamaktadır. Programcının sınamayı kendisinin yapması gerekmektedir. 

train_on_batch metodunun parametrik yapısı şöyledir:

train_on_batch(x, y=None, sample_weight=None, class_weight=None, return_dict=False)


Burada x ve y parametreleri parçalı eğitimde kullanılacak x ve y değerlerini 
almaktadır. sample_weight ve class_weight parametreleri ağırlıklandırmak için 
kullanılmaktadır. return_dict parametresi True geçilirse metot bize geri dönüş 
değeri olarak loss değerini ve metrik değerleri bir sözlük nesnesi biçiminde 
vermektedir. 

train_on_batch metodu ile parçalı eğitim biraz daha düşük seviyeli olarak yapılmaktadır. 
Bu biçimde parçalı eğitimde epoch döngüsünü ve batch döngüsünü programcı kendisi 
oluşturmalıdır. Örneğin:

for epoch in range(EPOCHS):
    <eğitim veri kümesi karıştırılıyor>
    for batch_no in range(NBATCHES):
        <bir batch'lik x ve y elde ediliyor>
        model.train_on_batch(x, y)

Tabii yukarıda da belirttiğimiz gibi bu biçimde çalışma aşağı seviyelidir. Yani 
bazı şeyleri programcının kendisinin yapması gerekmektedir. Örneğin fit metodu 
bize bir History sınıfı türünden bir callback nesnesi veriyordu. Bu nesnenin 
içerisinden de biz tüm epoch'lara ilişkin değerleri elde edebiliyorduk. train_on_batch 
işlemleriyle eğitimde bu işlemlerin bizim tarafımızdan yapılması gerekmektedir. 
train_on_batch metodunun return_dict parametresi True geçilirse batch işlemi 
sonucundaki loss ve metik değerler bize bir sözlük biçiminde verilmektedir. 
Örneğin:

for epoch in range(EPOCHS):
    <eğitim veri kümesi karıştırılıyor>
    for batch_no in range(NBATCHES):
        <bir batch'lik x ve y elde ediliyor>
        rd = model.train_on_batch(x, y, return_dict=True)


Burada her epoch sonrasında değil her batch sonrasında değerlerin elde edildiğine 
dikkat ediniz. Aslında biz Tensorflow ya da PyTorch kütüphanelerini aşağı seviyeli 
olarak kullanacak olsaydık zaten yine işlemleri bu biçimde iki döngü yoluyla
yapmak durumunda kalacaktık. Genellikle uygulamacılar her batch işleminde elde 
edilen değerlerin bir ortalamasını epoch değeri olarak kullanmaktadır. 

Bu yöntemde epoch sonrasındaki sınama işlemlerinin de programcı tarafından manuel 
olarak yapılması gerekmektedir. Yani programcı sınama veri kümesini kendisi 
oluşturmalı ve sınamayı kendisi yapmalıdır. Aslında evaulate metodu test amaçlı 
kullanılsa da sınama amaçlı da kullanılabilir. Bu durumda sınama işlemi şöyle 
yapılabilir:

for epoch in range(EPOCHS):
    <eğitim veri kümesi karıştırılıyor>
    for batch_no in range(NBATCHES):
        <bir batch'lik x ve y elde ediliyor>
        rd = model.train_on_batch(x, y, return_dict=True)
    val_result = model.evaluata(validation_x, validation_y)


Tabii burada evaluate işlemini de parça parça yapmak isteyebilirsiniz. Bu durumda 
yine bir döngü oluşturup test_on_batch fonksiyonunu kullanabilirsiniz. test_on_batch 
metodunun parametrik yapısı şöyledir

test_on_batch(x, y=None, sample_weight=None, return_dict=False)


Kullanım tamamen train_batch metodu gibidir. 

Test işlemi de tüm epoch'lar bittiğinde yine parçalı bir biçimde test_on_batch 
metoduyla yapılabilir. Kestirim işlemi de yine benzer bir biçimde predcit_on_batch 
metoduyla yapılmaktadır. Metodun parametrik yapısı şöyledir:

predict_on_batch(x)

Metot kestirim değerleriyle geri dönmektedir. Örneğin:

for batch_no in range(NBATCHES):
    <bir batch'lik x elde ediliyor>
    predict_result = model.predict_on_batch(predict_x)

---------------------------------------------------------------------------------
"""



# Seyrek(Sparce) Matrisler

"""
---------------------------------------------------------------------------------
Elemanlarının çok büyük kısmı 0 olan matrislere "seyrek matrisler (sparse matrices)" 
denilmektedir. Seyreklik (sparsity) 0 olan elemanların tüm elemanlara oranıyla 
belirlenmektedir. Bir matirisin seyrek olarak ele alınması için herkes tarafından
kabul edilen bir seyreklik oranı yoktur. Seyreklik oranı ne kadar yüksek olursa 
onların alternatif veri yapılarıyla ifade edilmeleri o kadar verimli olmaktadır. 
Makine öğrenmesinde seyrek matrisle sıkça karşılaşılmaktadır. Örneğin bir grup 
yazıyı vektörel hale getirdiğimizde aslında bir seyrek matris oluşmaktadır. Benzer 
biçimde one-hot encoding dönüştürmesi de bir seyrek matris oluşturmaktadır. 
scikit-learn kütüphanesindeki 

OneHotEncoder sınıfının ve CountVectorizer sınıfının çıktı olarak seyrek matris 
verdiğini anımsayınız.

Seyrek matrislerin daha az yer kaplayacak biçimde tutulmasındaki temel yaklaşım 
matrisin yalnızca sıfırdan farklı elemanlarının ve onların yerlerinin tutulmasıdır. 
Örneğin bir milyon elemana sahip bir seyrek matriste yalnızca 100 eleman sıfırdan 
faklıysa biz bu 100 elemanın değerini ve matristeki yerini tutarsak önemli bir 
yer kazancı sağlayabiliriz.

---------------------------------------------------------------------------------
Seyrek matrisleri ifade etmek için alternatif birkaç veri yapısı kullanılmaktadır. 
Bunlardan biri DOK (Dictionary Of Keys) denilen veri yapısıdır. Bu veri yapısında 
matrisin yalnızca 0'dan farklı olan elemanları bir sözlükte tutulur. Sözlüğün 
biçimi aşağıdaki gibidir:

{(34674,17000): 1, (542001, 170): 4, ...}

Burada anahtar satır ve sütun numarasını belirten demettir. Değer ise o elemandaki 
değeri belirtir. Örneğin aşağıdaki gibi bir matris söz konusu olsun:

0 0 5
3 0 0 
0 6 0

Buradaki dok sözlüğü şöyle olacaktır:

{(0, 2): 5, (1, 0): 3, (2, 1): 6}

---------------------------------------------------------------------------------
"""
"""
---------------------------------------------------------------------------------
NumPy ---> Vektörel işlemler yapan C'de yazılmış taban bir kütüphanedir. Pek çok 
        proje NumPy kütüphanesini kendi içerisinde kullanmaktadır.


Pandas ---> Bu kütüphane sütunlu veri yapılarını (yani istatistiksel veri tablolarını) 
        ifade etmek için kullanılmaktadır. Kütüphanenin en önemli özelliği farklı 
        türlere ilişkin sütunsal bilgilerin DataFrame isimli bir veri yapısı ile 
        temsil edilmesidir. Bu kütüphane de NumPy kullanılarak yazılmıştır.


scikit-learn ---> Bir makine öğrennesi kütüphanesidir. Ancak yapay sinir ağlarıyla 
            ilgili özellikler yoktur (minimal düzeydedir). Yani bu kütüphane yapay 
            sinir ağlarının dışındaki makine öğrenmesi için kullanılmaktadır. 
            scikit-learn kendi içerisinde NumPy, Pandas ve SciPy kütüphanelerini 
            kullanmaktadır.


SciPy ---> Genel amaçlı matematik ve nümerik analiz kütüphanesidir. Bu kütüphanenin 
        doğrudan makine öğrenmesiyle bir ilgisi yoktur. Ancak matematiğin çeşitli 
        alanlarına ilişkin nümeraik analiz işlemleri yapan geniş bir taban kütüphanedir. 
        Bu kütüphane de kendi içerisinde NumPy ve Pandas kullanılarak yazılmıştır. 


TensorFlow ---> Google tarafından yapay sinir ağları ve makine öğrenmesi için 
        oluşturulmuş taban bir kütüphanedir. Bu kütüphane scikit-learn kütüphanesinden 
        farkı olarak Tensor adı altında biren fazla CPU ya da çekirdek kullanacak 
        biçimde özellikle yapay sinir ağları için oluşturulmuş taban bir kütüphanedir. 
        Kütüphane Google tarafından geliştirilmiştir.


Keras ---> Yapay sinir ağı işlemlerini kolaylaştırmak için oluşturulmuş olan yüksek 
        seviyeli bir kütüphanedir. Eskiden bu kütüphane "backend" olarak farklı 
        kütüphaneleri kullanbiliyordu. Eski hali devam ettirilse de kütüphane 
        tamamen TensorFlow içerisine dahil edilmiştir ve TensorFlow kütüphanesinin 
        yüksek seviyeli bir katmanı haline getirilmiştir.


PyTorch ---> Tamamen TensorFlow kütüphanesinde hedeflenen işlemleri yapan taban 
        bir yapay sinir ağı ve makine öğrenmesi kütüphanesidir. Facebook (Meta) 
        tarafından geliştirilmiştir. 


Theano --> TensorFlow, PyTorch SciPy benzeri bir taban kütüphanedir. Akademik 
        çevreler tarafından geliştirilmiştir.

---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
NumPy kütüphanesi içerisinde seyrek matrislerle işlem yapan sınıflar ya da fonksiyonlar 
bulunmamaktadır. Ancak SciPy kütüphanesi içerisinde seyrek matrislerle ilgili 
işlemler yapan sınıflar ve fonksiyonlar vardır. scikit_learn kütüphanesi doğrudan 
SciPy kütüphanesinin seyrek matris sınıflarını kullanmaktadır.

---------------------------------------------------------------------------------
DOK biçimindeki seyrek matrisler SciPy kütüphanesinde scipy.sparse modülü içerisindeki 
dok_matrix sınıfıyla temsil edilmiştir. Biz bir dok_matrix sınıfı türünden nesneyi 
yalnızca boyut belirterek yaratabiliriz. Daha sonra bu nesneyi sanki bir NumPy 
dizisiymiş gibi onu kullanabiliriz. Seyrek matrisi normal bir Numpy dizisine 
dönüştürmek için sınıfın todense ya da toarray metotları kullanılmaktadır. 
Örneğin:
    
from scipy.sparse import dok_matrix

dok = dok_matrix((10, 10), dtype='int32')
dok[1, 3] = 10
dok[3, 5] = 20

a = dok.todense()
print(dok)
print('-' * 20)
print(a)


dok_matrix sınıfının minimum, maximum, sum, mean gibi birtakım faydalı metotları 
bulunmaktadır. nonzero metodu sıfır dışındaki elemanların indekslerini vermektedir. 

---------------------------------------------------------------------------------
 Bir seyrek matris nesnesi ile biz NumPy dizileri üzerinde yaptığımız işlemlerin 
benzerlerini yapabiliriz. Örneğin bir seyrek matrisi dilimleyebiliriz. Bu durumda 
yine bir seyrek matris elde ederiz. dok_matrix sınıfının keys metodu yalnızca 
anahtarları, values metodu ise yalnızca değerleri vermektedir. 

Biz seyrek matrislerin karşılıklı elemanları üzerinde işlemler yapabiliriz. Ancak 
her türlü işlem değişik veri yapılarına sahip seyrek matrislerde aynı verimlilikte 
yapılamamaktadır. Örneğin iki dok_matrix nesnesini toplayabiliriz ya da çarpabiliriz. 
Ancak bu işlemler yavaş olma eğilimindedir. Örneğin:

    
from scipy.sparse import dok_matrix

dok1 = dok_matrix((5, 5), dtype='int32')
dok1[1, 2] = 10
dok1[0, 1] = 20

print(dok1.keys())
print(dok1.values())

dok2 = dok_matrix((5, 5), dtype='int32')
dok2[3, 2] = 10
dok2[4, 1] = 20
dok2[1, 2] = 20

---------------------------------------------------------------------------------   
from scipy.sparse import dok_matrix
import numpy as np

a = np.random.randint(0, 2, (10, 10))
b = np.random.randint(0, 2, (10, 10))

dok1 = dok_matrix(a, dtype='float32')
dok2 = dok_matrix(b, dtype='float32')

dok3 = dok1 + dok2

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Diğer bir seyrek matris veri yapısı da "LIL (List of Lists)" denilen veri yapısıdır. 
Bu veri yapısında matrisin satır satır 0 olmayan elemanları ayrı listelerde tutulur. 
Başka bir listede de bu sıfır olmayan elemanların sütunlarının indeksi tutulmaktadır. 
LIL matrisler SciPy kütüphanesinde scipy.sparse modülündeki lil_matrix sınıfıyla 
temsil edilmektedir. Bu sınıfın genel kullanımı dok_matrix sınıfında olduğu gibidir.  
Sınıfın data ve rows örnek öznitelikleri bize bu bilgileri vermektedir. Örneğin 
aşağıdaki gibi bir matrisi lil_matrix yapmış olalım:

[[ 0  0 10 20  0]
 [15  0  0  0 40]
 [12  0 51  0 16]
 [42  0 18  0 16]
 [ 0  0  0  0  0]]

Buradaki data listesi şöyle olacaktır:

array([list([10, 20]), list([15, 40]), list([12, 51, 16]), list([42, 18, 16]), list([])], dtype=object)

rows listesi de şöyle olacaktır:

array([list([2, 3]), list([0, 4]), list([0, 2, 4]), list([0, 2, 4]), list([])], dtype=object)

LIL matrisler de artimetik işlemlerde yavaştır. Dilimleme işlemleri de bu matrisler 
de nispeten yavaş yapılmaktadır.

---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
Aslında uygulamada DOK ve LIL matrisler seyrek kullanılmaktadır. Daha çok CSR ve 
CSC veri yapıları tercih edilmektedir. CSR (Compressed Sparse Row), ve CSC 
(Compressed Sparse Column) matrisleri genel veri yapısı olarak birbirlerine çok 
benzemektedir. Bunlar adeta birbirlerinin tersi durumundadır. 

Bu veri yapıları seyrek matrislerin karşılıklı elemanlarının işleme sokulması 
durumunda DOK ve LIL veri yapılarına göre daha avantajlıdır. CSR satır dilimlemesini 
CSC ise sütun dilimlemesi hızlı yapabilmektedir. Ancak bu matrislerde sparse bir 
matrisin 0 olmayan bir elemanına atama yapmak nispeten yavaş bir işlemdir. 

---------------------------------------------------------------------------------
CSR veri yapısı da SciPy kütüphanesinde scipy.sparse modülünde csr_matrix sınıfıyla 
temsil edilmektedir. CSR matrislerde sıfırdan farklı elemanlar üç dizi (liste) 
halinde tutulmaktadır:

----- > data, indices, indptr. 

Bu diziler sınıfın aynı isimli örnek özniteliklerinden elde edilebilmektedir. 
data dizisi sıfır olmayan elemanların tutulduğu tek boyutlu dizidir. indices dizisi 
data dizisindeki elemanların kendi satırlarının hangi sütunlarında bulunduğunu 
belirtmektedir. indptr dizisi ise sıfır olmayan elemanların hangi satırlarda olduğuna 
ilişkin ilk ve son indeks (ilk indeks dahil, son indeks dahil değil) değerlerinden 
oluşmaktadır. indptr dizisi hep yan yana iki eleman olarak değerlendirilmelidir. 
Soldaki eleman ilk indeksi, sağdaki eleman ise son indeksi belirtir. 

Örneğin:

0, 0, 9, 0, 5
8, 0, 3, 0, 7
0, 0, 0, 0, 0
0, 0, 5, 0, 9
0, 0, 0, 0, 0

Burada söz konusu üç dizi şöyledir:

data: [9, 5, 8, 3, 7, 5, 9]
indices: [2, 4, 0, 2, 4, 2, 4]
indptr: [0, 2, 5, 5, 7, 7]

csr_matrix sınıfının genel kullanımı diğer seyrek matris sınıflarındaki gibidir. 
Ancak CSR ce CSC matrislerde sıfır olan bir elemana atama yapmak yavaş bir işlemdir. 
Çünkü bu işlemler yukarıda belirtilen üç dizide kaydırmalara yol açmaktadır. Bu 
tür durumlarda DOK ya da LIL matrisler daha hızlı işleme yol açarlar. Bu nedenle 
bu matrisler kullanılırken sıfır olmayan bir elemana atama yapıldığında bir uyarı 
mesajıyla karşılaşabilirsiniz. O halde CSR ve CSC matrisleri işin başında 
oluşturulmalı ve sonra da onların elemanları bir daha değiştirilmemelidir.

CSR matrislerinde satırsal, CSC matrislerinde sütunsal dilimlemeler hızlıdır. 
Aynı zamanda bu iki matrisin karşılıklı elemanları üzerinde hızlı işlemler 
yapılabilmektedir. 

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
CSC formatı aslında CSR formatına çok benzerdir. CSR formatı adeta CSC formatının 
sütunsal biçimidir. Yani iki format arasındaki tek fark CSR formatında satır 
indeksleri tutulurken, CSC formatında sütun indekslerinin tutulmasıdır. Yani 
yapılan işlemlerin hepsi satır-sütun temelinde terstir. Örneğin:

    
0, 0, 9, 0, 5
8, 0, 3, 0, 7
0, 0, 0, 0, 0
0, 0, 5, 0, 9
0, 0, 0, 0, 0

data: [8, 9, 3, 5, 5, 7, 9]
indices: [1, 0, 1, 3, 0, 1, 3] 
indptr: [0, 1, 1, 4, 4, 7]

---------------------------------------------------------------------------------
from scipy.sparse import csc_matrix

a = [[0, 0, 9, 0, 5], [8, 0, 3, 0, 7], [0, 0, 0, 0, 0], [0, 0, 5, 0, 9], [0, 0, 0, 0, 0]]
csc = csc_matrix(a)

print(csc.todense(), end='\n\n')
print(f'data: {csc.data}')              # data: [8 9 3 5 5 7 9]
print(f'indices: {csc.indices}')        # indices: [1 0 1 3 0 1 3]
print(f'indptr: {csc.indptr}')          # indptr: [0 1 1 4 4 7]

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Seyrek bir matris train_test_split fonksiyonuyla ayrıştırılabilir. Çünkü zaten 
train_test_split fonksiyonu dilimleme yoluyla işlemlerini yapmaktadır. Ancak 
seyrek matrislere len fonksiyonu uygulanamaz. Fakat seyrek matrislerin boyutları 
yine shape örnek özniteliği ile elde edilebilir. train_test_split fonksiyonu 
seyrek matrisi de karıştırabilmektedir.

---------------------------------------------------------------------------------
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

dense = np.zeros((10, 5))

for i in range(len(dense)):
    rcols = np.random.randint(0, 5, 2)
    dense[i, rcols] = np.random.randint(0, 100, 2)
    
sparse_dataset_x = csr_matrix(dense)
dataset_y = np.random.randint(0, 2, 10)

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(sparse_dataset_x, 
                                                                                dataset_y, test_size=0.2)

print(training_dataset_x)
print('-' * 20)
print(training_dataset_y)
print()
print()

print(test_dataset_x)
print('-' * 20)
print(test_dataset_y)

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Seyrek matrislerin birbirlerine göre avantaj ve dezavantajları şöyle özetlenebilir:

- DOK matriste elemanlara okuma ya da yazma amaçlı erişim hızlı bir biçimde 
gerçekleştirilmektedir. Ancak DOK matrisler matris işlemlerinde etkin değildir. 
DOK matrisler dilimleme de de etkin değildir. 

- LIL matrisler de okuma amaçlı eleman erişimlerinde ve satırsal dilimlemelerde 
hızlıdırlar. Ancak sütunsal dilimlemelerde ve matris işlemlerinde yavaştırlar. 
0 olan elemanlara yazma amaçlı erişimlerde çok hızlı olmasalar da yavaş değillerdir. 
Bu matrislerin matris işlemleri için CSR ve CSC formatlarına dönüştürülmesi 
uygundur ve bu dönüştürme hızlıdır. 

- CSR matrisler satırsal dilimlemelerde CSC matrisler ise sütunsal dilimlemelerde 
hızlıdırlar. Ancak CSR sütünsal dilimlemelerde, CSC de satırsal dilimlemelerde 
yavaştır. Her iki matris de matris işlemlerinde hızlıdır. Bu matrislerde 
elemanların değerlerini değiştirmek (özellikle 0 olan elemanların) yavaştır.     

O halde biz eğer eleman değerleri değiştirilmeyecekse CSR ya da CSC matris 
kullanabiliriz. Ancak eleman değerleri değiştirilecekse önce işlemlemlerimize DOK 
ya da LIL matrisle başlayıp değişikler yapıldıktan sonra matris işlemlerine
başlamadan önce matrisimizi CSR ya da SCS formatına dönüştürebiliriz.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Aslında biz daha önce bazı konularda seyrek matris kavramıyla zaten karşılaşmıştık. 
Örneğin scikit-learn içerisindeki OneHotEncoder sınıfının sparse_output parametresi 
False geçilmezse bu sınıf bize transform işleminde SCiPy'ın CSR formatında seyrek 
matrisini vermektedir. 

---------------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
import numpy as np

a = np.array(['Mavi', 'Yeşil', 'Kırmızı', 'Mavi', 'Kırmızı', 'Mavi', 'Yeşil'])

ohe = OneHotEncoder()
result = ohe.fit_transform(a.reshape(-1, 1))

print(result)
print()
print(type(result))
print()
print(result.todense())

---------------------------------------------------------------------------------
Benzer biçimde scikit-learn kütüphanesindeki CountVectorizer sınıfı da yine bize 
SCiPy'ın CSR formatında seyrek matrisini vermektedir.


from sklearn.feature_extraction.text import CountVectorizer

texts = ['this film is very very good', 'I hate this film', 'It is good', 'I don\'t like it']

cv = CountVectorizer()
cv.fit(texts)
result = cv.transform(texts)

print(result)
print()
print(result.todense())

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Seyrek matris sınıfları büyük kısmı sıfır olan matrislerin bellekte daha az yer 
kaplamasını sağlamak için kullanılmaktadır. Aslında seyrek matrislerle işlemler 
normal (dense) matrislerle işlemlerden her zaman daha yavaştır. Pekiyi bizim 
elimizde bir seyrek matris varsa biz bunu Keras'ta nasıl kullanabiliriz? Örneğin 
CountVectorizer işleminden büyük bir seyrek matris elde etmiş olalım ve Keras'ın 
TextVectorization katmanını kullanmıyor olalım. Bu durumda bu seyrek matrisi sinir 
ağlarında nasıl kullanabiliriz? 

Seyrek matrislerin Keras sinir ağlarında kullanılmasının temelde iki yöntemi vardır:

1) Parçalı eğitim uygulanırken seyrek matrisin ilgili batch'lik kısımları o anda 
   yoğun matrise dönüştürüp verilebilir.

2) Tensorflow kütüphanesinin Input katmanına sonradan bir sparse parametresi 
   eklenmiştir. Bu parametre True yapılarak artık doğrudan dataset_x değerleri 
   SciPy sparse matrisi olarak verilebilmektedir. 

---------------------------------------------------------------------------------
"""




#  ------------------------------ Picture Operations ------------------------------ 

"""
---------------------------------------------------------------------------------
Bilgisayar ekranını bir pixel matrisi olarak düşünebiliriz. Örneğin ekranımızın 
çözünürlüğü 1920x1080 ise (ilk yazılan sütun ikinci yazılan satır) bu ekranımızın 
her satırında 1920 tane pixel olduğu ve toplam 1080 tane satırın bulunduğu anlamına 
gelmektedir. (Yani bu durumda ekranımızda 1920 * 1080 = 2073600 pixel vardır.)

Bugün kullandığımız bilgisayarlarda her pixel (RGB)"kırmızı (red), yeşil (green), 
mavinin (blue)" bir byte'lık tonal birleşimleriyle oluşturulmaktadır. Yani belli 
bir renk kırmızının 0-255 arasındaki bir değeri, yeşilin 0-255 arasındaki bir 
değeri ve mavinin 0-255 arasındaki bir değeri ile oluşturulmaktadır. Örneğin R=255, 
G=0, B=0 ise bu tam kırmızı olan renktir. Kırmızı ile yeşil ışınsal olarak bir 
araya getirilirse sarı renk elde edilmektedir. Bu biçimde bütün renkler aslında 
bu üç ana rengin tonal birleşimleriyle elde edilmektedir. R, G ve B için toplam 
256 farklı değer olduğuna göre elde edilebilecek toplam renk sayısı 256 * 256 * 256
'dır. Bunu 2 ** 8 * 2 ** 8 * 2 ** 8 biçiminde de ifade edebiliriz. 2 ** 24 değeri 
yaklaşık 16 milyon (16777216) civarındadır.

Ekranda iki pixel arasında bir doğru çizdiğimizde bu doğru kırıklı gibi görünebilir. 
Bunun nedeni doğrunun sınırlı sayıda pixel ile oluşturulmasıdır. Kartezyen 
koordinat sisteminde sonsuz tane nokta vardır. Yani çözünürlük sonsuzdur. Ancak 
ekran koordinat sisteminde sınırlı sayıda pixel vardır. Bu nedenle ekranda doğru 
gibi, daire gibi en temel geometrik şekiller bile kırıklı gözükebilmektedir. 
Şüphesiz çözünürlük artırıldığı zaman bu kırıklı görünüm azalacaktır. Ancak 
çözünürlüğün çok artırılması da başka dezavantajlar doğurabilmektedir. Örneğin 
notebook'larda ve mobil cihazlarda CPU dışındaki en önemli güç tüketimi LCD ekranda 
oluşmaktadır. Çözünürlük artırıldıkça ekran kartları ve LCD birimleri daha fazla 
güç harcar hale gelmektedir.

Peki ekran boyutunu sabit tutup çözünürlüğü küçültürsek ne olur? Bu durumda pixel'ler 
büyür ve görüntü daha büyür ancak netlik azalır. Çözünürlüğü sabit tutup ekranımızı 
büyütürsek de benzer durum oluşacaktır. O halde göz için belli bir büyüklükte 
ekran ve çözünürlük daha önemlidir. İşte buna DPI (Dot Per Inch) denilmektedir. 
DPI bir inch'te kaç pixel olduğunu belirtmektedir. Çözünürlük sabit tutulup ekran 
büyütülürse DPI düşer, ekran küçültülürse DPI yükselir. Bugün kullandığımız akıllı 
cep telefonlarında DPI oldukça yüksektir. Buna "retinal çözünürlük" de denilmektedir. 
Gözümüzün de donanımsal (fizyolojik) bir çöznürlüğü vardır. Belli bir DPI'dan 
daha yüksek çözünürlük sağlamanın bizim için bir faydası kalmamaktadır. 

Bilgisayar bilimlerinin sınırlı sayıda pixelle geometrik şekillerin ve resimlerin 
nasıl oluşturulduğunu inceleyen ve bunlar üzerinde işlemlerin nasıl yapılabileceğini 
araştıran bölümüne İngilizce "computer graphics" denilmektedir. 

---------------------------------------------------------------------------------
Bilgisayar ortamındaki en doğal resim formatları "bitmap (ya da raster)" formatlardır. 
Örneğin BMP, GIF, TIF, PNG gibi formatlar bitmap formatlardır. Bitmap dosya 
formatlarında resmin her bir pixel'inin rengi dosya içerisinde saklanır. Dolayısıyla 
resmi görüntüleyecek kişinin tek yapacağı şey o pixel'leri o renklerde görüntülemektir. 
Ancak bitmap formatlar çok yer kaplama eğilimindedir. Örneğin 100x100 pixellik 
bir resim kabaca 100 * 100 * 3 byte yer kaplar. Bu yüzden resimler üzerinde kayıplı 
sıkıştırma yöntemleri oluşturulmuştur. 

Örneğin JPEG formatı aslında bitmap format üzerinde bir çeşit kayıplı sıkıştırmanın 
uygulandığı bir formattır. Yani bir BMP dosyasını JPG dosyasına dönüştürdüğümüzde 
resim çok bozulmaz. Ama aslında biraz bozulur. Sonra onu yeniden BMP dosyasına 
dönüştürdüğümüzde orijinal resmi elde edemeyiz. Ancak JPEG gibi formatlar resimleri 
çok az bozup çok iyi sıkıştırabilmektedir. O halde aslında doğal formatlar BMP 
formatı ve benzerleridir. JPEG gibi formatlar doğal formatlar değildir. Sıkıştırılmış 
formatlardır. Bitmap formatlardaki resimler orijinal boyutuyla görüntülenmektedir. 
Çünkü bu resimlerin büyütülüp küçültülmesinde (scale edilmesinde) görüntü 
bozulabilmektedir.

Bugünkü bilgisayar sistemlerinde arka planda bir görüntü varken onun önüne bir 
görüntü getirilip arkadaki görüntü adeta bir tül perdeden görünüyormuş gibi bir 
etki yaratılabilmektedir. Bu etki aslında ön plandaki pixel ile arka plandaki 
pixel'in bit operasyonuna sokulmasıyla sağlanmaktadır. Bu operasyon bugünkü grafik 
kartlarında grafik kartının kendisi tarafından yapılmaktadır. Ancak bu saydamlılık 
(transparency) özelliğinin de derecesi söz konusu olmaktadır. Programcı bu 
saydamlılık derecesini grafik kartına RGB değerleriyle birlikte birlikte verebilmektedir. 
RGB değerlerinin yanı sıra saydamlılılık belirten bu değere "alpha channel" 
denilmetedir. 

Bazı bitmap formatlar pixel renklerinin yanı sıra her pixel için saydamlılık 
bilgilerini de tutmaktadır. Böylece dikdörtgensel resim başka bir resmin üstüne 
basıldığında ön plan resmin bazı kısımlarının görüntülenmesi engellenebilmektedir. 
Örneğin PNG formatı bu biçimde transparanlık bilgisi de tutulmaktadır. Ancak 
BMP formatında böyle bir transparanlık bilgisi tutulmamaktadır.

Siyah beyaz demek her pixel'in yalnızca siyah ya da beyaz olabildiği resim demektir. 
Böyle bir resimde bir pixel bir bit ile ifade edilebilir. Ancak siyah beyaz 
resimlerde resim bir siluet gibi gözükmektedir. Gri tonlamalı (gray scale) resimlerde
ise her pixel siyahın (grinin) bir tonu biçiminde renkledirilmektedir. Eski 
siyah-beyaz fotoğraflar aslında gri tonlamalı fotoğraflardır. Gri tonlamalı 
resimlerde aslında her pixelin RGB renkleri aynıdır. Yani R = G = B biçimindedir. 
Gri tonlamalı resimlerde grinin 256 tonu görüntülenebilmektedir. Dolayısıyla gri 
tonlamalı bir resimde her pixel bir byte ile ifade edilebilmektedir. 

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Matplotlib kütüphanesinde bir resmi resim dosyasından (JEPG, BMP, PNG vs.) okuyarak 
onun pixel'lerini elde edip bize bir NumPy dizisi biçiminde veren imread isimli 
bir fonksiyon vardır. Biz bir resim dosyasını imread ile okuduğumuzda artık o 
resmin saf pixel değerlerini elde ederiz. Örneğin:

import matplotlib.pyplot as plt

image_data = plt.imread('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\15- PictureOperations\\AbbeyRoad.jpg')


Bu örnekte biz bir NumPy dizisi elde etmiş olduk. Söz konusu resim renkli ir resim 
olduğu için elde edilen dizinin de shape demeti (1000, 1500, 3) biçimindedir. Yani 
söz konusu resim 1000x1500 pixel'lik bir resimdir, ancak resmin her pixel'i RGB 
değerlerinden oluşmaktadır. Biz buarada image_data[i, j] biçiminde matrisin bir 
elemanına erişmek istersek aslında resmin i'inci satır j'inci sütunundaki pixel'in 
RGB renklerini bir NumPy dizisi olarak elde ederiz. 

Matplotlib kütüphanesinin imshow isimli fonksiyonu pixel bilgilerini alarak resmi 
görüntüler. Tabii imshow resmi orijinal boyutuyla görüntülememektedir. imshow 
resmi ölçeklendirip figür büyüklüğünde görüntülemektedir. Örneğin:

    
image_data = plt.imread('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\15- PictureOperations\\AbbeyRoad.jpg')
plt.imshow(image_data)
plt.show()
print(image_data.shape)

---------------------------------------------------------------------------------
Matplotlib bir resim üzerinde ondan parça almak, onu büyütmek, küçültmek gibi işlemler 
için uygun değildir. Bu tür işlemler için Python programcıları başka kütüphanelerden 
faydalamaktadır. Örneğin bu bağlamda en yaygın kullanılan kütüphane 
"Python Image Library (PIL ya da Pillow diye kısaltılmaktadır)" isimli kütüphanedir. 
Matplotlib yalnızca resim dosyalarını okuyup bize pixel'lerini verir ve pixel'leri 
verilmiş resmi görüntüler. 

---------------------------------------------------------------------------------
Örneğin biz bir resmi ters çevirmek için resmin tüm satırlarını ters yüz etmemiz 
gerekir. Bu işlemi aslında NumPy'ın flip fonksiyonu pratik bir biçimde yapmaktadır. 
Bir resmin pixel'leri üzerinde aşağı seviyeli çalışma yapmak için Matplotlib 
ve NumPy iyi araçlardır.  Örneğin:

import numpy as np    
image_data = plt.imread('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\15- PictureOperations\\AbbeyRoad.jpg')

plt.imshow(image_data)
plt.show()

result_image = np.flip(image_data, axis=1)
plt.imshow(result_image)
plt.show()


Burada pixel verileri yatay eksende döndürülmüştür.

---------------------------------------------------------------------------------
NumPy'da rot90 fonksiyonu resmim pixellerini 90 derece döndürmektedir. Fonksiyon 
resmin pixel verileriyle saat yönününün tersinde kaç defa 90 derece döndürüleceğini 
(default değeri 1) bizden istemektedir. Örneğin:

    
mage_data = plt.imread('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\15- PictureOperations\\AbbeyRoad.jpg')

plt.imshow(image_data)
plt.show()

result_image = np.rot90(image_data, 3)
plt.imshow(result_image)
plt.show()


Burada resim 3 kere 90 derece döndürülmüştür.

---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
Renkli resimleri imread fonksiyonu ile okuduğumuzda "row x col x 3" boyutunda bir 
matris elde ederiz. Gri tonlamalı resimleri aynı fonksiyonla okuduğumuzda ise 
row x col x 1 boyutunda bir matris elde ederiz. Tabii aslında "row x col" biçiminde 
iki boyutlu bir matrisin eleman sayısı ile "row x col x 1" biçiminde üç boyutlu 
bir matrisin eleman sayısı arasında bir farklılık yoktur. Bazen gri tonlamalı 
resimler "row x col x 1" yerine "row x col" biçiminde de karşımıza çıkabilmektedir. 

---------------------------------------------------------------------------------
Peki renkli bir resmi gri tonlamalı bir resim haline nasıl getirebiliriz? Bunun 
için en basit yöntem her pixel'in RGB renklerinin ortalamasını almaktır. Bunu basit 
bir biçimde np.mean fonksiyonunda axis=2 parametresini kullanarak sağlayabiliriz.
Örneğin:

 
import numpy as np
import matplotlib.pyplot as plt

image_data = plt.imread('C:\\Users\\Lenovo\\Desktop\\GitHub\\YapayZeka\\Src\\15- PictureOperations\\AbbeyRoad.jpg')

gray_scaled_image_data = np.mean(image_data, axis=2)

plt.imshow(gray_scaled_image_data, cmap='gray')
plt.show()

---------------------------------------------------------------------------------
"""



# MNIST (Modified National Institute of Standards and Technology)

"""
---------------------------------------------------------------------------------
Resim tanıma üzerinde en sık kullanılan popüler bir veri kümelerinden biri MNIST 
denilen veri kümesidir. Bu veri kümesinde her biri 28x28 pixel'den oluşan gri 
tonlamalı resimler vardır. Bu resimler çeşitli kişilerin 0 ile 9 arasındaki sayıları 
elle çizmesiyle oluşturulmuştur. Veri kümesi "resmin üzerindeki sayının kestirilmesi"
gibi resim tanıma uygulamalarında kullanılmaktadır. Veri kümesinde toplam 60000 
tane resim bulunmaktadır. Veri kümesini zip'lenmiş CSV dosyaları biçiminde 
aşağıdaki bağlantıdan indirebilrsiniz:

https://www.kaggle.com/oddrationale/mnist-in-csv  

Buradan minist_train.csv ve mnist_test.csv dosyaları elde edilmektedir. 

MNIST verileri dosyadan okunmuş ve iki saklı katmanlı bir sinir ağı ile model 
oluşturulmuştur. Model test edildiğinde %97 civarında bir başarı elde edilmektedir. 
Daha sonra 28x28'lik kendi oluşturduğumuz bitmap resimlerle kestirim işlemi 
yapılmıştır. Tabii kestirim işlemi eğitim verileriyle aynı biçimde oluşturulmuş 
rakamlarla yapılmalıdır. Eğitim verilerinde "anti-aliasing" özelliği bulunmaktadır. 
Biz de Microsoft Paint ile fırça kullanarak anti-aliasing eşliğinde kestirilecek 
resimleri oluşturduk. Pixel verileri eğitime sokulmadan önce min-max ölçeklemesi 
de yapılmıştır. Tabii [0, 255] arasındaki verilerde min-max ölçeklemesi aslında 
bu pixel verilerinin 255'e bölümüyle oluşturulabilmektedir. Modele çok fazla epoch 
uygulandığında "overfitting" problemi ortaya çıkmaktadır. Bu nedenle epoch sayısı 
20 olarak belirlenmiştir.

---------------------------------------------------------------------------------
"""



# ------------------------ Evrişim (convolution) ------------------------ 


"""
---------------------------------------------------------------------------------
Evrişim (convolution) genel olarak "sayısal işaret işleme (digital signal processing)" 
faaliyetlerinde kullanılan bir tekniktir. Bir verinin başka bir veriyle girişime  
sokulması anlamına gelir. Evrişim en çok görüntü verileri üzerinde kullanılmaktadır. 
Ancak görüntünün dışında işitsel (audio) ve hareketli görüntüler (video) verileri 
üzerinde de sıkça uygulanmaktadır. Evrişim işlemleri ile oluşturulan yapay sinir 
ağlarına Evrişimsel Sinir Ağları (Convolutional Neural Network)" denilmektedir 
ve İngilizce CNN biçiminde kısaltılmaktadır. 

Resimlerde evrişim işlemi pixel'lerin birbirleri ile ilişkili hale gelmesini 
sağlamaktadır. Evrişim sayesinde pixel'ller bağımsız biribirinden kopuk durumdan 
çıkıp komşu pixellerle ilişkili hale gelir. Aynı zamanda evrişim bir filtreleme 
etkisi oluşturmaktadır. Görüntü işlemede filtreleme işlemleri evrişimlerle 
sağlanmaktadır.

Bir resmin evrişim işlemine sokulması için elimizde bir resim ve bir küçük matrisin 
olması gerekir. Bu küçük matrise "filtre (filter)" ya da "kernel" denilmektedir. 
Kernel herhangi bir boyutta olabilir. Kare bir matris biçiminde olması gerekmez. 
Ancak uygulamada NxN'lik kare matrisler kullanılmaktadır ve genellikle buradaki 
N değeri 3, 5, 7, 9 gibi tek sayı olmaktadır. Evrişim işlemi şöyle yapılmaktadır: 

Kernel resmin sol üst köşesi ile çakıştırılır. Sonra resmin arkada kalan kısmıyla 
dot-product işlemine sokulur. (Yani kernel'ın asıl resimle çakıştırıldığı pixel 
değerleri birbiriyle çarpılıp toplanır.) Buradan bir değer elde edilir. Bu değer 
yeni resmin kernel ile çakıştırılan orta noktasındaki pixel'i olur. Sonra kernel 
resim üzerinde kaydırılır ve aynı işlem yine yapılır. Kaydırma sağa doğru ve sonra 
da aşağıya doğru yapılır. Böylece evrişim işlemi sonucunda başka bir resim elde 
edilmiştir. Örneğin aşağıdaki gibi 5x5'lik gri tonlamalı bir resim söz konusu olsun:

---------------------------------------------------------------------------------
a11 a12 a13 a14 a15
a21 a22 a23 a24 a25
a31 a32 a33 a34 a35
a41 a42 a43 a44 a45
a51 a52 a53 a54 a55

Kullandığımız kernel da aşağıdaki gibi 3x3'lük olsun:

b11 b12 b13
b21 b22 b23
b31 b32 b33

Bu kernel resmin sol üst köşesi ile çakıştırılıp dot product uygulanırsa şöyle 
bir değer elde edilir:

c11 = b11 * a11 + b12 * a12 + b13 * a13 + b21 * a21 + b22 * a22 + b23 * a23 + b31 * a31 + b32 * a32 + b33 * a33

Şimdi kernel'ı bir sağa kaydırıp aynı işlemi yapalım:

c12 = b11 * a12 + b12 * a13 + b13 * a14 + b21 * a22 + b22 * a23 + b23 * a24 + b31 * a32 + b32 * a32 + b33 * a34

Şimdi kernel'ı bir sağa daha kaydıralım:

c13 = b11 * a13 + b12 * a14 + b13 * a15 + b21 * a23 + b22 * a24 + b23 * a25 + b31 * a33 + b32 * a34 + b33 * a35

Şimdi kernel'ı aşağı kaydıralım:

c21 = b11 * a21 + b12 * a22 + b13 * a23 + b21 * a31 + b22 * a32 + b23 * a33 + b31 * a41 + b32 * a42 + b33 * a43

İşte bu biçimde işlemlere devam edersek aşağıdkai gibi bir C matrisi (resmi) elde ederiz:

c11 c12 c13
c21 c22 c23
c31 c32 c33

Eğer işlemler yukarıdaki gibi yapılırsa hedef olarak elde edilecek resmin 
genişlik ve yüksekliği şöyle olur:

Hedef Resmin Genişliği = Asıl Resmin Genişliği - Kernel Genişliği + 1
Hedef Resmin Yüksekliği = Asıl Resmin Yüksekliği - Kernel Yüksekliği + 1

Örneğin ana resim 5X5'lik ve kernel'da 3X3'lik ise evrişim işleminin sonucunda 
elde edilecek resim 3X3'lük olacaktır. 

---------------------------------------------------------------------------------
Görüldüğü gibi hedef resim asıl resimden küçük olmaktadır. Eğer biz hedef resmin 
asıl resimle aynı büyüklükte olmasını istersek asıl resmin soluna, sağına, yukarısına 
ve aşağısına eklemeler yaparız. Bu eklemelere İngilizce "padding" denilmektedir. 
Hedef resmin asıl resimle aynı büyüklükte olması için asıl resme (kernel genişliği 
ya da yükseliği - 1) kadar padding uygulanmalıdır.

Toplam Padding Genişliği = Kernel Genişliği - 1
Toplam Padding Yüksekliği = Kernel Yüksekliği - 1

Tabii bu toplam pading genişliği ve yüksekliği iki yöne eşit bir biçimde 
yaydırılmalıdır. Yani başka bir deyişle asıl resim evrişim işlemine sokulmadan 
önce dört taraftan büyütülmelidir. Örneğin 5x5'lik resme 3x3'lük kernel 
uygulamak isteyelim:

a11 a12 a13 a14 a15
a21 a22 a23 a24 a25
a31 a32 a33 a34 a35
a41 a42 a43 a44 a45
a51 a52 a53 a54 a55

Asl resmin padding'li hali şu biçimde görünecektir:

pad pad pad pad pad pad pad
pad a11 a12 a13 a14 a15 pad 
pad a21 a22 a23 a24 a25 pad 
pad a31 a32 a33 a34 a35 pad      (7x7)
pad a41 a42 a43 a44 a45 pad 
pad a51 a52 a53 a54 a55 pad 
pad pad pad pad pad pad pad 

Pekiyi padding'ler asıl resme dahil olmadığına göre hangi içeriğe sahip olacaktır? 
İşte tipik olarak iki yöntem kullanılmaktadır. Birincisi padding'leri 0 almak 
ikincisi ise ilk ve son n tane satır ya da sütunu tekrarlamaktır. Genellikle bu 
ikinci yöntem tercih edilmektedir.

Evrişim işleminde kaydırma birer birer yapılmayabilir. Örneğin ikişer ikişer, üçer 
üçer yapılabilir. Bu kaydırmaya "stride" denilmektedir. stride değeri artırılırsa 
hedef resim padding de yapılmadıysa daha fazla küçülecektir. Hedef resmi küçültmek 
için stride değeri artırılabilmektedir. 


Evrişim işlemi ile ne elde edilmek istenmektedir? Resimlerde evrişim işlemi resmi 
filtrelemek için kullanılır. Resim filtrelenince farklı bir hale gelmektedir. 
Görüntü işlemede resmin bazı yönlerini açığa çıkartmak için amaca uygun çeşitli 
filtreler kullanılabilmektedir. Örneğin biz bir filtre sayesinde resmi bulanık 
(blurred) hale getirebiliriz. Başka bir filtre sayesinde resmin içerisindeki 
nesnelerin sınır çizgilerini elde edebiliriz. Bu konu "sayısal görüntü işleme" 
ile ilgildir. Detayları çeşitli kaynaklardan edinilebilir. 

Evirişim işlemini padding uygulanmadan yapan basit bir fonksiyonu şöyle yazabiliriz:

---------------------------------------------------------------------------------
import numpy as np

def conv(image, kernel):
    image_height = image.shape[0]
    image_width = image.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    
    target = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype='uint8')
    
    for row in range(image_height - kernel_height + 1):
        for col in range(image_width - kernel_width + 1):
            dotp = 0
            for i in range(kernel_height):
                for k in range(kernel_width):
                    dotp += image[row + i, col + k] * kernel[i, k]
            target[row, col] = np.clip(dotp, 0, 255)
    return target


Evrişim işleminin bu biçimde uygulanması yavaş bir yöntemdir. Bu tür işlemlerde 
mümkün olduğunca NumPy içerisindeki fonksiyonlardan faydalanılmalıdır. Çünkü 
NumPy'ın fonksiyonlarının önemli bir bölümü C'de yazılmıştır ve manuel Python
kodlarına göre çok daha hızlı çalışmaktadır.

---------------------------------------------------------------------------------
Peki evrişim işleminin yapay sinir ağları için anlamı nedir? İşte burada işlemler 
yukarıdaki filtreleme örneğin tersi olacak biçimde yürütülmektedir. Yani bir resmin 
sınıfını belirlemek için onu filtreye sokabiliriz. Ancak bu filtrenin nasıl bir 
filtre olacağını da ağın bulmasını sağlayabiliriz. O halde evrişimsel ağlarda biz 
uygulayacağımız filtreyi bilmemekteyiz. 

Biz kestirimin daha isabetli yapılması için resmin nasıl bir filtreden geçirilmesi 
gerektiğini ağın bulmasını sağlarız. Yani ağ yalnızca filtreyi uygulamaz bizzat 
filtrenin kendisini de bulmaya çalışır. Ancak resmin yalnızca filtreden geçirilmesi 
yeterli değildir. Resim filtreden geçirildikten sonra yine Dense katmanlara sokulur. 
Yani filtreleme genellikle ön katmanlarda yapılan bir işlemdir.  Filtreleme 
katmanlarından sonra yine Dense katmanlar kullanılır. Tabii resmi filtrelerden 
geçirmek ve bu filtreleri ağın kendisinin bulmasını sağlamak modeldeki eğitilebilir 
parametrelerin sayısını artırmaktadır. 

Aslında evrişim işlemi nöronlarla ifade edilebilir. Çünkü evrişim sırasında yapılan 
dot-product işlemi aslında nöron girişlerinin ağırlık değerleriyle çarpılıp 
toplanması işlemi ile aynıdır. Yani biz aslında evrişim işlemini sanki bir 
katmanmış gibi ifade edebiliriz.

---------------------------------------------------------------------------------
"""

"""
---------------------------------------------------------------------------------
Biz yukarıda evrişim işleminin gri tonlamalı resimlerde nasıl yapıldığını açıkladık. 
Peki evrişim işlemi RGB resimlerde nasıl yürütülmektedir? RGB resimler aslında 
R, G ve B'lerden oluşan 3 farklı resim gibi ele alınmaktadır. Dolayısıyla üç farklı 
kernel bu R, G ve B resimlere ayrı ayrı uygulanmaktadır. Görüntü işleme 
uygulamalarında bu farklı kernel'ların her bir kanala uygulanması sonucunda ayrı 
bir değer elde edilir. Bu da hedef pixel'in RGB değerleri olur. Ancak sinir 
ağlarında genel olarak 3 farklı kernel her bir kanala uygulandıktan sonra elde 
edilen değerler toplanarak teke düşürülmektedir. Yani adeta biz evrişimsel ağlarda 
renkli resimleri evrişim işlemine soktuktan sonra onlardan gri tonlamalı bir resim 
elde etmiş gibi oluruz. Pekiyi bu işlemde kaç tane bias değeri kullanılacaktır?
Her kanal (channel) için ayrı bir bias değeri kullanılmamaktadır. Bias değeri bu 
kanallardan evrişim sonucunda elde edilen üç değerin toplanmasından sonra işleme 
sokulmaktadır. Dolayısıyla bias değeri yalnızca bir tane olacaktır. 

Örneğin biz 10x10'luk bir RGB resme evrişim uygulamak isteyelim. Kullanacağımız 
filtre matrisi (kernel) 3x3'lük olsun. Burada her kanal için ayrı bir 3x3'lük 
filtre matrisi kullanılacaktır. Bu durumda evrişim katmanında eğitilebilir 
parametrelerin sayısı 3 * 3 * 3 + 1 = 28 tane olacaktır. Eğer biz bu örnekte padding 
kullanmıyorsak ve stride değeri de 1 ise (yani kaydırma birer birer yapılıyorsa) 
bu durumda elde edilen hedef resim 8x8x1'lik olacaktır. Uygulanan evrişim sonucunda 
resmin RGB olmaktan çıkıp adreta gray scale hale getirildiğine dikkat ediniz.

---------------------------------------------------------------------------------
Aslında uygulamada resim tek bir filtreye de sokulmamaktadır. Birden fazla filtreye 
sokulmaktadır. Örneğin biz bir resimde 3x3'lük 32 farklı filtre kullanabiliriz. 
Bu durumda ağın bu 32 filtreyi de belirlemesini isteyebiliriz. Filtre sayısı 
artırıldıkça her filtre resmin bir yönünü keşfedeceğinden resmin anlaşılması da 
iyileştirilmektedir. Şimdi 10x10'luk resmi 3x3'lük filtre kullanarak padding 
uygulamadan 32 farklı filtreye sokmuş olalım. Biz bu resmi tek bir filtreye 
soktuğumuzda 8x8x1'lik bir hedef resim elde ediyorduk. İşte 32 farklı filtreye 
soktuğumuzda her filtreden 8x8x1'lik bir resim elde edileceğinegöre toplamda 
8x8x32'lik bir resim elde edilmiş olur. 

Şimdi de 32 filtre kullandığımız durumda 10x10x3'lük RGB resim için eğitilebilir 
parametrelerin sayısını hesaplayalım. Bir tane filtre için yukarıda toplam 
eğitilebilir parametrelerin sayısını 3 * 3 * 3 + 1 olarak hesaplamıştık. Bu 
filtrelerden 32 tane olduğuna göre toplam eğitilebilir parametrelerin sayısı 
32 * (3 * 3 * 3 + 1) = 32 * 27 + 32 = 32 * 28 = 896 tane olacaktır. 

Pekiyi 10x10'luk resmimiz gri tonlamalı olsaydı 32 filtre ve 3x3'lük kernel için 
toplam eğitilebilir parametrelerin sayısı ne olurdu? Bu durumda 3x3'lük toplam 
32 farklı filtre kullanılacağı için ve her filtrede bir tane bias değeri söz 
konusu olacağı için toplam eğitilebilir parametrelerin sayısı da 
32 * (3 * 3 + 1) = 32 * 10 = 320 tane olacaktır. 

---------------------------------------------------------------------------------
"""







